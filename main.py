import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import warnings
from collections import defaultdict
from tqdm import tqdm
from torch.cuda.amp import GradScaler
import itertools 

from config import *
from models import *
from channel import *
from training import *
from utils import *

def main():

    optimize_gpu_memory()  
    # --- Device and Performance Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Enable TF32 for Ampere+ GPUs (Significant speedup)
    if torch.cuda.is_available() and hasattr(torch.backends.cuda.matmul, 'allow_tf_32'):
        torch.backends.cuda.matmul.allow_tf_32 = True
        torch.backends.cudnn.allow_tf_32 = True
        print("   -> TF32 acceleration enabled.")
    
    # Enable cuDNN benchmark (Speedup for fixed input sizes)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("   -> cuDNN Benchmark enabled.")

    # --- Model Initialization ---
    scout = EnhancedScoutAI().to(device)
    transmitter = EnhancedTransmitterAI().to(device)
    receiver = EnhancedReceiverAI(use_neural_equalizer=True).to(device)  # Enable neural equalizer
    ae_decoder = ChannelAutoencoderDecoder().to(device)   
    critic = Critic().to(device)
    manager = ManagerAI().to(device)   
    
    
    # Suppress warnings
    warnings.filterwarnings("ignore", message="ComplexHalf support is experimental")
    scaler = GradScaler()    
    
    # --- Checkpoint Loading ---
    start_epoch = 0
    best_loss = float('inf')
    optimizer = None
    checkpoint = {}
    
    model_dir = 'saved_6g_grand_master'
    os.makedirs(model_dir, exist_ok=True)    
    checkpoint_path = os.path.join(model_dir, 'checkpoint_latest.pth')
    
    phase4_path = os.path.join(model_dir, 'phase4a.pth')     
    checkpoint_loaded = False
    if os.path.exists(checkpoint_path):
        print(f"Resuming from latest checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        scout.load_state_dict(checkpoint['scout_state_dict'])
        transmitter.load_state_dict(checkpoint['transmitter_state_dict'])
        receiver.load_state_dict(checkpoint['receiver_state_dict'], strict=False)
      
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        ae_decoder.load_state_dict(checkpoint.get('ae_decoder_state_dict', {}), strict=False)
        critic.load_state_dict(checkpoint.get('critic_state_dict', {}), strict=False)
        manager.load_state_dict(checkpoint.get('manager_state_dict', {}), strict=False) 
        checkpoint_loaded = True
        # This overrides the "Zombie" value (0.25) from the checkpoint files
        print("   -> 🔧 Forcing Receiver Input Scalers to 1.0 (Overriding Checkpoint)")
        for i, stabilizer in enumerate(receiver.input_stabilizers):
            # Only reset the High Rates (Index 4+) where we use SimpleScaler
            if i >= 4: 
                nn.init.constant_(stabilizer.scale, 1.0)
                print(f"      -> Reset Rate {RATES[i]}bps Scaler to 1.0")    
        # We manually set the mixing weight to 0.1 to allow gradients to flow.
        print("   -> 🔌 Jumpstarting Rate 8 Neural Equalizer...")
        
        # Rate 8 is Index 7. Corrector Index = 7 - 3 = 4.
        target_corrector = receiver.equalization_correctors[4]
        
        with torch.no_grad():
            # attention_weights[0] is Main Signal (Keep ~1.0)
            # attention_weights[1] is Neural Correction (Force to 0.1)
            target_corrector.attention_weights[1].fill_(0.1) 
            
        print("      -> Set Correction Weight to 0.1 (Was 0.0)")                
    elif os.path.exists(phase4_path):
        print(f"Resuming from phase4 checkpoint: {phase4_path}")
        checkpoint = torch.load(phase4_path)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        scout.load_state_dict(checkpoint['scout_state_dict'])
        transmitter.load_state_dict(checkpoint['transmitter_state_dict'])
        receiver.load_state_dict(checkpoint['receiver_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        ae_decoder.load_state_dict(checkpoint.get('ae_decoder_state_dict', {}), strict=False)
        critic.load_state_dict(checkpoint.get('critic_state_dict', {}), strict=False)
        manager.load_state_dict(checkpoint.get('manager_state_dict', {}), strict=False)  
        checkpoint_loaded = True
    else:
        print("No checkpoint found. Starting fresh training.")
        start_epoch = 0  
    # Reset manager if starting RL phase
    if start_epoch >= PHASE_4A_END:
        print("\n" + "="*60)
        print(f"  SURGICAL RESET DETECTED AT EPOCH {start_epoch}.")
        print("  -> Resuming into RL Phase. Re-initializing rate head.")
        print("="*60 + "\n")
        
        transmitter.rate_head_csi_embedding.apply(weights_init)
        transmitter.rate_head_noise_embedding.apply(weights_init)
        transmitter.rate_head_transformer.apply(weights_init)
        transmitter.rate_head_decoder.apply(weights_init)

        if 'optimizer_state_dict' in checkpoint:
            checkpoint['optimizer_state_dict'] = None

    # ================================================================
    #                        TRAINING START
    # ================================================================
    print("\n🚀 Starting Training: GRAND MASTER EDITION with Neural Equalizer...")
    receiver_training_history = {
        'best_accuracy': 50.0,  
        'epoch_progress': [],
        'current_stage': 'initial'
    } 
   # Initialize with default large buffer size (batch size)
    main_channel = RealisticMIMOChannel(
        batch_size=BATCH_SIZE, 
        device=device, 
        distance_meters=100,
        fixed_snr_db=20.0
    )   
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        epoch_start_time = time.time()
        
        # Set Gumbel softmax temperature
        current_tau = get_gumbel_softmax_tau(epoch)       
        if hasattr(transmitter, '_orig_mod'):
            transmitter._orig_mod.set_tau(current_tau)
        else:
            transmitter.set_tau(current_tau)        
        
        # --- Optimizer and Scheduler Setup ---
        is_major_boundary = (
            optimizer is None or
            epoch == PHASE_1_END or
            epoch == PHASE_2_END or
            epoch == PHASE_3_END or
            epoch == PHASE_4A_END or
            epoch == PHASE_4_END
        )
        
        is_new_specialist_lesson = False
        if PHASE_3_END < epoch < PHASE_4A_END:
            epochs_into_phase4a = epoch - PHASE_3_END
            if epochs_into_phase4a in lesson_boundaries:
                is_new_specialist_lesson = True
                current_lesson_index = lesson_boundaries.index(epochs_into_phase4a)
                print(f"\n>>> NEW SPECIALIST LESSON: {RATES[current_lesson_index]} BPS <<<")

        if is_major_boundary or is_new_specialist_lesson:
            best_loss = float('inf')

            # Freeze all models by default
            scout.requires_grad_(False).eval()
            transmitter.requires_grad_(False).eval()
            receiver.requires_grad_(False).eval()
            ae_decoder.requires_grad_(False).eval()
            manager.requires_grad_(False).eval()
            critic.requires_grad_(False).eval()

            # Apply phase-specific training
            if epoch < PHASE_1_END:
                print("   -> PHASE 1: Constellation Foundation")
                transmitter.constellation_encoder.requires_grad_(True).train()

            elif epoch < PHASE_2_END:
                print("   -> PHASE 2: Receiver Apprentice")
                receiver.requires_grad_(True).train()

            elif epoch < PHASE_3_END:
                print("   -> PHASE 3: Precoder Apprentice")
                scout.requires_grad_(True).train()
                transmitter.requires_grad_(True).train()
                ae_decoder.requires_grad_(True).train()
                transmitter.constellation_encoder.requires_grad_(False).eval()

                if epoch < PHASE_3_LESSON_1_END:  
                    print("   -> Phase 3A")
                    receiver.requires_grad_(False).eval()
                    
                elif epoch < PHASE_3_LESSON_2_END:  # 360-510  
                    print("   -> Phase 3B")
                    receiver.requires_grad_(True).train()
                    for name, param in receiver.named_parameters():
                        if 'equalization' in name or 'correctors' in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False     
                else:  # 510-610
                    print("   -> Phase 3C")
                    receiver.requires_grad_(True).train()            
            elif epoch < PHASE_4A_END:
                epochs_into_phase4a = epoch - PHASE_3_END
                current_lesson_index = 0
                for i in range(len(lesson_boundaries)):
                    if epochs_into_phase4a < lesson_boundaries[i]:
                        current_lesson_index = i - 1
                        break
                else:
                    current_lesson_index = len(lesson_boundaries) - 1               
                
                print(f"   -> PHASE 4A: Training {RATES[current_lesson_index]}bps Specialist")
                transmitter.precoders[current_lesson_index].requires_grad_(True).train()
                receiver.decoders[current_lesson_index].requires_grad_(True).train()
                transmitter.constellation_encoder.requires_grad_(False).eval()
                for name, param in receiver.named_parameters():
                    if 'equalization' in name or 'correctors' in name:

                        corrector_index = current_lesson_index  
                        if f'equalization_correctors.{corrector_index}' in name:
                            param.requires_grad = True  
                            print(f"active: {name}")
                        else:
                            param.requires_grad = False   
                print("all equalization:")
                for name, param in receiver.named_parameters():
                    if 'equalization' in name or 'correctors' in name:
                        print(f"  {name} - requires_grad: {param.requires_grad}")

                print(f"finding: rate_specific_equalizers.{current_lesson_index}")                              
            else:
                print("   -> PHASE 4B: Manager RL")
                manager.requires_grad_(True).train()
                critic.requires_grad_(True).train()
            
            # Optimizer setup
            base_lr = 1.0
            all_trainable_params = list(filter(lambda p: p.requires_grad,
                                   list(scout.parameters()) + list(transmitter.parameters()) +
                                   list(receiver.parameters()) + list(ae_decoder.parameters()) +
                                   list(manager.parameters()) + list(critic.parameters())))

            if epoch >= PHASE_2_END:
                print("   -> Configuring optimizer with per-group learning rates.")
                receiver_param_ids = {id(p) for p in receiver.parameters()}
                receiver_params_list = [p for p in all_trainable_params if id(p) in receiver_param_ids]
                other_params_list = [p for p in all_trainable_params if id(p) not in receiver_param_ids]

                trainable_params_groups = [
                    {'params': receiver_params_list, 'name': 'receiver', 'lr': base_lr},
                    {'params': other_params_list, 'name': 'other', 'lr': base_lr}
                ]
            elif epoch >= PHASE_1_END:  
                print("   -> Configuring optimizer with per-group learning rates.")
                
                equalizer_param_ids = set()
                for name, param in receiver.named_parameters():
                    if 'equalization' in name or 'correctors' in name:
                        equalizer_param_ids.add(id(param))
                
                receiver_params_list = [p for p in all_trainable_params if id(p) not in equalizer_param_ids]
                equalizer_params_list = [p for p in all_trainable_params if id(p) in equalizer_param_ids]

                trainable_params_groups = [
                    {'params': receiver_params_list, 'name': 'receiver', 'lr': base_lr},
                    {'params': equalizer_params_list, 'name': 'equalizer', 'lr': base_lr * 0.1},  
                ]
               
            else:
                trainable_params_groups = [{'params': all_trainable_params, 'name': 'other', 'lr': base_lr}]
                
            print("\n   Training components:")
            all_models = {'scout': scout, 'transmitter': transmitter, 'receiver': receiver, 'ae_decoder': ae_decoder}
            for model_name, model in all_models.items():
                for param_name, p in model.named_parameters():
                    if p.requires_grad:
                        print(f"     - {model_name}.{param_name}")

            optimizer = optim.AdamW(trainable_params_groups, lr=1.0)
            
            is_resuming_from_checkpoint = ('optimizer_state_dict' in checkpoint and start_epoch == epoch + 1)
            
            if is_resuming_from_checkpoint and not is_new_specialist_lesson:
                print("   -> Loading optimizer state from checkpoint.")
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("   -> WARNING: Could not load optimizer state. Starting fresh.")
            else:
                print("   -> New phase/lesson. Creating fresh optimizer.")

            for group in optimizer.param_groups: 
                group.setdefault('initial_lr', group['lr'])
            lambda_list = [lambda e, name=group['name']: get_learning_rate_multipliers_2(e)[name] for group in optimizer.param_groups]
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_list, last_epoch=epoch - 1)
            
            print("\n" + "=" * 60 + "\n")

        if epoch < PHASE_1_END:   
            phase_desc = "Phase 1: Unified Foundation"
        elif epoch < PHASE_2_END: 
            phase_desc = "Phase 2: Receiver Apprentice"
        elif epoch < PHASE_3_END: 
            phase_desc = "Phase 3: Precoder Apprentice"
        elif epoch < PHASE_4A_END: 
            phase_desc = "Phase 4A: Team Synergy"            
        else:                     
            phase_desc = "Phase 4B: Manager RL"

        scout.train(); transmitter.train(); receiver.train()
        epoch_loss = 0.0
        epoch_losses = defaultdict(float)

        pbar = tqdm(range(BATCHES_PER_EPOCH), desc=f"--- Epoch {epoch+1}/{TOTAL_EPOCHS} | {phase_desc} ---", dynamic_ncols=True)
        if epoch >= PHASE_3_END and epoch < PHASE_4A_END:
            current_moe_index, moe_progress = get_current_moe_progress(epoch)
            current_bps = RATES[current_moe_index]
            print(f"📊 MOE: {current_bps}bps, process: {moe_progress*100:.1f}%")
            
            receiver_lr = get_moe_adaptive_learning_rate(current_moe_index, moe_progress)
            equalizer_lr = get_moe_equalizer_learning_rate(current_moe_index, moe_progress)
            print(f"📈 LR - receiver: {receiver_lr:.2e}, equalizer: {equalizer_lr:.2e}")            
        for batch_idx in pbar:
            # --- Data Acquisition ---
            if epoch < PHASE_1_END:  # Phase 1: Constellation Foundation
                pbar.set_description(f"--- Epoch {epoch+1}/{TOTAL_EPOCHS} | Phase 1: Constellation Foundation ---")
                mode, force_rate = get_training_mode(epoch, batch_idx, BATCHES_PER_EPOCH)
                channel_gpu = None

                original_data = torch.randint(0, 2, (BATCH_SIZE, MAX_RATE_BPS), device=device, dtype=torch.float32)

                true_csi = torch.zeros(BATCH_SIZE, CSI_VECTOR_SIZE, device=device)
                noise_power_tensor = torch.zeros(BATCH_SIZE, device=device)            
            elif epoch < PHASE_2_LESSON_0_END:
                pbar.set_description(f"--- Epoch {epoch+1}/{TOTAL_EPOCHS} | Phase 2, Lesson 0 (Infant Stage) ---")
                mode, force_rate = get_training_mode(epoch, batch_idx, BATCHES_PER_EPOCH)
                original_data = torch.randint(0, 2, (BATCH_SIZE, MAX_RATE_BPS), device=device, dtype=torch.float32)
                true_csi = torch.zeros(BATCH_SIZE, CSI_VECTOR_SIZE, device=device)
                noise_power_tensor = torch.zeros(BATCH_SIZE, device=device)                
            elif epoch < PHASE_3_END:
                pbar.set_description(f"--- Epoch {epoch+1}/{TOTAL_EPOCHS} | {phase_desc} ---")
                apply_channel_curriculum(main_channel, epoch)
                channel_gpu = main_channel
                original_data = torch.randint(0, 2, (BATCH_SIZE, MAX_RATE_BPS), device=device, dtype=torch.float32)
                true_csi = channel_gpu.get_csi()
                mode, force_rate = get_training_mode(epoch, batch_idx, BATCHES_PER_EPOCH)
                noise_power_tensor = torch.full((BATCH_SIZE,), channel_gpu.noise_power_watts, device=device)                
            elif epoch < PHASE_4A_END:
                pbar.set_description(f"--- Epoch {epoch+1}/{TOTAL_EPOCHS} | {phase_desc} ---")

                force_rate = apply_moe_synergy_curriculum(main_channel, epoch)
                channel_gpu = main_channel                

                if force_rate == 0 and best_loss > 0.65:
                    channel_gpu = RealisticMIMOChannel(BATCH_SIZE, device, use_fading=True, distance_meters=1000, fixed_snr_db=15.0)           
                original_data = torch.randint(0, 2, (BATCH_SIZE, MAX_RATE_BPS), device=device, dtype=torch.float32)
                true_csi = channel_gpu.get_csi()
                mode = 'supervised'
                noise_power_tensor = torch.full((BATCH_SIZE,), channel_gpu.noise_power_watts, device=device)  
            else:
                pbar.set_description(f"--- Epoch {epoch+1}/{TOTAL_EPOCHS} | {phase_desc} ---")
                apply_channel_curriculum(main_channel, epoch)
                channel_gpu = main_channel
                original_data = torch.randint(0, 2, (BATCH_SIZE, MAX_RATE_BPS), device=device, dtype=torch.float32)
                true_csi = channel_gpu.get_csi()
                mode, force_rate = get_training_mode(epoch, batch_idx, BATCHES_PER_EPOCH)
                noise_power_tensor = torch.full((BATCH_SIZE,), channel_gpu.noise_power_watts, device=device)                
            
            optimizer.zero_grad(set_to_none=True)
            estimated_csi = scout(true_csi)            
            use_autocast = True
            if mode == 'supervised' and force_rate is not None and force_rate >= 4:
                use_autocast = False

            # --- Forward Pass ---
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                if epoch < PHASE_1_END:

                    total_constellation_loss = torch.tensor(0.0, device=device)
                    loss_breakdown = {}

                    for rate_index, bps in enumerate(RATES):
                        num_points = 2**bps
                        bits = torch.tensor(list(itertools.product([0, 1], repeat=bps)), device=device)
                        data_chunk = torch.zeros(num_points, MAX_RATE_BPS, device=device)
                        data_chunk[:, :bps] = bits.float()
                        rate_one_hot_geom = F.one_hot(torch.full((num_points,), rate_index, device=device), len(RATES)).float()
                        encoded_points = transmitter.constellation_encoder(data_chunk, rate_one_hot_geom)
                        
                        specialist_loss_dict = PowerTargetingConstellationLoss(encoded_points, rate_index, epoch)
                        total_constellation_loss += specialist_loss_dict['total']
                        
                        loss_breakdown[f'loss_{bps}bps'] = specialist_loss_dict['total']
                        loss_breakdown[f'power_loss_{bps}bps'] = specialist_loss_dict['power_loss']
                        loss_breakdown[f'origin_{bps}bps'] = specialist_loss_dict['origin_loss']
                        loss_breakdown[f'repulsion_{bps}bps'] = specialist_loss_dict['repulsion_loss']
                        loss_breakdown[f'avg_power_{bps}bps'] = specialist_loss_dict['avg_power']

                    loss = total_constellation_loss
                    loss_dict = {'total': loss, **loss_breakdown}  
                    # Add the debug print
                    if batch_idx % 100 == 0:
                        print(f"\n--- Phase 1 Debug (Epoch {epoch+1}, Batch {batch_idx}) ---")
                        
                        for key, value in loss_breakdown.items():
                            print(f"   - {key}: {value:.4f}")                     
                else:
                    if mode == 'supervised':
                        # 1. Transmitter
                        if epoch < PHASE_2_END:  
                            with torch.no_grad(): 
                                tx_signal_complex, encoded_symbols_real, rate_one_hot, original_latent_vec, used_precoder  = transmitter(
                                    original_data, estimated_csi, noise_power_tensor, force_rate_index=force_rate
                                )
                        else:

                            tx_signal_complex, encoded_symbols_real, rate_one_hot, original_latent_vec ,used_precoder= transmitter(
                                original_data, estimated_csi, noise_power_tensor, force_rate_index=force_rate
                            )
                     
                        current_power = torch.mean(torch.abs(tx_signal_complex)**2)
                        scale = torch.sqrt(1.0 / (current_power + 1e-9))
                        tx_signal_normalized = tx_signal_complex * scale 
                        
                        if epoch < PHASE_2_LESSON_0_END:
                            if encoded_symbols_real.shape[1] == 2:
                                real_part = encoded_symbols_real[:, 0].unsqueeze(1).repeat(1, 4)
                                imag_part = encoded_symbols_real[:, 1].unsqueeze(1).repeat(1, 4)
                                complex_signal = torch.complex(real_part, imag_part)
                                equalized_real = torch.view_as_real(complex_signal).view(encoded_symbols_real.shape[0], -1)
                            elif encoded_symbols_real.shape[1] == 8:
                                equalized_real = encoded_symbols_real
                            else:
                                batch_size = encoded_symbols_real.shape[0]
                                equalized_real = torch.zeros(batch_size, 8, device=device)
                                min_dim = min(8, encoded_symbols_real.shape[1])
                                equalized_real[:, :min_dim] = encoded_symbols_real[:, :min_dim]
                            
                            reconstructed_latent_vec = None

                        elif epoch < PHASE_3_END:
                            rx_noisy = channel_gpu.apply(tx_signal_normalized, add_noise=True)

                            rx_power = torch.mean(torch.abs(rx_noisy)**2, dim=1, keepdim=True)
                            agc_scale = torch.sqrt(1.0 / (rx_power + 1e-9))
                            rx_noisy = rx_noisy * agc_scale
                            
                            if epoch < PHASE_2_END:

                                with torch.no_grad():
                                    precoder_to_use = get_svd_precoder_power_iter(channel_gpu.H_fading)
                            else:

                                precoder_to_use = used_precoder 

                            with torch.amp.autocast(device_type='cuda', enabled=False):
                                rx_noisy_c64 = rx_noisy.to(torch.complex64)
                                h_fading_c64 = channel_gpu.H_fading.to(torch.complex64)
                                precoder_c64 = precoder_to_use.to(torch.complex64)

                                h_eff_complex = torch.bmm(h_fading_c64, precoder_c64.unsqueeze(-1)).squeeze(-1)
                                
                                current_snr_db = getattr(channel_gpu, 'actual_snr_db', 10.0)
                                snr_tensor = torch.full((rx_noisy.shape[0], 1), current_snr_db, device=device)

                                equalized_complex = adaptive_mmse_equalization(rx_noisy_c64, h_eff_complex, snr_tensor)

                                equalized_real = torch.view_as_real(equalized_complex).view(equalized_complex.shape[0], -1)

                                equalized_real = torch.clamp(equalized_real, -10.0, 10.0)

                            if epoch < PHASE_2_END:
                                reconstructed_latent_vec = None
                            else:
                                reconstructed_latent_vec = ae_decoder(equalized_real, estimated_csi, noise_power_tensor, rate_one_hot=rate_one_hot)
                        
                        elif epoch < PHASE_4A_END: 
                            
                            rx_noisy = channel_gpu.apply(tx_signal_normalized, add_noise=True)
                            
                            rx_power = torch.mean(torch.abs(rx_noisy)**2, dim=1, keepdim=True)
                            agc_scale = torch.sqrt(1.0 / (rx_power + 1e-9))
                            rx_noisy = rx_noisy * agc_scale
                            
                            current_snr_db = getattr(channel_gpu, 'actual_snr_db', 10.0)
                            
                            snr_db_tensor = torch.full((rx_noisy.shape[0], 1), current_snr_db, device=device) 
                            
                            precoder_to_use = used_precoder
                            with torch.amp.autocast(device_type='cuda', enabled=False):
                                rx_noisy_c64 = rx_noisy.to(torch.complex64)
                                h_fading_c64 = channel_gpu.H_fading.to(torch.complex64)
                                precoder_c64 = precoder_to_use.to(torch.complex64)
                                
                                h_eff_complex = torch.bmm(h_fading_c64, precoder_c64.unsqueeze(-1)).squeeze(-1)
                                equalized_complex = adaptive_mmse_equalization(rx_noisy_c64, h_eff_complex, snr_db_tensor)
                                
                                equalized_real = torch.view_as_real(equalized_complex).view(equalized_complex.shape[0], -1)
                                equalized_real = torch.clamp(equalized_real, -10.0, 10.0)
                                
                            reconstructed_latent_vec = ae_decoder(equalized_real, estimated_csi, noise_power_tensor, rate_one_hot=rate_one_hot)                            
                        else: 
                            equalized_real = torch.zeros_like(encoded_symbols_real)
                            reconstructed_latent_vec = None
                            reconstructed_data_logits = torch.zeros_like(original_data)
                       
                        reconstructed_data_logits , correction_mags= receiver(equalized_real, estimated_csi, noise_power_tensor, rate_one_hot=rate_one_hot)                       

                        loss_dict = rate_aware_supervised_loss(
                            reconstructed_data_logits, original_data, rate_one_hot,
                            estimated_csi, true_csi, encoded_symbols_real, epoch,
                            reconstructed_latent_vec=reconstructed_latent_vec,
                            original_latent_vec=original_latent_vec,neural_correction_magnitude=correction_mags
                        )
                        loss = loss_dict['total']
                    elif mode == 'supervised_rl':
                        random_dist = np.random.uniform(50, 1000)
                        temp_channel = RealisticMIMOChannel(BATCH_SIZE, device, distance_meters=random_dist, verbose=False)
                        
                        snr_db_context = torch.full((BATCH_SIZE,), temp_channel.actual_snr_db, device=device)
                        
                        snr_feature = (snr_db_context.unsqueeze(1) - 10.0) / 20.0

                        rate_logits = manager(estimated_csi.clone().detach(), snr_feature.clone().detach())
                        
                        with torch.no_grad():
                            shannon_capacity = get_shannon_capacity(snr_db_context)
                            safety_margin = 1.0 # Tighter margin since we have better physics now
                            wise_target_capacity = shannon_capacity - safety_margin
                            target_indices = torch.zeros_like(shannon_capacity, dtype=torch.long)
                            
                            for i, rate in enumerate(RATES):
                                target_indices[wise_target_capacity >= rate] = i
                        
                        loss = F.cross_entropy(rate_logits, target_indices)
                        loss_dict = {'total': loss, 'supervised_rl_loss': loss}                     

                    # ---------------------------------------------------------------------
                    # MODE: REINFORCEMENT LEARNING (PPO / Actor-Critic)
                    # ---------------------------------------------------------------------
                    elif mode == 'reinforcement':
                        rl_random_distance = np.random.uniform(low=50, high=1500)

                        rl_channel_gpu = RealisticMIMOChannel(
                            BATCH_SIZE, 
                            device, 
                            use_fading=True, 
                            distance_meters=rl_random_distance,
                            verbose=False
                        )
                        

                        true_csi = rl_channel_gpu.get_csi()
                        estimated_csi = scout(true_csi) # Pass through Scout (Estimator)
                        
                        snr_db_real = rl_channel_gpu.actual_snr_db
                        snr_db_context = torch.full((BATCH_SIZE,), snr_db_real, device=device)
                        snr_feature = (snr_db_context.unsqueeze(1) - 10.0) / 20.0

                        rate_logits = manager(estimated_csi.clone().detach(), snr_feature.clone().detach())
                        rate_logits = enhanced_exploration(rate_logits, snr_db_context, epoch)
                        rate_probs = F.softmax(rate_logits, dim=1)                        
                        
                        predicted_values_raw = critic(estimated_csi, snr_feature)
                        predicted_values = predicted_values_raw.squeeze(-1)
                        
                        the_action_one_hot = F.gumbel_softmax(rate_logits, tau=current_tau, hard=True)
                        chosen_rate_indices = torch.argmax(the_action_one_hot, dim=1)
                        
                        with torch.no_grad():
                            transmitter.eval(); receiver.eval()
                            
                            shuffle_indices = torch.randperm(original_data.shape[0])
                            shuffled_data = original_data[shuffle_indices]
                            
                            tx_signal_complex, encoded_symbols_real, _, _,used_precoder  = transmitter(
                                shuffled_data, estimated_csi, 
                                torch.zeros(BATCH_SIZE, device=device), # dummy noise tensor
                                override_rate_one_hot=the_action_one_hot
                            )
                            
                            sample_power = torch.mean(torch.abs(tx_signal_complex)**2, dim=1, keepdim=True)
                            scale = torch.sqrt(1.0 / (sample_power + 1e-9))
                            tx_signal_normalized = tx_signal_complex * scale
                            
                            rx_noisy = rl_channel_gpu.apply(tx_signal_normalized, add_noise=True)
                            
                            rx_power = torch.mean(torch.abs(rx_noisy)**2, dim=1, keepdim=True)
                            agc_scale = torch.sqrt(1.0 / (rx_power + 1e-9))
                            rx_noisy = rx_noisy * agc_scale
                            with torch.amp.autocast(device_type='cuda', enabled=False):
                                rx_noisy_c64 = rx_noisy.to(torch.complex64)
                                h_fading_c64 = rl_channel_gpu.H_fading.to(torch.complex64)
                                precoder_c64 = used_precoder.to(torch.complex64) # <--- Clean and consistent

                                h_eff_complex = torch.bmm(h_fading_c64, precoder_c64.unsqueeze(-1)).squeeze(-1)
                                
                                snr_tensor = torch.full((BATCH_SIZE, 1), snr_db_real, device=device)
                                
                                equalized_complex = adaptive_mmse_equalization(rx_noisy_c64, h_eff_complex, snr_tensor)
                                equalized_real = torch.view_as_real(equalized_complex).view(BATCH_SIZE, -1)
                                equalized_real = torch.clamp(equalized_real, -30.0, 30.0)

                            noise_dummy = torch.zeros(BATCH_SIZE, device=device)
                            reconstructed_data_logits , correction_mags = receiver(
                                equalized_real, estimated_csi, noise_dummy, rate_one_hot=the_action_one_hot
                            )
                            
                            transmitter.train(); receiver.train()

                        predicted_bits = (torch.sigmoid(reconstructed_data_logits) > 0.5).float()
                        
                        bps_per_item = torch.tensor([RATES[i] for i in chosen_rate_indices], device=device)
                        
                        mask = torch.arange(MAX_RATE_BPS, device=device).unsqueeze(0) < bps_per_item.unsqueeze(1)
                        
                        bit_errors = torch.abs(predicted_bits - shuffled_data) * mask
                        num_errors = torch.sum(bit_errors, dim=1)
                        
                        ber_per_sample = num_errors / (bps_per_item + 1e-9)
                        achieved_throughput = bps_per_item * (1.0 - ber_per_sample)
                        
                        with torch.no_grad():
                            theoretical_max_bps = get_shannon_capacity(snr_db_context)
                            
                            actual_rewards = systematic_balanced_reward_fixed(
                                snr_db_context, 
                                chosen_rate_indices, 
                                achieved_throughput, 
                                theoretical_max_bps,
                                device
                            )                           
                            
                        performance_gap = achieved_throughput - theoretical_max_bps
                        
                        loss = actor_critic_loss(
                            rate_probs, the_action_one_hot,
                            predicted_values,
                            actual_rewards,
                            batch_idx, epoch, snr_db_context, theoretical_max_bps, achieved_throughput, performance_gap
                        )                        
                        loss_dict = {'total': loss, 'rl_prof': loss}

            if use_autocast:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for group in optimizer.param_groups for p in group['params']), max_norm=5.0)#change from 1.0
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_((p for group in optimizer.param_groups for p in group['params']), max_norm=5.0)#change from 1.0
                optimizer.step()  

            epoch_loss += loss.item()
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Tot': f"{loss.item():.3f}",
                    'BCE': f"{loss_dict.get('bce', 0):.3f}",   # Bit Error Loss (The most important one!)
                    'Sct': f"{loss_dict.get('scout', 0):.3f}", # Channel Est Loss
                    'AE':  f"{loss_dict.get('autoencoder', 0):.3f}" # Latent Space Loss
                })
            
            if batch_idx == 0 and (epoch+1)%10==0 and force_rate is not None and force_rate>2: # Run once per epoch
                deep_debug_signal_path(epoch, transmitter, receiver, device, target_rate_idx=force_rate)      
        # --- End of Epoch Processing ---
        scheduler.step()
        avg_total_loss = epoch_losses['total'] / BATCHES_PER_EPOCH        
        avg_bce = epoch_losses['bce'] / BATCHES_PER_EPOCH
        avg_scout = epoch_losses['scout'] / BATCHES_PER_EPOCH
        avg_ae = epoch_losses['autoencoder'] / BATCHES_PER_EPOCH       
        epoch_time = time.time() - epoch_start_time
        
        lr_receiver = scheduler.get_last_lr()[0]
        lr_other = scheduler.get_last_lr()[-1]
        log_str = (
            f"  📉 Loss: {avg_total_loss:.4f} "
            f"(BCE: {avg_bce:.4f} | Sct: {avg_scout:.4f} | AE: {avg_ae:.4f}) | "
            f"Best: {best_loss:.4f} | "
            f"LR: {lr_receiver:.1e} | Time: {epoch_time:.0f}s"
        )        
        print(log_str)

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            checkpoint = {
                'epoch': epoch,
                'best_loss': best_loss,
                'scout_state_dict': scout.state_dict(),
                'transmitter_state_dict': transmitter.state_dict(),
                'receiver_state_dict': receiver.state_dict(),
                'ae_decoder_state_dict': ae_decoder.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'manager_state_dict': manager.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(model_dir, 'model_best.pth'))
            print(f"  💾 model best saved at epoch {epoch}")


        if (epoch +1 ) % 10 == 0:
            if epoch < PHASE_1_END:
                check_constellation_integrity(epoch, transmitter, receiver, device)
            elif epoch < PHASE_2_END:
               receiver_training_history = check_receiver_robustness_phase2_dyn(
                    epoch, scout, transmitter, receiver, device, receiver_training_history
                )
                
               receiver_training_history = check_receiver_fundamentals_phase2_dyn(
                    epoch, transmitter, receiver, device, receiver_training_history
                )               
            else:
                moe_results, avg_throughput = check_moe_bps_performance(
                    epoch, scout, transmitter, receiver, device, phase_desc
                )
        if (epoch + 1) % 5 == 0:  # Check every 5 epochs
            debug_equalizer_weights(epoch, receiver)
        if epoch + 1 == PHASE_1_END:
            path = os.path.join(model_dir, 'constellation_pretrain.pth')
            print(f"  💾 End of Foundation Phase. Saving pre-trained model to '{path}'...")
            torch.save({
                'epoch': epoch, 'best_loss': best_loss,
                'scout_state_dict': scout.state_dict(),
                'transmitter_state_dict': transmitter.state_dict(),
                'receiver_state_dict': receiver.state_dict(),
                'critic_state_dict': critic.state_dict(), 
                'manager_state_dict': manager.state_dict(),                
                'optimizer_state_dict': optimizer.state_dict(),            
                'ae_decoder_state_dict':ae_decoder.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, path)
            check_constellation_integrity(epoch, transmitter, receiver, device) # Modify the function to return key metrics
         
        if epoch + 1 == PHASE_2_END:
            path = os.path.join(model_dir, 'phase2.pth')
            print(f"  💾 End of Foundation Phase. Saving phase2 model to '{path}'...")
            torch.save({
                'epoch': epoch, 'best_loss': best_loss,
                'scout_state_dict': scout.state_dict(),
                'transmitter_state_dict': transmitter.state_dict(),
                'receiver_state_dict': receiver.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'manager_state_dict': manager.state_dict(),                
                'optimizer_state_dict': optimizer.state_dict(),            
                'ae_decoder_state_dict':ae_decoder.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, path)
        elif epoch + 1 == PHASE_3_END:
            path = os.path.join(model_dir, 'phase3.pth')
            print(f"  💾 End of Foundation Phase. Saving phase3 model to '{path}'...")
            torch.save({
                'epoch': epoch, 'best_loss': best_loss,
                'scout_state_dict': scout.state_dict(),
                'transmitter_state_dict': transmitter.state_dict(),
                'receiver_state_dict': receiver.state_dict(),
                'critic_state_dict': critic.state_dict(), 
                'manager_state_dict': manager.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ae_decoder_state_dict':ae_decoder.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, path)       
        elif epoch + 1 == PHASE_4A_END:
            path = os.path.join(model_dir, 'phase4a.pth')
            print(f"  💾 End of Foundation Phase. Saving phase4A model to '{path}'...")
            torch.save({
                'epoch': epoch, 'best_loss': best_loss,
                'scout_state_dict': scout.state_dict(),
                'transmitter_state_dict': transmitter.state_dict(),
                'receiver_state_dict': receiver.state_dict(),
                'critic_state_dict': critic.state_dict(), 
                'manager_state_dict': manager.state_dict(),                
                'optimizer_state_dict': optimizer.state_dict(),
                'ae_decoder_state_dict':ae_decoder.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, path)                 
        torch.save({
            'epoch': epoch, 'best_loss': best_loss,
            'scout_state_dict': scout.state_dict(),
            'transmitter_state_dict': transmitter.state_dict(),
            'receiver_state_dict': receiver.state_dict(),
            'critic_state_dict': critic.state_dict(), 
            'manager_state_dict': manager.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ae_decoder_state_dict':ae_decoder.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }, checkpoint_path)


    print("\n🎉 Training completed!")

if __name__ == "__main__":
    main()