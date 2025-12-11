import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from tqdm import tqdm
from config import *

def get_shannon_capacity(snr_db_tensor):
    """Calculate theoretical channel capacity using Shannon-Hartley theorem"""
    if not isinstance(snr_db_tensor, torch.Tensor):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        snr_db_tensor = torch.tensor([snr_db_tensor], device=device, dtype=torch.float32)

    snr_linear = 10.0**(snr_db_tensor / 10.0)
    capacity_bps = torch.log2(1.0 + snr_linear + 1e-9)
    return F.relu(capacity_bps)

def PowerTargetingConstellationLoss(encoded_points, rate_index, epoch):
    bps = RATES[rate_index]
    target_power = torch.tensor(CONSTELLATION_POWER_TARGETS[bps], device=encoded_points.device)
    
    power_per_point = torch.sum(encoded_points**2, dim=1)
    distance_from_origin = torch.sqrt(power_per_point + 1e-9)
    origin_penalty = -torch.sum(torch.log(distance_from_origin))
    
    if encoded_points.shape[0] > 1:
        pairwise_distances = torch.pdist(encoded_points, p=2) + 1e-9
        electrostatic_loss = torch.sum(1.0 / pairwise_distances)
    else:
        electrostatic_loss = torch.tensor(0.0, device=encoded_points.device)
    
    current_avg_power = torch.mean(power_per_point)
    power_targeting_loss = (current_avg_power - target_power)**2

    if bps == 8:
        w_origin = 0.25
        w_repulsion = 0.04
        w_power = 1.0
    elif bps == 7:
        w_origin = 0.20
        w_repulsion = 0.03
        w_power = 1.0
    elif bps == 6:
        w_origin = 0.15
        w_repulsion = 0.02
        w_power = 1.0
    elif bps >= 4:
        w_origin = 0.12
        w_repulsion = 0.01
        w_power = 1.0
    else:
        w_origin = 0.1
        w_repulsion = 0.005
        w_power = 1.0
        
    total_loss = (w_power * power_targeting_loss) + \
                 (w_origin * origin_penalty) + \
                 (w_repulsion * electrostatic_loss)

    return {
        'total': total_loss,
        'power_loss': power_targeting_loss,
        'origin_loss': origin_penalty,
        'repulsion_loss': electrostatic_loss,
        'avg_power': current_avg_power
    }

def CentroidRegularizationLoss(points, bit_labels, weight=0.05):
    """Provides geometric anchor to prevent constellation phase inversion"""
    if points.shape[0] < 2:
        return torch.tensor(0.0, device=points.device)

    first_bit = bit_labels[:, 0].bool()
    points_for_0 = points[~first_bit]
    points_for_1 = points[first_bit]

    if points_for_0.nelement() == 0 or points_for_1.nelement() == 0:
        return torch.tensor(0.0, device=points.device)

    centroid_0 = torch.mean(points_for_0[:, 0])
    centroid_1 = torch.mean(points_for_1[:, 0])
    loss = F.relu(centroid_0 - centroid_1)
    return loss * weight
def rate_aware_supervised_loss(
    reconstructed_data_logits, original_data, rate_one_hot,
    estimated_csi, true_csi, encoded_symbols_real, epoch,
    reconstructed_latent_vec, original_latent_vec,
    neural_correction_magnitude=None  # 🟢 NEW ARGUMENT
):
    losses = {
        'total': torch.tensor(0.0, device=original_data.device),
        'bce': torch.tensor(0.0, device=original_data.device),
        'scout': torch.tensor(0.0, device=original_data.device),
        'autoencoder': torch.tensor(0.0, device=original_data.device),
        'correction_penalty': torch.tensor(0.0, device=original_data.device),
        'regularization': torch.tensor(0.0, device=original_data.device) # 🟢 NEW KEY
    }
    
    # --- 1. BCE Loss (The Main Objective) ---
    rate_indices = torch.argmax(rate_one_hot, dim=1)
    bps_per_sample = torch.tensor([RATES[i] for i in rate_indices], device=original_data.device)
    max_bps = original_data.shape[1]
    
    # Mask out unused bits
    mask = torch.arange(max_bps, device=original_data.device).unsqueeze(0) < bps_per_sample.unsqueeze(1)
    per_bit_loss = F.binary_cross_entropy_with_logits(reconstructed_data_logits, original_data, reduction="none")
    masked_loss = per_bit_loss * mask
    bce_loss = masked_loss.sum() / (mask.sum() + 1e-9)
    
    losses['bce'] = bce_loss
    losses['total'] += losses['bce']

    # --- 2. Auxiliary Losses (Conditional) ---
    if epoch >= PHASE_2_END:
        weights = get_loss_weights(epoch)
        
        # Scout Loss (Always keep)
        losses['scout'] = F.mse_loss(estimated_csi, true_csi)
        losses['total'] += weights['scout'] * losses['scout']
        
        # Autoencoder Loss (Disable for Low Rates/Low SNR)
        if reconstructed_latent_vec is not None and original_latent_vec is not None:
            ae_loss_raw = F.mse_loss(reconstructed_latent_vec, original_latent_vec, reduction='none')
            ae_loss_per_sample = ae_loss_raw.mean(dim=1)
            
            # Weight Mask: 0.05x for Rate 0/1, 1.0x for others
            ae_rate_weight = torch.ones_like(ae_loss_per_sample)
            ae_rate_weight[rate_indices <= 1] = 0.05 
            
            losses['autoencoder'] = (ae_loss_per_sample * ae_rate_weight).mean()
            losses['total'] += weights['autoencoder'] * losses['autoencoder']

        # Correction Penalty (Output Power Constraint)
        if 'compute_correction_penalty' in globals():
            penalty_raw = compute_correction_penalty(reconstructed_data_logits, original_data, rate_indices, epoch)
            
            avg_rate_index = rate_indices.float().mean().item()
            penalty_weight = 0.03
            if avg_rate_index < 1.5: 
                penalty_weight = 0.0
                
            losses['correction_penalty'] = penalty_raw
            losses['total'] += penalty_weight * losses['correction_penalty']

    # --- 3. 🟢 Laziness Penalty (Regularization) ---
    # Penalize the MAGNITUDE of the neural correction.
    # This prevents the network from distorting clean signals (4dB issue).
    if neural_correction_magnitude is not None:
        reg_loss = torch.mean(neural_correction_magnitude ** 2)
        # 🚨 FIX: Adaptive Regularization
        # Low Rates (0-3): Keep penalty to prevent overfitting noise.
        # High Rates (4-7): REMOVE penalty. They need to work hard!
        
        avg_rate_index = torch.argmax(rate_one_hot, dim=1).float().mean().item()
        
        if avg_rate_index <= 3.0:
            reg_weight = 0.001  # Keep constraints for PSK
        else:
            reg_weight = 0.0    # 🟢 FREE THE BEAST for QAM
        
        losses['regularization'] = reg_loss
        losses['total'] += reg_weight * reg_loss

    return losses
def compute_correction_penalty(reconstructed_data_logits, original_data, rate_indices, epoch):
    penalty = torch.tensor(0.0, device=reconstructed_data_logits.device)
    
    for rate_idx, bps in enumerate(RATES):
        mask = (rate_indices == rate_idx)
        if mask.sum() > 5:
            original_power = torch.mean(original_data[mask] ** 2)
            reconstructed_power = torch.mean(torch.sigmoid(reconstructed_data_logits[mask]) ** 2)
            power_ratio = reconstructed_power / (original_power + 1e-9)
            
            # 🎯 阶段特定的目标范围
            if epoch >= PHASE_2_END:  # Phase 3: 严格限制
                if bps == 1: max_allowed = 1.2
                elif bps <= 3: max_allowed = 1.5
                elif bps <= 6: max_allowed = 2.0  
                else: max_allowed = 3.0
            else:  # Phase 2: 宽松限制
                if bps == 1: max_allowed = 2.0
                elif bps <= 3: max_allowed = 3.0
                elif bps <= 6: max_allowed = 5.0
                else: max_allowed = 8.0
            
            if power_ratio > max_allowed:
                excess = power_ratio - max_allowed
                
                # 🎯 阶段特定的惩罚强度
                if epoch >= PHASE_2_END:  # Phase 3: 较强惩罚
                    rate_penalty = (excess ** 1.5) * 10.0
                else:  # Phase 2: 温和惩罚  
                    rate_penalty = excess * 2.0  # 线性惩罚
                
                # 高码率额外惩罚
                if bps >= 7: rate_penalty *= 3.0
                elif bps >= 5: rate_penalty *= 2.0
                    
                penalty += rate_penalty
   
    return penalty

def get_loss_weights(epoch):
    scout_weight = 5.0
    autoencoder_weight = 2.5

    if epoch >= PHASE_2_END and epoch < PHASE_3_END:
        burn_in_end = PHASE_2_END + 400
        if epoch < burn_in_end:
            start_scout_w = 0.5
            start_ae_w = 0.25
            progress = (epoch - PHASE_2_END) / (burn_in_end - PHASE_2_END)
            scout_weight = start_scout_w + progress * (5.0 - start_scout_w)
            autoencoder_weight = start_ae_w + progress * (2.5 - start_ae_w)
        
    return {'scout': scout_weight, 'autoencoder': autoencoder_weight}

def get_svd_precoder_power_iter(channel_matrices):
    """Calculate Maximum Ratio Transmission precoder using power iteration"""
    v = torch.randn(channel_matrices.shape[0], channel_matrices.shape[2], 1, 
                   dtype=channel_matrices.dtype, device=channel_matrices.device)
    H_H = torch.conj(channel_matrices.transpose(-1, -2))
    for _ in range(3):
        v = H_H @ (channel_matrices @ v)
        v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-9)
    return v.squeeze(-1)

def count_duplicates(points, decimals=4):
    """Count number of duplicate points in a constellation"""
    rounded_points = torch.round(points * (10**decimals))
    unique_points, counts = torch.unique(rounded_points, dim=0, return_counts=True)
    num_duplicates = torch.sum(counts > 1).item()
    total_points_in_duplicates = torch.sum(counts[counts > 1]).item()
    return num_duplicates, total_points_in_duplicates

def weights_init(m):
    """Apply Xavier initialization to Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def print_grad_norms(model, model_name="Model"):
    """Debugging tool to inspect gradient magnitudes"""
    print(f"--- ∇ GRADIENT NORMS for {model_name} ∇ ---")
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            print(f"  -> Layer: {name:<40} | Grad Norm: {param_norm.item():.4f}")
        else:
            if p.requires_grad:
                print(f"  -> Layer: {name:<40} | Grad Norm: !!!!! NO GRADIENT !!!!!")
    total_norm = total_norm ** 0.5
    print(f"  -> TOTAL NORM for {model_name}: {total_norm:.4f}")
    print("-" * (33 + len(model_name)))
             
def systematic_balanced_reward_fixed(snr_db_context, chosen_rate_indices, achieved_throughput, theoretical_max_bps, device):
    """
    Optimized Vectorized Reward Function.
    Focuses on Throughput maximization while respecting Physics (Shannon).
    """
    batch_size = len(chosen_rate_indices)
    
    # 1. Setup Tensors
    # RATES vector corresponding to indices 0..7
    rates_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device, dtype=torch.float32)
    chosen_rates = rates_tensor[chosen_rate_indices]
    
    # 2. Calculate Base Reward (Throughput)
    # This is the most important signal. 
    # High Rate + Low BER = High Reward (e.g. 8.0)
    # High Rate + High BER = Low Reward (e.g. 0.0)
    reward_throughput = achieved_throughput
    
    # 3. Calculate "Efficiency" Bonus (Shannon Alignment)
    # We want the agent to be close to the Shannon Limit, but not exceed it blindly.
    # Gap = Capacity - Chosen Rate
    capacity_gap = theoretical_max_bps - chosen_rates
    
    # Bonus: If we are close to capacity (0 < gap < 3), good job!
    # Penalty: If we far exceeded capacity (gap < -1) and failed, bad job.
    reward_efficiency = torch.zeros(batch_size, device=device)
    
    # Case A: Good Aggression (Selected rate is close to capacity)
    mask_optimal = (capacity_gap >= 0) & (capacity_gap < 2.5)
    reward_efficiency[mask_optimal] += 1.0
    
    # Case B: Over-Aggressive (Selected rate >> capacity) AND Failed (Throughput low)
    # Only punish if it actually failed. If it succeeded, let it be (Receiver might be superhuman).
    mask_fail = (capacity_gap < -1.0) & (achieved_throughput < 1.0)
    reward_efficiency[mask_fail] -= 2.0
    
    # Case C: Under-Aggressive (Selected rate << capacity)
    # Wasting potential spectrum.
    mask_lazy = (capacity_gap > 4.0)
    reward_efficiency[mask_lazy] -= 1.0

    # 4. Total Reward Construction
    # We weight Throughput heavily (1.0) because that is the ultimate goal.
    total_reward = (1.0 * reward_throughput) + (0.5 * reward_efficiency)
    
    # 5. Sanity Clamp (Prevent exploding gradients in RL)
    # Range roughly -2.0 to +9.0
    return torch.clamp(total_reward, min=-5.0, max=10.0)   
    
def optimize_gpu_memory():
    """优化GPU内存使用"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 设置更积极的内存配置
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # 为了速度牺牲确定性

def apply_channel_curriculum(channel, epoch):
    """
    Updates the EXISTING channel object.
    modified: Enforces Minimum SNR = 0 dB to match system hardware limits.
    """
    
    # ---------------------------------------------------------
    # PHASE 1: INITIALIZATION
    # ---------------------------------------------------------
    if epoch < PHASE_1_END:
        channel.reset_environment(
            distance_meters=100, 
            fixed_snr_db=20.0, 
            rician_k_factor=10.0
        )
    
    # ---------------------------------------------------------
    # PHASE 2: RECEIVER ROBUSTNESS
    # ---------------------------------------------------------
    elif epoch < PHASE_2_LESSON_1_END:
        # Lesson 2-1: 15 dB
        channel.reset_environment(
            distance_meters=200, 
            fixed_snr_db=15.0, 
            rician_k_factor=5.0
        )
        
    elif epoch < PHASE_2_LESSON_2_END:
        # Lesson 2-2: 10 dB
        channel.reset_environment(
            distance_meters=300, 
            fixed_snr_db=10.0,  # Adjusted from 12
            rician_k_factor=0.0
        )
        
    elif epoch < PHASE_2_END:
        # Lesson 2-3: 5 - 15 dB (Safe Zone)
        random_snr = np.random.uniform(5.0, 15.0)
        channel.reset_environment(
            distance_meters=400, 
            fixed_snr_db=random_snr, 
            rician_k_factor=0.0
        )

    # ---------------------------------------------------------
    # PHASE 3: END-TO-END
    # ---------------------------------------------------------
    elif epoch < PHASE_3_LESSON_1_END:
        # Lesson 3-1: 5 - 12 dB
        random_dist = np.random.uniform(200, 600)
        random_snr = np.random.uniform(5.0, 12.0)
        channel.reset_environment(
            distance_meters=random_dist, 
            fixed_snr_db=random_snr,
            rician_k_factor=0.0
        )
        
    elif epoch < PHASE_3_LESSON_2_END:
        # Lesson 3-2: 0 - 8 dB (Hard but valid)
        random_dist = np.random.uniform(600, 1200)
        random_snr = np.random.uniform(0.0, 8.0)
        channel.reset_environment(
            distance_meters=random_dist, 
            fixed_snr_db=random_snr,
            rician_k_factor=0.0
        )
        
    elif epoch < PHASE_3_END:
        # Lesson 3-3: 0 - 5 dB (The Limit)
        # Previous: -5 to 5. Adjusted to 0 to 5.
        random_dist = np.random.uniform(1000, 2000)
        random_snr = np.random.uniform(0.0, 5.0) 
        channel.reset_environment(
            distance_meters=random_dist, 
            fixed_snr_db=random_snr,
            rician_k_factor=0.0
        )
        
    # ---------------------------------------------------------
    # PHASE 4: MOE / GENERALIZATION
    # ---------------------------------------------------------
    else:
        # Full Physics Mode
        random_distance = np.random.uniform(low=50, high=2500)
        # Clamp Min SNR to 0.0
        random_snr = np.random.uniform(0.0, 30.0)
        
        channel.reset_environment(
            distance_meters=random_distance, 
            fixed_snr_db=random_snr,
            rician_k_factor=0.0
        )
        
    return channel

def apply_moe_synergy_curriculum(channel, epoch):
    """
    Refactored MOE Curriculum with:
    1. Auto-Switching for Rate 8 (Geometry -> Robustness).
    2. Critical Bug Fix for Noise Accumulation (prevents loss explosion).
    3. User-defined lesson plans.
    """
    phase_4a_start = PHASE_3_END
    epochs_into_phase4a = epoch - phase_4a_start
    
    # --- 1. Identify Current Lesson & Local Progress ---
    current_lesson_index = 0
    accumulated_epochs = 0
    epochs_within_lesson = 0
    
    for rate_idx, duration in PHASE_4A_DURATION_PER_RATE.items():
        if epochs_into_phase4a < accumulated_epochs + duration:
            current_lesson_index = rate_idx
            epochs_within_lesson = epochs_into_phase4a - accumulated_epochs
            break
        accumulated_epochs += duration
    
    force_rate = current_lesson_index

    # --- 2. Base Lesson Plans (Based on your request) ---
    lesson_plans = {
        # Rate 0 (1bps): Wide Dynamic Range
        0: (100, 1500, -0.5, 1.0), 

        # Rate 1 (2bps/QPSK): Convergence Rescue
        1: (200, 1000, -0.5, 0.5),

        # Rate 2 (3bps/8-Point): Transition
        2: (200, 800, -0.5, 0.5),

        # Rate 3 (4bps/16-QAM): Robustness
        3: (50, 600, -0.5, 0.5),

        # Rate 4 (5bps / 32-QAM): Harder Phase
        # Min Noise 0.0 (Thermal). Max Noise 0.6 (~4x Thermal).
        4: (50, 500, 0.0, 0.6), 

        # Rate 5 (6bps / 64-QAM): Hardened State
        # Goal: Force errors so 'w_correction' grows.
        5: (50, 450, -0.2, 0.3),

        # Rate 6 (7bps / 128-QAM): Hardened State
        # Min Noise: -0.2 (Still clean-ish)
        # Max Noise: 0.5 (Real noise, ~24dB)
        6: (20, 250, -0.2, 0.5),        
        
        # Rate 7 (8bps / 256-QAM): Default Start (Geometry First)
        # This will be overridden by the Auto-Switch logic below
        7: (10, 100, -0.5, -0.2)                     
    }

    # --- 3. 🌙 RATE 8 (INDEX 7) AUTO-SWITCH LOGIC ---
    if current_lesson_index == 7:
        if epochs_within_lesson < 50:
            # PHASE 1: GEOMETRY FIRST (Easy)
            # High SNR to learn the 256-point spiral.
            #print(f"   🌙 Rate 8 Setup: GEOMETRY PHASE ({epochs_within_lesson}/50)")
            lesson_plans[7] = (10, 100, -0.5, -0.2) 
        else:
            # PHASE 2: HARDENING (Harder)
            # Triggers automatically after 50 epochs.
            #print(f"   ☀️ Rate 8 Setup: ROBUSTNESS PHASE")
            # Increase distance slightly, allow positive noise
            #lesson_plans[7] = (10, 150, -0.5, 0.2)
            # New: (10, 200, -0.2, 0.3)
            #   - Min Noise: -0.2 (Still clean enough to see the grid).
            #   - Max Noise: 0.3 (Real noise).
            #   - Goal: Force errors so the Equalizer weights move from 0.0.
            lesson_plans[7] = (10, 200, -0.2, 0.3) 
    # --- 4. Select Parameters ---
    if current_lesson_index in lesson_plans:
        plan = lesson_plans[current_lesson_index]
        random_distance = np.random.uniform(plan[0], plan[1])
        noise_mult_log = np.random.uniform(plan[2], plan[3])
        noise_multiplier = 10**noise_mult_log
    else:
        random_distance = np.random.uniform(50, 500)
        noise_multiplier = 1.0

    # --- 5. Apply Updates (WITH CRITICAL FIX) ---
    channel.reset_environment(distance_meters=random_distance, fixed_snr_db=None)
    
    # 🚨 BUG FIX START: Hard Reset of Noise Power 🚨
    # We must reset to baseline Thermal Noise before applying the multiplier.
    # Otherwise, noise multiplies cumulatively (0.5 * 0.5 * ...) causing explosion.
    kT_dbm_hz = -174
    bandwidth_hz = RealisticConfig.BANDWIDTH_HZ
    noise_figure_db = RealisticConfig.NOISE_FIGURE_DB
    noise_power_dbm = kT_dbm_hz + 10 * np.log10(bandwidth_hz) + noise_figure_db
    base_noise_watts = 10**((noise_power_dbm - 30) / 10.0)
    
    # Reset channel noise to baseline
    channel.noise_power_watts = base_noise_watts
    # 🚨 BUG FIX END
    
    # Apply specific multiplier for this batch
    if noise_multiplier != 1.0:
        channel.noise_power_watts *= noise_multiplier
        
    # Recalculate dependent stats
    channel.noise_std_dev = np.sqrt(channel.noise_power_watts / 2.0)
    
    # Guard clause: If logic pushes SNR < 0 (impossible physics), cap noise
    current_rx_power = channel.expected_rx_power_watts 
    new_snr_linear = current_rx_power / (channel.noise_power_watts + 1e-12)
    
    if new_snr_linear < 1.0: # < 0 dB
        channel.noise_power_watts = current_rx_power
        channel.noise_std_dev = np.sqrt(channel.noise_power_watts / 2.0)
        new_snr_linear = 1.0
        
    channel.actual_snr_db = 10 * np.log10(new_snr_linear + 1e-12)

    return force_rate

def check_constellation_integrity(epoch, transmitter, receiver, device, current_phase="Phase 1"):
    """
    <<< PHASE-AWARE CONSTELLATION DIAGNOSTIC V5.0 >>>
    Updated for new RealisticMIMOChannel model
    """
    print("\n" + "—" * 80)
    print(f" 🛠️  Epoch {epoch+1} {current_phase} - Constellation Diagnostic  🛠️")
    print("—" * 80)

    transmitter.eval()
    if "Phase 1" in current_phase:
        print("   📝 NOTE: Phase 1 - Testing constellation geometry only (receiver untrained)")
    else:
        receiver.eval()

    test_plan = {
        1: {'snr': 10.0}, 2: {'snr': 12.0}, 3: {'snr': 14.0}, 
        4: {'snr': 16.0}, 5: {'snr': 18.0}, 6: {'snr': 20.0}, 
        7: {'snr': 22.0}, 8: {'snr': 24.0}
    }

    BATCH_SIZE_TEST = 4096

    with torch.no_grad():
        constellation_quality_scores = []
        
        for bps, test_config in test_plan.items():
            snr_db = test_config['snr']
            
            print(f"\n--- Rate {bps} bps ---")
            
            try:
                rate_index = RATES.index(bps)
            except ValueError:
                print(f"   -> Skipping {bps}bps (not in RATES)")
                continue

            # --- Part 1: Constellation Geometry Analysis (ALWAYS TESTED) ---
            num_points = 2**bps
            bits = list(itertools.product([0, 1], repeat=bps))
            data_chunk = torch.zeros(num_points, MAX_RATE_BPS, device=device)
            data_chunk[:, :bps] = torch.tensor(bits, device=device)
            
            rate_one_hot_geom = F.one_hot(torch.full((num_points,), rate_index, device=device), num_classes=len(RATES)).float()
            points = transmitter.constellation_encoder(data_chunk, rate_one_hot_geom)

            # Geometry metrics
            if points.shape[0] > 1:
                min_dist = torch.min(torch.pdist(points, p=2)).item()
                avg_dist = torch.mean(torch.pdist(points, p=2)).item()
            else:
                min_dist = 0.0
                avg_dist = 0.0
            
            num_dup_pairs, num_dup_points = count_duplicates(points)
            avg_power = torch.mean(torch.sum(points**2, dim=1)).item()
            target_power = CONSTELLATION_POWER_TARGETS[bps]
            power_error = abs(avg_power - target_power) / target_power * 100

            # Quality scoring
            quality_score = 0
            if min_dist > 0.1: quality_score += 2
            elif min_dist > 0.05: quality_score += 1
            
            if power_error < 5: quality_score += 2
            elif power_error < 10: quality_score += 1
            
            if num_dup_points == 0: quality_score += 1
            
            constellation_quality_scores.append(quality_score)

            print(f"   [1] Geometry -> Min Dist: {min_dist:.4f} | Avg Dist: {avg_dist:.4f}")
            print(f"        Power: {avg_power:.3f} (Target: {target_power:.1f}, Error: {power_error:.1f}%)")
            print(f"        Collapsed Points: {num_dup_points}/{num_points}")
            print(f"        Quality Score: {quality_score}/5")

            # Quality assessment
            if quality_score >= 4:
                assessment = "✅ EXCELLENT"
            elif quality_score >= 3:
                assessment = "⚠️  GOOD" 
            elif quality_score >= 2:
                assessment = "⚠️  FAIR"
            else:
                assessment = "❌ POOR"
            print(f"        Assessment: {assessment}")

            # --- Part 2: E2E Accuracy (ONLY IN LATER PHASES) ---
            if "Phase 1" not in current_phase:
                from channel import RealisticMIMOChannel
                
                test_data = torch.randint(0, 2, (BATCH_SIZE_TEST, MAX_RATE_BPS), device=device).float()
                rate_one_hot = F.one_hot(torch.full((BATCH_SIZE_TEST,), rate_index, device=device), len(RATES)).float()
                
                # 🚨 使用新的信道模型
                test_channel = RealisticMIMOChannel(
                    batch_size=BATCH_SIZE_TEST,
                    device=device,
                    use_fading=False,  # Phase 1诊断使用无衰落
                    distance_meters=100,  # 固定距离
                    fixed_snr_db=snr_db   # 使用测试SNR
                )
                
                # 获取CSI（虽然是零，但保持接口一致）
                dummy_csi = test_channel.get_csi()
                noise_power_tensor = torch.full((BATCH_SIZE_TEST,), test_channel.noise_power_watts, device=device)
                
                # 发射器
                tx_signal_complex, encoded_symbols_real, _, _ = transmitter(
                    test_data, dummy_csi, noise_power_tensor, force_rate_index=rate_index
                )
                
                # 🚨 信号归一化
                tx_signal_normalized = tx_signal_complex / torch.norm(tx_signal_complex, dim=1, keepdim=True)
                
                # 应用信道
                rx_noisy = test_channel.apply(tx_signal_normalized, add_noise=True)
                # 🚨 FIX START: Add Automatic Gain Control (AGC) 🚨
                # This matches the logic we added to main.py
                rx_power = torch.mean(torch.abs(rx_noisy)**2, dim=1, keepdim=True)
                agc_scale = torch.sqrt(1.0 / (rx_power + 1e-9))
                rx_noisy = rx_noisy * agc_scale
                # 🚨 FIX END                 
                # 🚨 使用与主训练相同的均衡器逻辑
                # 对于诊断，使用简单的MRC均衡
                if rx_noisy.dim() == 2 and rx_noisy.shape[1] == 4:  # [B, 4] 复数
                    # 4天线信号 - 使用最大比合并
                    h_eff = torch.ones(BATCH_SIZE_TEST, 4, device=device, dtype=torch.complex64)  # 假设理想信道
                    h_eff_conj = torch.conj(h_eff)
                    Es_average = torch.mean(torch.abs(tx_signal_normalized)**2)
                    noise_power_for_eq = noise_power_tensor.unsqueeze(1)
                    snr_inverse_term = noise_power_for_eq / (Es_average + 1e-9)
                    denominator = torch.abs(h_eff)**2 + snr_inverse_term
                    mmse_factor = h_eff_conj / (denominator + 1e-9)
                    
                    equalized_complex = rx_noisy * mmse_factor  # [B, 4] 复数
                    equalized_real = torch.view_as_real(equalized_complex).view(BATCH_SIZE_TEST, -1)  # [B, 8]
                else:
                    # 单天线或未知格式 - 使用简单均衡
                    equalized_real = torch.view_as_real(rx_noisy).view(BATCH_SIZE_TEST, -1)  # 展开为实数
                
                # 接收器
                reconstructed_logits,_ = receiver(
                    equalized_real, dummy_csi, noise_power_tensor, rate_one_hot=rate_one_hot
                )
                
                predicted_bits = (torch.sigmoid(reconstructed_logits) > 0.5).float()
                mask = torch.arange(MAX_RATE_BPS, device=device).unsqueeze(0) < bps
                correct_bits = torch.sum((predicted_bits == test_data) * mask).item()
                total_bits = torch.sum(mask).item() * BATCH_SIZE_TEST
                accuracy = (correct_bits / total_bits) * 100 if total_bits > 0 else 0.0

                print(f"   [2] E2E Accuracy (AWGN @ {snr_db:.1f}dB): {accuracy:.2f}%")
                print(f"        Rx Power: {torch.mean(torch.abs(rx_noisy)**2):.2e}")
                print(f"        Noise Power: {test_channel.noise_power_watts:.2e}")
                
                # Accuracy assessment
                if accuracy > 95.0:
                    acc_assessment = "✅ EXCELLENT"
                elif accuracy > 90.0:
                    acc_assessment = "⚠️  GOOD"
                elif accuracy > 80.0:
                    acc_assessment = "⚠️  FAIR" 
                else:
                    acc_assessment = "❌ NEEDS TRAINING"
                print(f"        Receiver Status: {acc_assessment}")

        # Overall constellation quality summary
        if constellation_quality_scores:
            avg_quality = sum(constellation_quality_scores) / len(constellation_quality_scores)
            print(f"\n🎯 CONSTELLATION SUMMARY -> Average Quality: {avg_quality:.1f}/5.0")
            
            if avg_quality >= 4.0:
                print("   🏆 STATUS: CONSTELLATIONS READY FOR PHASE 2")
            elif avg_quality >= 3.0:
                print("   ✅ STATUS: CONSTELLATIONS ADEQUATE")
            else:
                print("   ⚠️  STATUS: CONSTELLATIONS NEED IMPROVEMENT")

    print("—" * 80 + "\n")
    transmitter.train()
    if "Phase 1" not in current_phase:
        receiver.train()
def check_moe_bps_performance(epoch, scout, transmitter, receiver, device, phase_name=""):
    """
    Updated Diagnostic: Injects Noise to hit target SNR (Physics-Aligned).
    Does NOT reduce Tx Power.
    """
    print("\n" + "="*100)
    print(f" 📊 Epoch {epoch+1} MOE BPS Diagnostic ({phase_name}) 📊")
    print("="*100)
    from channel import RealisticMIMOChannel

    scout.eval(); transmitter.eval(); receiver.eval()
    
    BATCH_SIZE_TEST = 1024 # Increased for stability
    
    # 定义bps特定的测试距离
    test_distances = {
        1: 1000,   # 1bps: 远距离，测试极限性能
        2: 600,    # 2bps: 中远距离  
        3: 400,    # 3bps: 中等距离
        4: 250,    # 4bps: 中近距离
        5: 150,    # 5bps: 近距离
        6: 80,     # 6bps: 较近距离
        7: 40,     # 7bps: 近距离
        8: 20      # 8bps: 极近距离
    }
    # MOE BPS测试配置
    test_plan = {
        1: {'snrs': [0.0, 2.0,4.0], 'complexity': 'low'},      # 更具挑战性
        2: {'snrs': [3.0, 5.0,7.0], 'complexity': 'low'}, 
        3: {'snrs': [6.0, 8.0], 'complexity': 'medium'},
        4: {'snrs': [9.0, 11.0], 'complexity': 'medium'},
        5: {'snrs': [14.0, 16.0,18.0], 'complexity': 'high'},
        6: {'snrs': [17.0, 20.0, 23.0], 'complexity': 'high'},
        7: {'snrs': [22.0, 25.0,28.0], 'complexity': 'very_high'},
        8: {'snrs': [26.0, 29.0,32.0], 'complexity': 'extreme'}
    }

    overall_results = {}
    
    # Create one channel instance to reuse
    test_channel = RealisticMIMOChannel(BATCH_SIZE_TEST, device, distance_meters=100)

    with torch.no_grad():
        for bps, config in test_plan.items():
            rate_index = RATES.index(bps)
            test_distance = test_distances[bps] if 'dist' not in config else config['dist']
            
            print(f"\n🎯 Test {bps}bps - {config.get('complexity', 'STD')} (Dist: {test_distance}m)")
            print("-" * 60)
            
            bps_results = {}
            
            for target_snr_db in config['snrs']:
                # 1. Reset Physics (Full Tx Power)
                # We do NOT use fixed_snr_db here because that scales Tx power.
                test_channel.reset_environment(distance_meters=test_distance, fixed_snr_db=None)
                
                # 2. Calculate Required Noise to hit Target SNR
                # Current Signal Power (Approx)
                rx_power_watts = test_channel.expected_rx_power_watts
                
                # Target Noise = Rx / (10^(SNR/10))
                target_snr_linear = 10**(target_snr_db / 10.0)
                required_noise_watts = rx_power_watts / (target_snr_linear + 1e-9)
                
                # 3. Inject Noise
                test_channel.noise_power_watts = required_noise_watts
                test_channel.noise_std_dev = np.sqrt(required_noise_watts / 2.0)
                
                # Update reported SNR for consistency
                test_channel.actual_snr_db = target_snr_db
                
                # 4. Generate Data
                test_data = torch.randint(0, 2, (BATCH_SIZE_TEST, bps), device=device).float()
                true_csi = test_channel.get_csi()
                
                # 5. Scout & Transmit (Use Used Precoder!)
                estimated_csi = scout(true_csi)
                noise_tensor = torch.full((BATCH_SIZE_TEST,), test_channel.noise_power_watts, device=device)
                
                tx_signal, _, rate_one_hot, _, used_precoder = transmitter(
                    test_data, estimated_csi, noise_tensor, force_rate_index=rate_index
                )
                
                # 6. Apply Channel (With our custom noise)
                # Power Norm
                sample_power = torch.mean(torch.abs(tx_signal)**2, dim=1, keepdim=True)
                scale = torch.sqrt(1.0 / (sample_power + 1e-9))
                tx_signal_norm = tx_signal * scale
                
                rx_noisy = test_channel.apply(tx_signal_norm, add_noise=True)
                # 🚨 FIX START: Add Automatic Gain Control (AGC) 🚨
                # This matches the logic we added to main.py
                rx_power = torch.mean(torch.abs(rx_noisy)**2, dim=1, keepdim=True)
                agc_scale = torch.sqrt(1.0 / (rx_power + 1e-9))
                rx_noisy = rx_noisy * agc_scale
                # 🚨 FIX END                
                # 7. Equalization
                precoder_c64 = used_precoder.to(torch.complex64)
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    rx_c64 = rx_noisy.to(torch.complex64)
                    h_c64 = test_channel.H_fading.to(torch.complex64)
                    
                    h_eff = torch.bmm(h_c64, precoder_c64.unsqueeze(-1)).squeeze(-1)
                    snr_tensor = torch.full((BATCH_SIZE_TEST, 1), target_snr_db, device=device)
                    
                    eq_complex = adaptive_mmse_equalization(rx_c64, h_eff, snr_tensor)
                    eq_real = torch.view_as_real(eq_complex).view(BATCH_SIZE_TEST, -1)
                    eq_real = torch.clamp(eq_real, -30.0, 30.0) # Loose clamp
                
                # 8. Receiver
                logits,_ = receiver(eq_real, estimated_csi, noise_tensor, rate_one_hot=rate_one_hot)
                
                # 9. Metrics
                predicted_bits = (torch.sigmoid(logits) > 0.5).float()
                correct = torch.sum(predicted_bits[:, :bps] == test_data).item()
                total = BATCH_SIZE_TEST * bps
                acc = (correct / total) * 100
                ber = 1.0 - (correct / total)
                
                throughput = bps * (1.0 - ber)
                print(f"   SNR {target_snr_db:4.1f}dB | Acc: {acc:6.2f}% | BER: {ber:.5f} | Throughput: {throughput:.2f}")
                
                bps_results[target_snr_db] = {'accuracy': acc, 'throughput': throughput}
            
            overall_results[bps] = bps_results

    scout.train(); transmitter.train(); receiver.train()
    return overall_results, 0

def adaptive_mmse_equalization(rx_signal, channel_matrix, snr_db):
    """
    Standard MMSE Equalization without artificial power scaling.
    Physics-aware: Rely on the correct SNR and Channel Matrix.
    """
    # 1. Prepare dimensions for broadcasting
    # Ensure snr_db is [Batch, 1] if input is [Batch] or [Batch, 1]
    if isinstance(snr_db, torch.Tensor):
        if snr_db.dim() == 1:
            snr_db = snr_db.unsqueeze(1)
            
    # 2. Calculate Linear SNR
    snr_linear = 10**(snr_db / 10.0)
    
    # 3. Standard MMSE Calculation
    # W = h* / (|h|^2 + 1/SNR)
    h_eff = channel_matrix
    h_eff_conj = torch.conj(h_eff)
    h_eff_power = torch.abs(h_eff)**2
    
    # Add broadcasting support for snr_linear
    denominator = h_eff_power + (1.0 / (snr_linear + 1e-9))
    mmse_weights = h_eff_conj / (denominator + 1e-9)
    
    # 4. Apply Equalization
    equalized = rx_signal * mmse_weights
    
    # 5. Optional: Simple Safety Clamp
    # We do NOT scale the power. We only clamp extreme outliers to prevent NaN/Inf in training.
    # Real MMSE outputs should be roughly in range [-1.5, 1.5] for normalized inputs.
    equalized_real = torch.view_as_real(equalized)
    equalized_real = torch.clamp(equalized_real, -20.0, 20.0) # Very wide safety net
    equalized = torch.view_as_complex(equalized_real)
    
    return equalized
def mutual_information_constraint(equalized_output, true_bits):
    # 计算输出分布的熵
    output_probs = torch.sigmoid(equalized_output)
    output_entropy = -torch.mean(output_probs * torch.log(output_probs + 1e-9))
    
    # 计算条件熵
    conditioned_probs = output_probs * true_bits + (1 - output_probs) * (1 - true_bits)
    conditional_entropy = -torch.mean(torch.log(conditioned_probs + 1e-9))
    
    # 互信息
    mutual_info = output_entropy - conditional_entropy
    
    # 约束：互信息应该接近理论信道容量
    target_capacity = 0.5  # 对于BPSK在低SNR下
    capacity_loss = torch.abs(mutual_info - target_capacity)
    
    return capacity_loss    

def perfect_equalization(rx_noisy, H_fading, noise_power):
    """完美均衡器（使用真实信道知识）"""
    with torch.amp.autocast(device_type='cuda', enabled=False):
        rx_noisy_c64 = rx_noisy.to(torch.complex64)
        h_fading_c64 = H_fading.to(torch.complex64)
        
        # 假设使用MRT预编码（所有天线权重相等）
        precoder = torch.ones(H_fading.shape[0], H_fading.shape[2], device=device, dtype=torch.complex64)
        
        h_eff_complex = torch.bmm(h_fading_c64, precoder.unsqueeze(-1)).squeeze(-1)
        h_eff_conj = torch.conj(h_eff_complex)
        Es_average = 1.0  # 因为信号已经归一化
        
        noise_power_for_eq = noise_power.unsqueeze(1)
        snr_inverse_term = noise_power_for_eq / (Es_average + 1e-9)
        denominator = torch.abs(h_eff_complex)**2 + snr_inverse_term
        mmse_factor = h_eff_conj / (denominator + 1e-9)
        
        equalized_complex = rx_noisy_c64 * mmse_factor
        equalized_real = torch.view_as_real(equalized_complex).view(rx_noisy.shape[0], -1)
        
        return equalized_real

def compute_accuracy(reconstructed_logits, true_data, bps):
    """计算准确率"""
    predicted_bits = (torch.sigmoid(reconstructed_logits) > 0.5).float()
    mask = torch.arange(MAX_RATE_BPS, device=device).unsqueeze(0) < bps
    correct_bits = torch.sum((predicted_bits == true_data) * mask).item()
    total_bits = torch.sum(mask).item() * true_data.shape[0]
    accuracy = (correct_bits / total_bits) * 100 if total_bits > 0 else 0.0
    return accuracy
 
def debug_equalizer_weights(epoch, receiver):
    """
    Diagnose the Neural Equalizer weights for the CURRENTLY training MOE expert.
    Automatically detects which rate is active based on the epoch.
    """
    # 1. Determine Current Rate Index based on Schedule
    # (Same logic as apply_moe_synergy_curriculum)
    phase_4a_start = PHASE_3_END
    epochs_into_phase4a = epoch - phase_4a_start
    
    current_rate_index = 0
    accumulated_epochs = 0
    
    # Find which rate specialist is currently active
    for rate_idx in sorted(PHASE_4A_DURATION_PER_RATE.keys()):
        duration = PHASE_4A_DURATION_PER_RATE[rate_idx]
        if epochs_into_phase4a < accumulated_epochs + duration:
            current_rate_index = rate_idx
            break
        accumulated_epochs += duration
    else:
        # If we are past the last scheduled phase, check the last one
        current_rate_index = max(PHASE_4A_DURATION_PER_RATE.keys())

    bps = RATES[current_rate_index]

    # 2. Select the Correct Layer (Logic matches Receiver.apply_neural_equalization)
    if current_rate_index < 3:
        # Low Rates (1, 2, 3 bps) use 'rate_specific_equalizers'
        target_layer = receiver.rate_specific_equalizers[current_rate_index]
        layer_type = "Rate Specific Equalizer"
    else:
        # High Rates (4+ bps) use 'equalization_correctors'
        corrector_index = current_rate_index - 3
        # Safety check for array bounds
        if corrector_index >= len(receiver.equalization_correctors):
            print(f"\n⚠️ Debug Skipped: Corrector Index {corrector_index} not found.")
            return
        target_layer = receiver.equalization_correctors[corrector_index]
        layer_type = f"Equalization Corrector (Idx {corrector_index})"

    # 3. Extract Weights
    # Check if layer has the expected attributes (EnhancedNeuralEqualizationCorrection)
    if not hasattr(target_layer, 'attention_weights'):
        print(f"\n⚠️ Debug Info: Rate {bps}bps layer does not use Attention Weights (Simple Layer).")
        return

    weights = target_layer.attention_weights.detach().cpu().numpy()
    feat_weights = target_layer.feature_weights.detach().cpu().numpy()
    
    w_original = weights[0]
    w_correction = weights[1]
    
    # 4. Print Diagnosis
    print(f"\n🔍 [Epoch {epoch}] Equalizer Diagnostics (Current Expert: Rate {bps}bps / {layer_type}):")
    print(f"   ⚖️  Original Signal Weight: {w_original:.4f}")
    print(f"   🧠  Neural Correction Weight: {w_correction:.4f}")
    
    # Interpretation
    if abs(w_correction) < 0.01:
        print("   ⚠️  WARNING: Neural Network is ASLEEP (Weight ~0).")
        print("       (This is normal for 'Clean' data, but bad if Accuracy is stuck low)")
    elif abs(w_correction) > abs(w_original):
        print("   🔥  STATUS: Neural Network is DOMINANT (Doing heavy lifting).")
    else:
        print("   ✅  STATUS: Hybrid Mode (Fine-tuning math output).")
        
    print(f"   📶  SNR Input Weight: {feat_weights[9]:.4f}")
    print(f"   📡  Channel Power Weight: {feat_weights[8]:.4f}")
def debug_tensor_stats(name, tensor):
    """Helper to print mean/std/max of a tensor."""
    if tensor is None:
        return
    
    # Detach and move to cpu for safety
    t = tensor.detach().float().cpu()
    
    mean = t.mean().item()
    std = t.std().item()
    abs_max = t.abs().max().item()
    
    print(f"   🔎 {name:<20} | Mean: {mean:+.4f} | Std: {std:.4f} | Max: {abs_max:.4f}")

def deep_debug_signal_path(epoch, transmitter, receiver, device, target_rate_idx=5):
    """
    Debugs the full signal path. 
    Creates a LOCAL channel to ensure batch sizes match (128).
    """
    print(f"\n🔬 --- DEEP DEBUG SIGNAL PATH (Epoch {epoch}) ---")
    from channel import RealisticMIMOChannel
    from utils import adaptive_mmse_equalization # Ensure visibility
    
    bps = RATES[target_rate_idx]
    print(f"   Target: Rate {bps} bps (Index {target_rate_idx})")
    
    # 1. Setup Local Environment
    batch_size = 128
    # Create a fresh channel just for this debug batch
    debug_channel = RealisticMIMOChannel(batch_size, device, distance_meters=100, fixed_snr_db=30.0)
    
    # 2. Generate Data
    data = torch.randint(0, 2, (batch_size, bps), device=device).float()
    
    # 3. Transmitter
    true_csi = debug_channel.get_csi()
    noise = torch.zeros(batch_size, device=device)
    
    with torch.no_grad():
        tx_complex, encoded_real, _, _, precoder = transmitter(
            data, true_csi, noise, force_rate_index=target_rate_idx
        )
        
    debug_tensor_stats("1. Tx Constellation", encoded_real)
    debug_tensor_stats("2. Tx Complex Sig", tx_complex)
    
    # 4. Channel Apply
    # Power Norm
    pwr = torch.mean(torch.abs(tx_complex)**2, dim=1, keepdim=True)
    scale = torch.sqrt(1.0 / (pwr + 1e-9))
    tx_norm = tx_complex * scale
    
    rx_noisy = debug_channel.apply(tx_norm, add_noise=True)
    # 🚨 FIX: Add AGC here too
    print(f"   📉 Raw Rx Power: {torch.mean(torch.abs(rx_noisy)**2).item():.2e}")
    rx_pwr = torch.mean(torch.abs(rx_noisy)**2, dim=1, keepdim=True)
    rx_scale = torch.sqrt(1.0 / (rx_pwr + 1e-9))
    rx_noisy = rx_noisy * rx_scale
    print(f"   📈 AGC Corrected Rx Power: {torch.mean(torch.abs(rx_noisy)**2).item():.2f}")    
    # 5. Manual MMSE
    with torch.amp.autocast(device_type='cuda', enabled=False):
        # Use the debug_channel's fading matrix (which is size 128)
        h = debug_channel.H_fading.to(torch.complex64)
        p = precoder.to(torch.complex64)
        h_eff = torch.bmm(h, p.unsqueeze(-1)).squeeze(-1)
        snr_tensor = torch.full((batch_size, 1), 30.0, device=device)
        
        eq_complex = adaptive_mmse_equalization(rx_noisy.to(torch.complex64), h_eff, snr_tensor)
        eq_real = torch.view_as_real(eq_complex).view(batch_size, -1)
        
    debug_tensor_stats("3. MMSE Output", eq_real)
    
    # 6. Receiver Input Scaler
    stabilizer = receiver.input_stabilizers[target_rate_idx]
    
    with torch.no_grad():
        stabilized = stabilizer(eq_real)
        
    debug_tensor_stats("4. Scaler Output", stabilized)
    
    scale_param = stabilizer.scale.detach().cpu().mean().item()
    print(f"   ⚖️  SimpleScaler Value: {scale_param:.4f}")
    
    print("------------------------------------------------")  
# -----------------------------------------------------------------------------
# Shared Helper: Physics-Compliant Test Step
# -----------------------------------------------------------------------------
def _run_physics_compliant_test(transmitter, receiver, device, rate_bps, snr_db=None, distance=None):
    """
    Unified testing logic that matches the corrected Phase 3 Training Loop.
    Ensures Power Normalization -> Physics Channel -> MMSE -> Decoder pipeline is valid.
    """
    BATCH_SIZE_TEST = 512
    from channel import RealisticMIMOChannel
    # 1. Setup Channel
    # If snr_db is provided, we force it (Fundamentals Test)
    # If distance is provided, we use physics (Robustness Test)
    channel = RealisticMIMOChannel(
        batch_size=BATCH_SIZE_TEST,
        device=device,
        use_fading=True,
        distance_meters=distance if distance else 500,
        fixed_snr_db=snr_db,
        verbose=False
    )
    
    # 2. Generate Data & Tx
    test_data = torch.randint(0, 2, (BATCH_SIZE_TEST, rate_bps), device=device).float()
    csi = channel.get_csi()
    noise_power_tensor = torch.full((BATCH_SIZE_TEST,), channel.noise_power_watts, device=device)
    
    rate_index = RATES.index(rate_bps)
    
    tx_signal, _, rate_one_hot, _,used_precoder  = transmitter(
        test_data, csi, noise_power_tensor, force_rate_index=rate_index
    )
    
    # 3. 🚨 Power Normalization (Crucial Fix)
    sample_power = torch.mean(torch.abs(tx_signal)**2, dim=1, keepdim=True)
    scale = torch.sqrt(1.0 / (sample_power + 1e-9))
    tx_signal_normalized = tx_signal * scale
    
    # 4. Apply Channel
    rx_noisy = channel.apply(tx_signal_normalized, add_noise=True)
    # 🚨 FIX START: Add Automatic Gain Control (AGC) 🚨
    # This matches the logic we added to main.py
    rx_power = torch.mean(torch.abs(rx_noisy)**2, dim=1, keepdim=True)
    agc_scale = torch.sqrt(1.0 / (rx_power + 1e-9))
    rx_noisy = rx_noisy * agc_scale
    # 🚨 FIX END     
    # 5. 🚨 Standard MMSE Equalization (Matches Training Loop)
    precoder = used_precoder
    
    with torch.amp.autocast(device_type='cuda', enabled=False):
        rx_noisy_c64 = rx_noisy.to(torch.complex64)
        h_fading_c64 = channel.H_fading.to(torch.complex64)
        precoder_c64 = precoder.to(torch.complex64)
        
        # Effective Channel
        h_eff = torch.bmm(h_fading_c64, precoder_c64.unsqueeze(-1)).squeeze(-1)
        
        # Get Correct SNR
        current_snr = getattr(channel, 'actual_snr_db', 10.0)
        snr_tensor = torch.full((BATCH_SIZE_TEST, 1), current_snr, device=device)
        
        # Equalize
        equalized_complex = adaptive_mmse_equalization(rx_noisy_c64, h_eff, snr_tensor)
        equalized_real = torch.view_as_real(equalized_complex).view(BATCH_SIZE_TEST, -1)
        equalized_real = torch.clamp(equalized_real, -10.0, 10.0)

    # 6. Receiver / Decode
    # Note: Phase 2 usually checks if receiver can decode equalized symbols
    reconstructed_logits ,_= receiver(equalized_real, csi, noise_power_tensor, rate_one_hot=rate_one_hot)
    predicted_bits = (torch.sigmoid(reconstructed_logits) > 0.5).float()
    
    # 7. Calculate Accuracy
    correct_bits = torch.sum(predicted_bits[:, :rate_bps] == test_data).item()
    total_bits = BATCH_SIZE_TEST * rate_bps
    accuracy = (correct_bits / total_bits) * 100.0
    
    return accuracy

# -----------------------------------------------------------------------------
# Function 1: Fundamentals Diagnostic
# -----------------------------------------------------------------------------
def check_receiver_fundamentals_phase2_dyn(epoch, transmitter, receiver, device, training_history):
    """
    🎯 Phase 2 Receiver Fundamentals - Updated for Physics Compliance
    """
    print(f"\n🎯 Epoch {epoch+1} - Receiver Fundamentals Diagnostic")
    print("=" * 60)
    
    baseline = training_history.get('best_accuracy', 50.0)
    
    # 🎯 Adjust plan based on baseline (No changes to logic, just context)
    if baseline <= 60.0:
        test_scenarios = {
            'Basic Decoding': {'rates': [1, 2], 'snrs': [10, 12, 14], 'focus': 'Foundations'},
        }
        success_threshold = 60.0
    elif baseline <= 75.0:
        test_scenarios = {
            'Consolidation': {'rates': [1, 2, 3], 'snrs': [8, 10, 12], 'focus': 'Stability'},
            'Expansion': {'rates': [4, 5], 'snrs': [10, 12, 14], 'focus': 'Mid-Rate'},
        }
        success_threshold = 70.0
    else:
        test_scenarios = {
            'Low Rate Stability': {'rates': [1, 2, 3], 'snrs': [5, 8, 10], 'focus': 'Robustness'},
            'Mid Rate Reliability': {'rates': [4, 5, 6], 'snrs': [8, 10, 12], 'focus': 'Reliability'},  
            'High Rate Challenge': {'rates': [7, 8], 'snrs': [12, 15, 18], 'focus': 'Performance Limit'},
        }
        success_threshold = 80.0
    
    print(f"   📊 Baseline: {baseline:.1f}% | Threshold: {success_threshold:.1f}%")
    
    transmitter.eval(); receiver.eval()
    
    scenario_results = {}
    
    with torch.no_grad():
        for scenario_name, config in test_scenarios.items():
            print(f"\n🔍 {scenario_name} - {config['focus']}")
            print("-" * 50)
            
            scenario_scores = []
            
            for bps in config['rates']:
                rate_accuracies = []
                
                for snr_db in config['snrs']:
                    # 🚨 CALL NEW HELPER
                    accuracy = _run_physics_compliant_test(
                        transmitter, receiver, device, 
                        rate_bps=bps, snr_db=snr_db
                    )
                    rate_accuracies.append(accuracy)
                    
                    # Status Check
                    relative_perf = accuracy - baseline
                    if relative_perf > 10: status = "🎯 Leading"
                    elif relative_perf > 5: status = "✅ Good"
                    elif relative_perf >= 0: status = "📈 Par" 
                    else: status = "🔄 Lagging"
                    
                    print(f"   {bps}bps @ {snr_db:2d}dB: {accuracy:5.1f}% ({relative_perf:+.1f}) - {status}")
                
                avg_rate_acc = sum(rate_accuracies) / len(rate_accuracies)
                scenario_scores.append(avg_rate_acc)
                
                if avg_rate_acc >= success_threshold + 10: mastery = "🎯 Mastered"
                elif avg_rate_acc >= success_threshold: mastery = "✅ Pass"
                elif avg_rate_acc >= baseline: mastery = "📈 Improving"
                else: mastery = "🔄 Learning"
                
                print(f"   📊 {bps}bps Avg: {avg_rate_acc:.1f}% - {mastery}")
            
            scenario_avg = sum(scenario_scores) / len(scenario_scores)
            scenario_results[scenario_name] = scenario_avg
            
            if scenario_avg >= success_threshold: scenario_status = "✅ Goal Met"
            elif scenario_avg >= baseline: scenario_status = "📈 Progressing"
            else: scenario_status = "🔄 Needs Focus"
            
            print(f"   🎯 {scenario_name} Overall: {scenario_avg:.1f}% - {scenario_status}")
    
    overall_avg = sum(scenario_results.values()) / len(scenario_results) if scenario_results else baseline
    training_history['best_accuracy'] = max(baseline, overall_avg)
    
    return training_history

# -----------------------------------------------------------------------------
# Function 2: Robustness Diagnostic
# -----------------------------------------------------------------------------
def check_receiver_robustness_phase2_dyn(epoch, scout, transmitter, receiver, device, training_history=None):
    """
    🎯 Phase 2 Receiver Robustness - Updated for Physics Compliance
    """
    print(f"\n🛡️  Epoch {epoch+1} - Receiver Robustness Test")
    print("=" * 60)
    
    if training_history is None:
        training_history = {'best_accuracy': 50.0, 'epoch_progress': [], 'current_stage': 'initial'}
    
    baseline_accuracy = training_history['best_accuracy']
    
    # Expectations logic
    if baseline_accuracy <= 55.0: expectations = {'good': 55.0, 'excellent': 60.0, 'stage': 'Initial'}
    elif baseline_accuracy <= 70.0: expectations = {'good': baseline_accuracy + 5, 'excellent': baseline_accuracy + 10, 'stage': 'Basic'}
    elif baseline_accuracy <= 85.0: expectations = {'good': baseline_accuracy + 3, 'excellent': baseline_accuracy + 8, 'stage': 'Intermediate'}
    else: expectations = {'good': 90.0, 'excellent': 95.0, 'stage': 'Advanced'}
    
    print(f"   📊 Baseline: {baseline_accuracy:.1f}% | Stage: {expectations['stage']}")
    
    scout.eval(); transmitter.eval(); receiver.eval()
    
    current_best = baseline_accuracy
    
    with torch.no_grad():
        # --- Test 1: Channel Distance Adaptation ---
        print(f"\n🔧 Channel Adaptation (Distance)")
        print("-" * 50)
        distances = [50, 200, 500, 1000] # Added 1000m for physics check
        
        for distance in distances:
            # We use a mid-range rate (e.g., 4bps) to test distance sensitivity
            # 🚨 CALL NEW HELPER with distance
            accuracy = _run_physics_compliant_test(
                transmitter, receiver, device, 
                rate_bps=4, distance=distance
            )
            
            if accuracy >= expectations['excellent']: adaptability = "🎯 Excellent"
            elif accuracy >= expectations['good']: adaptability = "✅ Good" 
            elif accuracy > baseline_accuracy: adaptability = "📈 Improving"
            elif accuracy >= baseline_accuracy - 5: adaptability = "⚖️ Stable"
            else: adaptability = "🔄 Weak"
            
            improvement = accuracy - baseline_accuracy
            print(f"   Dist {distance}m (4bps): {accuracy:5.1f}% ({improvement:+.1f}) - {adaptability}")
            current_best = max(current_best, accuracy)

        # --- Test 2: Rate Switching ---
        print(f"\n🔧 Rate Switching Capability")
        print("-" * 50)
        rate_sequences = [[1,4,2,5,3], [4,7,2,6,3]]
        
        for rate_seq in rate_sequences:
            seq_accuracies = []
            for rate in rate_seq:
                # Test switching at a standard distance (e.g., 300m)
                acc = _run_physics_compliant_test(
                    transmitter, receiver, device, 
                    rate_bps=rate, distance=300
                )
                seq_accuracies.append(acc)
            
            accuracy = sum(seq_accuracies) / len(seq_accuracies)
            
            if accuracy >= expectations['excellent']: switching = "🎯 Fluid"
            elif accuracy >= expectations['good']: switching = "✅ Stable"
            elif accuracy > baseline_accuracy: switching = "📈 Learning"
            elif accuracy >= baseline_accuracy - 5: switching = "⚖️ Basic"
            else: switching = "🔄 Struggling"
            
            improvement = accuracy - baseline_accuracy
            print(f"   Seq {rate_seq}: {accuracy:5.1f}% ({improvement:+.1f}) - {switching}")
            current_best = max(current_best, accuracy)
    
    training_history['best_accuracy'] = current_best
    training_history['epoch_progress'].append(current_best)
    training_history['current_stage'] = expectations['stage']
    
    progress_since_last = current_best - baseline_accuracy
    if progress_since_last > 5: overall = "🚀 Significant Progress"
    elif progress_since_last > 1: overall = "📈 Steady Progress" 
    elif progress_since_last >= 0: overall = "⚖️ Stable"
    else: overall = "🔄 Needs Adjustment"
    
    print(f"\n🏆 Best this Epoch: {current_best:.1f}% ({progress_since_last:+.1f}) - {overall}")
    
    return training_history    