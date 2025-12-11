import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from config import *
from models import *
from channel import *
from utils import *

def get_gumbel_softmax_tau(epoch):
    """Annealing schedule for exploration"""
    if epoch >= PHASE_3_END and epoch < PHASE_4A_END:
        start_tau = 5.0
        end_tau = 0.8
        progress = (epoch - PHASE_3_END) / (PHASE_4A_END - PHASE_3_END)
        progress = np.clip(progress, 0.0, 1.0)
        current_tau = start_tau + progress * (end_tau - start_tau)
        return current_tau
    elif epoch >= PHASE_4A_END:
        rl_phase_start_epoch = PHASE_4A_END
        rl_phase_duration = max(1, TOTAL_EPOCHS - rl_phase_start_epoch)
        annealing_point = rl_phase_start_epoch + (rl_phase_duration * 0.5)

        start_tau = 5.0
        end_tau = 0.5
        
        if epoch < annealing_point:
            return start_tau
        else:
            progress = (epoch - annealing_point) / (TOTAL_EPOCHS - annealing_point)
            progress = np.clip(progress, 0.0, 1.0)
            current_tau = start_tau - progress * (start_tau - end_tau)
            return current_tau
    else:
        return 0.5

def get_channel_curriculum(epoch, batch_size, device):
    """
    Updated Channel Curriculum - Physics & Data Log Calibrated
    
    Progression:
    1. Phase 1: High SNR, Strong Line-of-Sight (Easy)
    2. Phase 2: Dropping SNR, Introduction of Scattering (Medium)
    3. Phase 3: Low SNR, Long Distance, Rayleigh Fading (Hard)
    """
    
    # ---------------------------------------------------------
    # PHASE 1: INITIALIZATION (Sanity Check)
    # Goal: Ensure gradients flow and model learns shapes.
    # Physics: Strong Rician (LOS), High SNR.
    # ---------------------------------------------------------
    if epoch < PHASE_1_END:
        return RealisticMIMOChannel(
            batch_size=batch_size, 
            device=device, 
            use_fading=True,      # Always use fading in MIMO (otherwise H is diagonal)
            distance_meters=100,  # Short distance
            fixed_snr_db=20.0,    # Clean signal (Easy)
            rician_k_factor=10.0  # Strong Line-of-Sight (Stable H matrix)
        )
    
    # ---------------------------------------------------------
    # PHASE 2: RECEIVER ROBUSTNESS
    # Goal: Train MMSE and Demodulator to handle noise.
    # ---------------------------------------------------------
    elif epoch < PHASE_2_LESSON_1_END:
        # Lesson 2-1: Lower SNR, still stable channel
        return RealisticMIMOChannel(
            batch_size=batch_size,
            device=device,
            use_fading=True,
            distance_meters=200,
            fixed_snr_db=15.0,    # Drop to 15dB
            rician_k_factor=5.0   # Moderate LOS
        )
    
    elif epoch < PHASE_2_LESSON_2_END:
        # Lesson 2-2: Transition to Rayleigh (Scattered)
        # Rician K=0 is equivalent to Rayleigh (NLOS) - Harder!
        return RealisticMIMOChannel(
            batch_size=batch_size,
            device=device,
            use_fading=True,
            distance_meters=300,
            fixed_snr_db=12.0,    # Drop to 12dB
            rician_k_factor=0.0   # Pure Scattering (Harder H matrix)
        )
    
    elif epoch < PHASE_2_END:
        # Lesson 2-3: SNR Variance
        # Sample SNR to teach adaptability
        random_snr = np.random.uniform(8.0, 15.0)
        return RealisticMIMOChannel(
            batch_size=batch_size,
            device=device,
            use_fading=True,
            distance_meters=400,
            fixed_snr_db=random_snr,
            rician_k_factor=0.0
        )
    
    # ---------------------------------------------------------
    # PHASE 3: END-TO-END (Transmitter Beamforming)
    # Goal: Deep Learning of Precoding for difficult channels.
    # ---------------------------------------------------------
    elif epoch < PHASE_3_LESSON_1_END:
        # Lesson 3-1: Mid Range, Mid SNR
        random_dist = np.random.uniform(200, 600)
        random_snr = np.random.uniform(5.0, 12.0)
        return RealisticMIMOChannel(
            batch_size=batch_size,
            device=device,
            use_fading=True,
            distance_meters=random_dist,
            fixed_snr_db=random_snr
        )
    
    elif epoch < PHASE_3_LESSON_2_END:
        # Lesson 3-2: Long Distance, Low SNR
        # Matches your log data (~1000m range)
        random_dist = np.random.uniform(600, 1200)
        random_snr = np.random.uniform(0.0, 8.0)
        return RealisticMIMOChannel(
            batch_size=batch_size,
            device=device,
            use_fading=True,
            distance_meters=random_dist,
            fixed_snr_db=random_snr
        )
    
    elif epoch < PHASE_3_END:
        # Lesson 3-3: "The Cell Edge" (Very Hard)
        # Prepares for the hardest MOE expert (Rate 1 @ -5dB)
        random_dist = np.random.uniform(1000, 2000)
        random_snr = np.random.uniform(-5.0, 5.0) # Crucial: Training on negative SNR
        return RealisticMIMOChannel(
            batch_size=batch_size,
            device=device,
            use_fading=True,
            distance_meters=random_dist,
            fixed_snr_db=random_snr
        )
    
    # ---------------------------------------------------------
    # PHASE 4: MOE / GENERALIZATION
    # ---------------------------------------------------------
    else:

        random_distance = np.random.uniform(low=50, high=2500)

        random_snr = np.random.uniform(-5.0, 30.0)
        
        return RealisticMIMOChannel(
            batch_size=batch_size,
            device=device,
            use_fading=True,
            distance_meters=random_distance,
            fixed_snr_db=random_snr
        )
def get_training_mode(epoch, batch_idx, batches_per_epoch):
    """Determine training mode"""
    if epoch < PHASE_1_END:
        force_rate = batch_idx % len(RATES)
        return ('constellation', force_rate)    
    elif epoch < PHASE_4A_END:
        force_rate = batch_idx % len(RATES)
        return ('supervised', force_rate)
    else:
        if batch_idx < int(batches_per_epoch * 0.95):
            return ('supervised_rl', None)
        else:
            return ('reinforcement', None)
def get_learning_rate_multipliers_2(epoch):
    """Learning rate multipliers for different phases"""
    multipliers = {}

    # --- PHASE 1: Constellation Foundation ---
    if epoch < PHASE_1_END:
        peak_lr = 2e-4
        end_lr = 1e-6
        plateau_end_epoch = int(PHASE_1_END * 0.75)
        if epoch < plateau_end_epoch:
            lr = peak_lr
        else:
            decay_epochs = PHASE_1_END - plateau_end_epoch
            progress_in_decay = (epoch - plateau_end_epoch) / decay_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress_in_decay))
            lr = end_lr + (peak_lr - end_lr) * cosine_decay
        multipliers['receiver'] = lr
        multipliers['other'] = lr

    # --- PHASE 2: Receiver Apprentice (Unchanged) ---
    elif epoch < PHASE_2_END:
        start_lr, end_lr = 5e-5, 1e-5
        progress = (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        lr = end_lr + (start_lr - end_lr) * cosine_decay
        multipliers['receiver'] = lr
        multipliers['other'] = lr

    # --- PHASE 3 & 4 (Complex Phases) ---
    else:
        base_start_lr = 2e-5
        base_end_lr = 1e-6
        if TOTAL_EPOCHS > PHASE_2_END:
            progress_total = (epoch - PHASE_2_END) / (TOTAL_EPOCHS - PHASE_2_END)
        else:
            progress_total = 1.0
        progress_total = np.clip(progress_total, 0.0, 1.0)
        cosine_decay_total = 0.5 * (1 + np.cos(np.pi * progress_total))
        default_slow_lr = base_end_lr + (base_start_lr - base_end_lr) * cosine_decay_total
        
        receiver_lr = default_slow_lr
        other_lr = default_slow_lr
        
        # Phase 3 Logic (Unchanged)
        if epoch < PHASE_3_LESSON_1_END:  
            receiver_lr = 0.0  
            other_lr = default_slow_lr
        elif epoch < PHASE_3_LESSON_2_END: 
            receiver_lr = default_slow_lr * 0.1 
            other_lr = default_slow_lr
        elif epoch < PHASE_3_END: 
            receiver_lr = default_slow_lr * 0.3 
            other_lr = default_slow_lr
            
        # Phase 4A: MOE Synergy (Calls Updated Helpers)
        elif epoch >= PHASE_3_END and epoch < PHASE_4A_END:
            current_moe_index, moe_progress = get_current_moe_progress(epoch)
            receiver_lr = get_moe_adaptive_learning_rate(current_moe_index, moe_progress)
            other_lr = default_slow_lr
            
        # Phase 4B: Manager RL (Unchanged)
        elif epoch >= PHASE_4A_END:
            manager_jolt_lr = 3e-5
            other_lr = manager_jolt_lr
            receiver_lr = default_slow_lr

        multipliers['receiver'] = receiver_lr
        multipliers['other'] = other_lr

    # Equalizer Logic
    if 'equalizer' not in multipliers:
        if epoch >= PHASE_3_END and epoch < PHASE_4A_END:
            current_moe_index, moe_progress = get_current_moe_progress(epoch)
            multipliers['equalizer'] = get_moe_equalizer_learning_rate(current_moe_index, moe_progress)
        else:
            multipliers['equalizer'] = multipliers.get('receiver', 1.0) * 0.3       
            
    return multipliers

def get_learning_rate_multipliers_2_old(epoch):
    """Learning rate multipliers for different phases"""
    multipliers = {}

    if epoch < PHASE_1_END:
        peak_lr = 2e-4
        end_lr = 1e-6
        plateau_end_epoch = int(PHASE_1_END * 0.75)
        if epoch < plateau_end_epoch:
            lr = peak_lr
        else:
            decay_epochs = PHASE_1_END - plateau_end_epoch
            progress_in_decay = (epoch - plateau_end_epoch) / decay_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress_in_decay))
            lr = end_lr + (peak_lr - end_lr) * cosine_decay
        multipliers['receiver'] = lr
        multipliers['other'] = lr

    elif epoch < PHASE_2_END:
        start_lr, end_lr = 5e-5, 1e-5
        progress = (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        lr = end_lr + (start_lr - end_lr) * cosine_decay
        multipliers['receiver'] = lr
        multipliers['other'] = lr

    else:
        base_start_lr = 2e-5
        base_end_lr = 1e-6
        if TOTAL_EPOCHS > PHASE_2_END:
            progress_total = (epoch - PHASE_2_END) / (TOTAL_EPOCHS - PHASE_2_END)
        else:
            progress_total = 1.0
        progress_total = np.clip(progress_total, 0.0, 1.0)
        cosine_decay_total = 0.5 * (1 + np.cos(np.pi * progress_total))
        default_slow_lr = base_end_lr + (base_start_lr - base_end_lr) * cosine_decay_total
        
        receiver_lr = default_slow_lr
        other_lr = default_slow_lr
        if epoch < PHASE_3_LESSON_1_END:  
            receiver_lr = 0.0  
            other_lr = default_slow_lr
            
        elif epoch < PHASE_3_LESSON_2_END:  
            receiver_lr = default_slow_lr * 0.1  
            other_lr = default_slow_lr
            
        elif epoch < PHASE_3_END:  
            receiver_lr = default_slow_lr * 0.3  
            other_lr = default_slow_lr
        elif epoch >= PHASE_4A_END:
            manager_jolt_lr = 3e-5
            other_lr = manager_jolt_lr
            receiver_lr = default_slow_lr
        elif epoch >= PHASE_3_END and epoch < PHASE_4A_END:
            current_moe_index, moe_progress = get_current_moe_progress(epoch)
            receiver_lr = get_moe_adaptive_learning_rate(current_moe_index, moe_progress)
            other_lr = default_slow_lr
            


        multipliers['receiver'] = receiver_lr
        multipliers['other'] = other_lr
    if 'equalizer' not in multipliers:
        if epoch >= PHASE_3_END and epoch < PHASE_4A_END:
            current_moe_index, moe_progress = get_current_moe_progress(epoch)
            multipliers['equalizer'] = get_moe_equalizer_learning_rate(current_moe_index, moe_progress)
        else:
            multipliers['equalizer'] = multipliers.get('receiver', 1.0) * 0.3       
    return multipliers
def get_moe_adaptive_learning_rate(moe_index, progress):
    """
    Adaptive Learning Rate for MOE Specialists.
    UPDATED: Significantly boosted for High Rates (5, 6, 7, 8) to handle QAM complexity.
    """
    # MOE-specific Base Learning Rates
    moe_base_lrs = {
        0: 4.5e-5,  # 1bps
        1: 4.0e-5,  # 2bps
        2: 3.5e-5,  # 3bps
        3: 3.0e-5,  # 4bps
        4: 2.8e-5,  # 5bps (Slight boost)
        5: 3.0e-5,  # 6bps (Boosted from 2.0e-5) -> Needs energy for 32/64 QAM
        6: 3.5e-5,  # 7bps (Boosted from 1.5e-5) -> Hard task
        7: 4.0e-5   # 8bps (Boosted from 1.0e-5) -> Hardest task
    }
    
    base_lr = moe_base_lrs.get(moe_index, 2.5e-5)
    
    # Adaptive Decay Schedule
    # For High Rates (Complexity Wall).
    if moe_index >= 5: # Rates 6, 7, 8
        if progress < 0.5:  # First 50%: Full speed
            return base_lr
        elif progress < 0.8: # Next 30%: Gentle decay
            return base_lr * 0.7
        else: # Final 20%: Fine tuning
            return base_lr * 0.4
    else:
        # Standard Decay for Lower Rates (easier to converge)
        if progress < 0.3:
            return base_lr
        elif progress < 0.7:
            return base_lr * 0.5
        else:
            return base_lr * 0.2

def get_moe_equalizer_learning_rate(moe_index, progress):
    """
    Adaptive Equalizer Learning Rate.
    """
    receiver_lr = get_moe_adaptive_learning_rate(moe_index, progress)
    
    # Equalizer Ratios
    # High rates need the Equalizer to work harder to untwist the signal
    equalizer_ratios = {
        0: 0.3,
        1: 0.25,
        2: 0.2,
        3: 0.2,   # Boosted slightly
        4: 0.15,  # Boosted
        5: 0.15,  # Boosted
        6: 0.12,  # Boosted
        #7: 0.10
        # 🚨 BOOST RATE 8 EQUALIZER，this is changed after 65% epochs
        # Old: 0.10. 
        # New: 1.0 (Same speed as Receiver). 
        # We need it to learn fast now that we woke it up.
        7: 1.0         
    }
    
    ratio = equalizer_ratios.get(moe_index, 0.1)
    return receiver_lr * ratio

def get_current_moe_progress(epoch):
    phase_4a_start = PHASE_3_END
    epochs_into_phase4a = epoch - phase_4a_start
    
    accumulated_epochs = 0
    for rate_idx, duration in PHASE_4A_DURATION_PER_RATE.items():
        if epochs_into_phase4a < accumulated_epochs + duration:
            current_moe_index = rate_idx
            moe_progress = (epochs_into_phase4a - accumulated_epochs) / duration
            return current_moe_index, moe_progress
        accumulated_epochs += duration
    
    return len(PHASE_4A_DURATION_PER_RATE) - 1, 1.0
def get_moe_synergy_curriculum(epoch, batch_size, device):
    """
  
    Design Philosophy:
    - Calibrated to your data log: 2000m = ~8dB, 500m = ~29dB.
    - Rate 0 (1bps) needs ~ -5dB. We must ADD noise to the 2000m channel.
    - Rate 7 (8bps) needs ~ 25dB. We must ADD noise to the 100m channel (56dB is too easy).
    """
    phase_4a_start = PHASE_3_END
    
    # Identify Current Lesson
    lesson_durations = PHASE_4A_DURATION_PER_RATE
    current_lesson_index = 0
    accumulated_epochs = 0
    for rate_idx, duration in lesson_durations.items():
        if epoch - phase_4a_start < accumulated_epochs + duration:
            current_lesson_index = rate_idx
            break
        accumulated_epochs += duration
    
    force_rate = current_lesson_index

    # -------------------------------------------------------------------------
    # 🎯 Calibrated Lesson Plans
    # Format: (Min Dist, Max Dist, Log10_Min_Noise, Log10_Max_Noise)
    # Log10(1.0) = 0.0 (No change)
    # Log10(10.0) = 1.0 (10x Noise -> Drops SNR by 10dB)
    # -------------------------------------------------------------------------
    lesson_plans = {
        # NEW TARGET: 0 dB to 6 dB
        # Base (1200-2000m) is ~8-12dB.
        # Need drop of ~2-10dB.
        # Multiplier 1.5x (0.2) to 10x (1.0)
        0: (600, 2000, 0.2, 1.0), 

        # Rate 1 (2bps): Ease this off slightly too
        # Target: 3 dB to 10 dB
        1: (600, 1800, 0.0, 0.8),

        # Rate 2 (3bps): Target 5 to 12 dB
        # Base (1000-1800m) is ~9-15dB. Need drop of ~2-6dB.
        2: (500, 1800, 0.2, 0.6),

        # Rate 3 (4bps): Target 8 to 15 dB
        # Base (800-1500m) is ~12-18dB. Need drop of ~0-4dB.
        3: (400, 1500, 0.0, 0.4),

        # Rate 4 (5bps): Target 12 to 18 dB
        # Base (600-1200m) is ~18-24dB. Need drop of ~6dB.
        4: (300, 1200, 0.2, 0.6),

        # Rate 5 (6bps): Target 15 to 22 dB
        # Base (400-900m) is ~24-35dB. Need drop ~10dB.
        5: (200, 900, 0.5, 1.0),

        # Rate 6 (7bps): Target 18 to 25 dB
        # Base (200-600m) is ~30-45dB. Need drop ~15dB.
        6: (100, 600, 1.0, 1.5),

        # Rate 7 (8bps): Target 22 to 30 dB
        # Base (50-400m) is ~40-60dB. Need drop ~20-30dB.
        # Note: We intentionally add noise here to prevent SNR from being "Infinite".
        7: (50, 400, 1.5, 2.5)
    }
    
    # Get parameters for current lesson
    if current_lesson_index in lesson_plans:
        plan = lesson_plans[current_lesson_index]
        random_distance = np.random.uniform(low=plan[0], high=plan[1])
        noise_multiplier = 10**np.random.uniform(plan[2], plan[3])
    else:
        # Fallback / Generalization Phase
        random_distance = np.random.uniform(low=50, high=2000)
        noise_multiplier = 10**np.random.uniform(-0.5, 1.5)

    # -------------------------------------------------------------------------
    # 🔌 Channel Instantiation & Fix
    # -------------------------------------------------------------------------
    channel = RealisticMIMOChannel(
        batch_size, 
        device, 
        use_fading=True, 
        distance_meters=random_distance,
        fixed_snr_db=None # Explicitly use Physics mode (Distance based)
    )
    
    # Apply Noise Injection (The "Interference" Simulation)
    if noise_multiplier != 1.0:
        channel.noise_power_watts *= noise_multiplier
        channel.noise_std_dev = np.sqrt(channel.noise_power_watts / 2.0)
        
        # Without this, MMSE uses the old low-noise SNR and fails to equalize the new noisy signal.
        current_rx_power = channel.expected_rx_power_watts 
        new_snr_linear = current_rx_power / (channel.noise_power_watts + 1e-12)
        channel.actual_snr_db = 10 * np.log10(new_snr_linear + 1e-12)
            
    return channel, force_rate

def actor_critic_loss(rate_probs, rate_one_hot, predicted_values, rewards, batch_idx, epoch, 
                     snr_db, theoretical_max_bps, achieved_throughput, performance_gap):
    """
    Actor-critic loss function.
    UPDATED: Added Advantage Normalization for training stability.
    """
    # 1. Calculate Raw Advantage (Reward - Baseline)
    # Detach predicted_values so we don't backpropagate through the Critic here (Actor only)
    raw_advantage = rewards - predicted_values.detach()
    
    # This prevents the "Exploding Gradient" problem when rewards are large.
    # It ensures the Actor learns consistently across different SNR ranges.
    advantage = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-9)
    
    # 2. Actor Loss (Policy Gradient)
    log_probs = torch.log(torch.clamp(rate_probs, min=1e-9))
    selected_log_probs = torch.sum(log_probs * rate_one_hot, dim=1)
    
    # advantage_scale can be removed or kept at 1.0 since we normalized
    actor_loss = -(selected_log_probs * advantage).mean()
    
    # 3. Critic Loss (Value Prediction)
    # Critic tries to predict the raw reward (MSE)
    critic_loss = F.mse_loss(predicted_values, rewards)
    
    # 4. Entropy Bonus (Exploration)
    # Encourage the model to keep options open
    entropy = -torch.sum(rate_probs * log_probs, dim=1).mean()
    
    # Dynamic Entropy Scaling (Optional but recommended)
    # Reduce exploration as time goes on
    entropy_weight = 0.05 if epoch > PHASE_4A_END + 200 else 0.1

    # 5. Total Loss
    # Standard weighting: 1.0 Actor + 0.5 Critic - Entropy
    total_loss = actor_loss + (0.5 * critic_loss) - (entropy_weight * entropy)

       
    return total_loss
def enhanced_exploration(rate_logits, snr_db, epoch, total_epochs=2000):
    """Enhanced exploration for RL"""
    theoretical_max_bps = get_shannon_capacity(snr_db)
    
    initial_temp = 2.5
    final_temp = 0.4
    current_temp = initial_temp * (final_temp / initial_temp) ** (epoch / total_epochs)
    
    capacity_low = theoretical_max_bps < 2.0
    capacity_medium_low = (theoretical_max_bps >= 2.0) & (theoretical_max_bps < 3.0)
    capacity_medium = (theoretical_max_bps >= 3.0) & (theoretical_max_bps < 4.0)
    capacity_medium_high = (theoretical_max_bps >= 4.0) & (theoretical_max_bps < 6.0)
    capacity_high = theoretical_max_bps >= 6.0
    
    if capacity_low.any():
        low_bias = torch.tensor([3.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0], device=rate_logits.device)
        rate_logits[capacity_low] += low_bias * 0.6
    
    if capacity_medium_low.any():
        medium_low_bias = torch.tensor([1.5, 1.0, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5], device=rate_logits.device)
        rate_logits[capacity_medium_low] += medium_low_bias * 0.4
    
    if capacity_medium.any():
        medium_bias = torch.tensor([-0.5, 0.8, 1.2, 0.3, -0.5, -1.5, -2.5, -3.5], device=rate_logits.device)
        rate_logits[capacity_medium] += medium_bias * 0.3
    
    if capacity_medium_high.any():
        medium_high_bias = torch.tensor([-1.5, -0.5, 0.5, 1.0, 0.8, 0.3, -0.5, -1.5], device=rate_logits.device)
        rate_logits[capacity_medium_high] += medium_high_bias * 0.2
    
    if capacity_high.any():
        high_bias = torch.tensor([-2.0, -1.0, -0.5, 0.3, 0.8, 1.0, 0.8, 0.5], device=rate_logits.device)
        rate_logits[capacity_high] += high_bias * 0.2
    
    progress = epoch / total_epochs
    eight_bps_penalty_strength = 0.3 * (1.0 - progress)
    eight_bps_penalty = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0], device=rate_logits.device)
    rate_logits += eight_bps_penalty * eight_bps_penalty_strength
    
    if epoch > total_epochs * 0.5:
        exploration_noise = torch.randn_like(rate_logits) * 0.05
        rate_logits += exploration_noise
    
    return rate_logits / current_temp
    
 
