import torch
import torch.nn as nn
import numpy as np

# =================================================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# =================================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RealisticConfig:
    CARRIER_FREQ_HZ = 2.6e9
    BANDWIDTH_HZ = 20e6
    TX_POWER_DBM = 23.0
    TX_ANTENNA_GAIN_DBI = 23.0
    RX_ANTENNA_GAIN_DBI = 5.0#0.0
    NOISE_FIGURE_DB = 7.0
    NUM_TX_ANTENNAS = 16
    NUM_RX_ANTENNAS = 4
    BS_HEIGHT_M = 30.0
    MS_HEIGHT_M = 1.5

# --- Training Parameters ---
RATES = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_RATE_BPS = 8
MAX_ANTENNAS = RealisticConfig.NUM_TX_ANTENNAS
CSI_VECTOR_SIZE = RealisticConfig.NUM_RX_ANTENNAS * RealisticConfig.NUM_TX_ANTENNAS * 2

BATCH_SIZE = 2048
LATENT_DIM = 16
BATCHES_PER_EPOCH = 200
NUM_WORKERS = 4

# --- Curriculum Phase Markers ---
PHASE_1_END = 60
PHASE_2_LESSON_0_DURATION = 25
PHASE_2_LESSON_1_DURATION = 50
PHASE_2_LESSON_2_DURATION = 50
PHASE_2_DURATION = 150

PHASE_2_LESSON_0_END = PHASE_1_END + PHASE_2_LESSON_0_DURATION
PHASE_2_LESSON_1_END = PHASE_2_LESSON_0_END + PHASE_2_LESSON_1_DURATION
PHASE_2_LESSON_2_END = PHASE_2_LESSON_1_END + PHASE_2_LESSON_2_DURATION
PHASE_2_END = PHASE_1_END + PHASE_2_DURATION

PHASE_3_LESSON_1_END = PHASE_2_END + 150#200
PHASE_3_LESSON_2_END = PHASE_3_LESSON_1_END + 100#150
PHASE_3_END = PHASE_2_END + 400#500
#610-660-710-760-960-1210-1560-1960-2460
PHASE_4A_DURATION_PER_RATE = {
    # Fast track the easy ones
    0: 50, 1: 50, 2: 50, 
    # --- Transformer "Cold Start" ---
    # This is the first time the Transformer weights are trained. 
    # Needs extra time to stabilize.
    3: 200,  # 16-QAM (Index 3)
    
    # --- The Spiral Geometry Phase ---
    # These rates need to learn complex non-linear shapes.
    4: 250,  # 32-QAM (Index 4)
    5: 200,  # 64-QAM (Index 5)
    
    # --- The "High Definition" Phase ---
    # These require pixel-perfect decision boundaries.
    # Fourier features need many steps to fine-tune high frequencies.
    6: 150,  # 128-QAM (Index 6)
    7: 400   # 256-QAM (Index 7)
}

lesson_boundaries = []
accumulated_epochs = 0
for rate_idx in sorted(PHASE_4A_DURATION_PER_RATE.keys()):
    lesson_boundaries.append(accumulated_epochs)
    accumulated_epochs += PHASE_4A_DURATION_PER_RATE[rate_idx]

PHASE_4A_END = PHASE_3_END + sum(PHASE_4A_DURATION_PER_RATE.values())
PHASE_4_END = PHASE_4A_END + 500
TOTAL_EPOCHS = PHASE_4_END

# Constellation Power Targets
CONSTELLATION_POWER_TARGETS = {
    1: 1.0, 2: 2.0, 3: 5.0, 4: 10.0,
    5: 20.0, 6: 42.0, 7: 95.0, 8: 190.0
}

# Rate-specific configurations
RATE_SPECIFIC_CONFIG = {
    1: {'latent_dim': 8, 'embed_dim': 32, 'hidden_dim': 64},
    2: {'latent_dim': 16, 'embed_dim': 48, 'hidden_dim': 96},
    3: {'latent_dim': 24, 'embed_dim': 56, 'hidden_dim': 112},
    4: {'latent_dim': 32, 'embed_dim': 64, 'hidden_dim': 128},
    5: {'latent_dim': 48, 'embed_dim': 80, 'hidden_dim': 192},
    6: {'latent_dim': 80, 'embed_dim': 128, 'hidden_dim': 256},
    7: {'latent_dim': 128, 'embed_dim': 160, 'hidden_dim': 320},
    8: {'latent_dim': 160, 'embed_dim': 192, 'hidden_dim': 384}
}