import torch
import numpy as np
from config import *

class RealisticMIMOChannel:
    def __init__(self, batch_size, device, use_fading=True, distance_meters=500,
                 fixed_snr_db=None, snr_range_db=None, rician_k_factor=None,
                 verbose=False):
        self.device = device
        self.batch_size = batch_size
        self.use_fading = use_fading
        self.rician_k_factor = rician_k_factor
        self.verbose = verbose
        self.distance_meters = distance_meters
        
        # 🚨 Fix: Unified Transmit Power
        self.tx_power_watts = 10**((RealisticConfig.TX_POWER_DBM - 30) / 10.0)
        
        # 🚨 Fix: Path Loss with immediate helper call
        total_path_loss_db = self._calculate_path_loss(distance_meters)
        
        total_antenna_gain_db = RealisticConfig.TX_ANTENNA_GAIN_DBI + RealisticConfig.RX_ANTENNA_GAIN_DBI
        effective_path_loss_db = total_path_loss_db - total_antenna_gain_db
        
        effective_path_loss_db = max(10, min(effective_path_loss_db, 150))
            
        self.effective_path_loss_db = effective_path_loss_db
        self.path_loss_linear = 10**(-effective_path_loss_db / 10.0)
        
        self.expected_rx_power_watts = self.tx_power_watts * self.path_loss_linear

        # Noise
        kT_dbm_hz = -174
        noise_bandwidth_db = 10 * np.log10(RealisticConfig.BANDWIDTH_HZ)
        noise_power_dbm = kT_dbm_hz + noise_bandwidth_db + RealisticConfig.NOISE_FIGURE_DB
        self.noise_power_watts = 10**((noise_power_dbm - 30) / 10.0)
        
        # SNR Logic
        if fixed_snr_db is not None:
            target_snr_linear = 10**(fixed_snr_db / 10.0)
            required_rx_power = target_snr_linear * self.noise_power_watts
            
            # 🚨 Fix: Use torch.sqrt compatible clipping
            self.tx_power_scale = required_rx_power / (self.tx_power_watts * self.path_loss_linear)
            self.tx_power_scale = float(np.clip(self.tx_power_scale, 1e-10, 1e10))
            
            actual_rx_power = self.tx_power_scale * self.tx_power_watts * self.path_loss_linear
            self.actual_snr_db = 10 * np.log10(actual_rx_power / self.noise_power_watts)
            self.target_snr_db = fixed_snr_db
        else:
            self.tx_power_scale = 1.0
            actual_snr_linear = self.expected_rx_power_watts / self.noise_power_watts
            self.actual_snr_db = 10 * np.log10(actual_snr_linear)
            self.target_snr_db = None

        self.noise_std_dev = np.sqrt(self.noise_power_watts / 2.0)
        self.H_fading = self._create_fading_matrix_batch()
        
        if self.verbose:
            self._print_channel_diagnostics()

    def _calculate_path_loss(self, distance_meters):
        wavelength = 3e8 / RealisticConfig.CARRIER_FREQ_HZ
        path_loss_exponent = 3.5 
        d0 = 10.0
        
        pl_d0 = 20 * np.log10(4 * np.pi * d0 / wavelength) + 10.0 
        
        if distance_meters < d0:
            total_path_loss_db = pl_d0
        else:
            total_path_loss_db = pl_d0 + 10 * path_loss_exponent * np.log10(distance_meters / d0)
            
        total_path_loss_db += np.random.normal(0, 8)
        return max(60.0, min(total_path_loss_db, 150.0))

    def _print_channel_diagnostics(self):
        print(f"\n📊 Channel Model Diagnostics - Dist: {self.distance_meters}m")
        print(f"   Actual SNR: {self.actual_snr_db:.1f} dB")
        if self.target_snr_db is not None:
            print(f"   Target SNR: {self.target_snr_db:.1f} dB")

    def _create_fading_matrix_batch(self):
        N_rx, N_tx = RealisticConfig.NUM_RX_ANTENNAS, RealisticConfig.NUM_TX_ANTENNAS
        normalization = 1.0 / np.sqrt(N_tx)       
        
        if not self.use_fading:
            H = torch.eye(N_rx, N_tx, dtype=torch.cfloat, device=self.device)
            return H.unsqueeze(0).repeat(self.batch_size, 1, 1) * normalization

        if self.rician_k_factor is not None and self.rician_k_factor > 0:
            K = self.rician_k_factor
            
            real_nlos = torch.randn(self.batch_size, N_rx, N_tx, device=self.device)
            imag_nlos = torch.randn(self.batch_size, N_rx, N_tx, device=self.device)
            H_nlos = torch.complex(real_nlos, imag_nlos) * np.sqrt(0.5)
            
            H_los = torch.ones(self.batch_size, N_rx, N_tx, dtype=torch.cfloat, device=self.device)
            
            K_tensor = torch.tensor(K, device=self.device, dtype=torch.float32)
            los_scale = torch.sqrt(K_tensor / (K_tensor + 1))
            nlos_scale = torch.sqrt(1.0 / (K_tensor + 1))
            
            H_combined = los_scale * H_los + nlos_scale * H_nlos
            return H_combined * normalization
        else:
            real_rayleigh = torch.randn(self.batch_size, N_rx, N_tx, device=self.device)
            imag_rayleigh = torch.randn(self.batch_size, N_rx, N_tx, device=self.device)
            H_rayleigh = torch.complex(real_rayleigh, imag_rayleigh) * np.sqrt(0.5)
            return H_rayleigh * normalization

    def get_csi(self):
        H_flat = self.H_fading.reshape(self.batch_size, -1)
        return torch.cat([H_flat.real, H_flat.imag], dim=1).float()

    def apply(self, tx_signal_complex, add_noise=True, return_components=False):
        # 🚨 Fix: Use torch.sqrt for differentiability
        scale = torch.tensor(self.tx_power_scale, device=self.device, dtype=torch.float32)
        tx_signal_scaled = tx_signal_complex * torch.sqrt(scale)
        
        if tx_signal_scaled.dim() == 2:
            tx_signal_3d = tx_signal_scaled.unsqueeze(-1)
        else:
            tx_signal_3d = tx_signal_scaled

        rx_signal_no_gain = torch.bmm(self.H_fading, tx_signal_3d)
        
        # 🚨 Fix: Explicit tensor creation for path loss to avoid dimension mismatch
        path_loss_scalar = torch.tensor(np.sqrt(self.path_loss_linear), device=self.device, dtype=torch.float32)
        rx_signal_no_noise = rx_signal_no_gain * path_loss_scalar
        
        if not add_noise:
            return rx_signal_no_noise.squeeze(-1)

        noise_real = torch.randn_like(rx_signal_no_noise.real) * self.noise_std_dev
        noise_imag = torch.randn_like(rx_signal_no_noise.imag) * self.noise_std_dev
        noise = torch.complex(noise_real, noise_imag)
        
        rx_noisy = rx_signal_no_noise + noise
        
        # 🚨 Fix: Improved SNR calculation (Avoids item() squashing)
        est_signal_power = torch.mean(torch.abs(rx_noisy)**2) - self.noise_power_watts
        est_signal_power = torch.max(est_signal_power, torch.tensor(1e-12, device=self.device))
        
        self.actual_snr_db = 10 * torch.log10(est_signal_power / self.noise_power_watts).item()
        
        if return_components:
            return rx_noisy.squeeze(-1), (rx_signal_no_noise.squeeze(-1), noise.squeeze(-1))        
        return rx_noisy.squeeze(-1)

    def reset_environment(self, distance_meters=None, fixed_snr_db=None, rician_k_factor=None):
        if rician_k_factor is not None:
            self.rician_k_factor = rician_k_factor

        if distance_meters is not None:
            self.distance_meters = distance_meters
            total_path_loss_db = self._calculate_path_loss(distance_meters)
            
            total_antenna_gain_db = RealisticConfig.TX_ANTENNA_GAIN_DBI + RealisticConfig.RX_ANTENNA_GAIN_DBI
            self.effective_path_loss_db = total_path_loss_db - total_antenna_gain_db
            self.path_loss_linear = 10**(-self.effective_path_loss_db / 10.0)
            self.expected_rx_power_watts = self.tx_power_watts * self.path_loss_linear

        if fixed_snr_db is not None:
            self.target_snr_db = fixed_snr_db
            target_snr_linear = 10**(fixed_snr_db / 10.0)
            required_rx_power = target_snr_linear * self.noise_power_watts
            
            self.tx_power_scale = required_rx_power / (self.tx_power_watts * self.path_loss_linear)
            self.tx_power_scale = float(np.clip(self.tx_power_scale, 1e-10, 1e10))
            
            actual_rx_power = self.tx_power_scale * self.tx_power_watts * self.path_loss_linear
            self.actual_snr_db = 10 * np.log10(actual_rx_power / self.noise_power_watts)
        else:
            self.target_snr_db = None
            self.tx_power_scale = 1.0
            actual_snr_linear = self.expected_rx_power_watts / self.noise_power_watts
            self.actual_snr_db = 10 * np.log10(actual_snr_linear)

        self.H_fading = self._create_fading_matrix_batch()

    def get_channel_info(self):
        H_magnitude = torch.abs(self.H_fading)
        H_phase = torch.angle(self.H_fading)
        
        info = {
            'distance_meters': self.distance_meters,
            'effective_path_loss_db': self.effective_path_loss_db,
            'expected_rx_power_dbm': 10*np.log10(self.expected_rx_power_watts)+30,
            'noise_power_dbm': 10*np.log10(self.noise_power_watts)+30,
            'actual_snr_db': self.actual_snr_db,
            'target_snr_db': self.target_snr_db,
            'tx_power_scale': self.tx_power_scale,
            'H_fading_magnitude_range': [H_magnitude.min().item(), H_magnitude.max().item()],
            'H_fading_real_range': [self.H_fading.real.min().item(), self.H_fading.real.max().item()],
        }
        return info



        self.H_fading = self._create_fading_matrix_batch()
