import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
class GaussianFourierProjection(nn.Module):
    """
    Projects low-dimensional coordinates into high-dimensional Fourier features.
    This helps the network learn high-frequency functions (like fine QAM grids)
    overcoming the "Spectral Bias" of standard MLPs.
    
    Reference: "Fourier Features Let Networks Learn High Frequency Functions..." (Tancik et al.)
    """
    def __init__(self, input_dim, mapping_size=256, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        
        # Random Gaussian Matrix (Fixed, not trainable)
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        # x: [Batch, Input_Dim]
        # B: [Input_Dim, Mapping_Size]
        
        # 1. Project input onto random axis: (2 * pi * x @ B)
        x_proj = (2.0 * np.pi * x) @ self.B
        
        # 2. Compute Sin and Cos
        # Output dim becomes 2 * mapping_size
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Scales each sample individually so its RMS power is 1.0.
    Preserves relative amplitude between I and Q, but standardizes total energy.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape: [Batch, Dim]
        norm_x = x.norm(2, dim=1, keepdim=True) / (x.size(1) ** 0.5)
        return (x / (norm_x + self.eps)) * self.scale
        
class SimpleScaler(nn.Module):
    """
    Scales the input by a learnable constant. 
    Does NOT normalize per-sample, preserving Amplitude Information (crucial for QAM).
    """
    def __init__(self, dim, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init_scale))

    def forward(self, x):
        return x * self.scale
        
class ResidualWrapper(nn.Module):
    """Wrapper for residual connection"""
    def __init__(self, block):
        super(ResidualWrapper, self).__init__()
        self.block = block
        self.skip_gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x + self.skip_gate * self.block(x)

class NeuralEqualizer(nn.Module):
    """
    Enhanced Neural Equalizer with improved architecture for MIMO systems.
    UPDATED: Uses LayerNorm instead of BatchNorm for RL stability.
    """
    def __init__(self, input_size=10, output_size=8, hidden_dim=256, num_layers=4, dropout_rate=0.1):
        super(NeuralEqualizer, self).__init__()
        
        layers = []
        
        # Input Block
        layers.extend([
            nn.Linear(input_size, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim), # Changed from BatchNorm1d
            nn.Dropout(dropout_rate)
        ])
        
        # Residual Blocks
        for i in range(num_layers - 2):
            residual_block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.LayerNorm(hidden_dim), # Changed from BatchNorm1d
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.LayerNorm(hidden_dim), # Changed from BatchNorm1d
                nn.Dropout(dropout_rate)
            )
            layers.append(ResidualWrapper(residual_block))
        
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim // 2), # Changed from BatchNorm1d
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_size)
        ])
        
        self.net = nn.Sequential(*layers)
        self.correction_scale = nn.Parameter(torch.tensor(0.0))         
        self.feature_norm = nn.LayerNorm(input_size) # Changed from BatchNorm1d

    def forward(self, features):
        features_normalized = self.feature_norm(features)
        raw_correction = self.net(features_normalized)
        final_correction = self.correction_scale * raw_correction
        return final_correction

class EnhancedNeuralEqualizationCorrection(nn.Module):
    """
    Enhanced equalization correction that combines traditional MMSE with neural network
    """
    def __init__(self, hidden_size=256, num_layers=4, dropout_rate=0.1):
        super(EnhancedNeuralEqualizationCorrection, self).__init__()
        
        self.neural_equalizer = NeuralEqualizer(
            input_size=10, # 8(symbol) + 1(power) + 1(SNR) = 10
            output_size=8,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        
        self.feature_weights = nn.Parameter(torch.ones(10))
        self.attention_weights = nn.Parameter(torch.tensor([1.0, 0.0, 0.0])) 
        
    def forward(self, equalized_symbol, channel_power_normalized, snr_normalized):
        weighted_features = torch.cat([
            equalized_symbol * self.feature_weights[0:8],
            (channel_power_normalized * self.feature_weights[8]).unsqueeze(1),
            (snr_normalized * self.feature_weights[9]).unsqueeze(1)
        ], dim=1)
        
        correction = self.neural_equalizer(weighted_features)
        
        original_weight = self.attention_weights[0]
        correction_weight = self.attention_weights[1]
        
        corrected_symbol = (original_weight * equalized_symbol + 
                          correction_weight * correction)
        
        output_scale = torch.sqrt(original_weight**2 + correction_weight**2 + 1e-9)
        final_symbol = corrected_symbol / output_scale
        
        return final_symbol

class LearnedEqualizationCorrection(nn.Module):
    """Baseline equalization correction for comparison"""
    def __init__(self, hidden_size=128, num_layers=3):
        super(LearnedEqualizationCorrection, self).__init__()
        layers = []
        input_size = 10 
        for i in range(num_layers):
            output_size = hidden_size if i < num_layers - 1 else 2
            layers.extend([
                nn.Linear(input_size, output_size),
                nn.PReLU() if i < num_layers - 1 else nn.Identity()
            ])
            input_size = output_size
        self.correction_net = nn.Sequential(*layers)
        self.learned_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, equalized_symbol, channel_power, snr_feature):
        features = torch.cat([
            equalized_symbol,
            channel_power.unsqueeze(1),
            snr_feature.unsqueeze(1)
        ], dim=1)
        correction = self.correction_net(features)
        corrected_symbol = equalized_symbol + self.learned_scale * correction
        return corrected_symbol

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        # Updated to LayerNorm for consistency
        self.layer1 = nn.Sequential(nn.Linear(size, size), nn.PReLU(), nn.LayerNorm(size))
        self.layer2 = nn.Sequential(nn.Linear(size, size), nn.PReLU(), nn.LayerNorm(size))

    def forward(self, x):
        return x + self.layer2(self.layer1(x))

class ReceiverAttentionHead(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class ReceiverMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_size):
        super().__init__()
        self.heads = nn.ModuleList([ReceiverAttentionHead(embed_size, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class ReceiverFeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size), nn.PReLU(),
            nn.Linear(4 * embed_size, embed_size), nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)

class ReceiverTransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        head_size = embed_size // num_heads
        self.sa = ReceiverMultiHeadAttention(num_heads, head_size, embed_size)
        self.ffwd = ReceiverFeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NeuralConstellationEncoder(nn.Module):
    def __init__(self):
        super(NeuralConstellationEncoder, self).__init__()
        self.encoders = nn.ModuleList()
        self.power_scalers = nn.ParameterList()

        for bps in RATES:
            config = RATE_SPECIFIC_CONFIG[bps]
            num_points = 2**bps
            embedding = nn.Embedding(num_embeddings=num_points, embedding_dim=config['embed_dim'])
            
            if bps >= 5:
                shaper = self.create_high_rate_shaper(config['embed_dim'], config['hidden_dim'])
            else:             
                shaper = nn.Sequential(
                    nn.Linear(config['embed_dim'], config['hidden_dim']), nn.PReLU(),
                    nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2), nn.PReLU(),
                    nn.Linear(config['hidden_dim'] // 2, 2)
                )
            self.encoders.append(nn.ModuleDict({'embedding': embedding, 'shaper': shaper}))

            target_power = CONSTELLATION_POWER_TARGETS[bps]
            initial_scaler_value = np.sqrt(target_power)
            power_scaler = nn.Parameter(torch.tensor(initial_scaler_value, dtype=torch.float32))
            self.power_scalers.append(power_scaler)
            
    def create_high_rate_shaper(self, embed_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.PReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.PReLU(),
            nn.Linear(hidden_dim // 4, 2)
        )
            
    def forward(self, data_chunk, rate_one_hot):
        rate_index = torch.argmax(rate_one_hot[0]).item()
        bps = RATES[rate_index]
        bit_powers = 2**torch.arange(bps - 1, -1, -1, device=data_chunk.device, dtype=torch.long)
        relevant_bits = data_chunk[:, :bps]
        indices = torch.sum(relevant_bits.long() * bit_powers, dim=1)

        selected_encoder = self.encoders[rate_index]
        unique_vectors = selected_encoder['embedding'](indices)
        raw_points = selected_encoder['shaper'](unique_vectors)
        
        current_power = torch.mean(torch.sum(raw_points**2, dim=1))
        normalized_points = raw_points / torch.sqrt(current_power + 1e-9)
        
        power_scaler = F.relu(self.power_scalers[rate_index])
        scaled_points = normalized_points * power_scaler
        final_points = torch.clamp(scaled_points, min=-1e6, max=1e6)
        
        return final_points

class EnhancedScoutAI(nn.Module):
    def __init__(self, num_blocks=4, embed_size=256):
        super(EnhancedScoutAI, self).__init__()
        self.embedding = nn.Linear(CSI_VECTOR_SIZE, embed_size)
        self.res_blocks = nn.Sequential(*[ResidualBlock(embed_size) for _ in range(num_blocks)])
        self.output_head = nn.Linear(embed_size, CSI_VECTOR_SIZE)

    def forward(self, x):
        if x.dim() > 2: x = x.squeeze()
        if x.dim() == 1: x = x.unsqueeze(0)
        x = self.embedding(x)
        x = self.res_blocks(x)
        return self.output_head(x)

class LowRatePrecoder(nn.Module):
    def __init__(self):
        super(LowRatePrecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(CSI_VECTOR_SIZE, 128), nn.PReLU(),
            nn.Linear(128, MAX_ANTENNAS * 2)
        )

    def forward(self, csi):
        components = self.net(csi)
        components = torch.clamp(components, min=-1e6, max=1e6)
        components_reshaped = components.view(-1, MAX_ANTENNAS, 2)
        norm = torch.norm(components_reshaped, p=2, dim=[1, 2], keepdim=True)
        components_normalized = components_reshaped / (norm + 1e-9)
        
        beam_vector_complex = torch.complex(
            components_normalized[..., 0],
            components_normalized[..., 1]
        )
        return beam_vector_complex

class EnhancedHighRatePrecoder(nn.Module):
    def __init__(self, num_blocks=6, embed_size=512, num_attention_heads=8, dropout_rate=0.1):
        super(EnhancedHighRatePrecoder, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Linear(CSI_VECTOR_SIZE, embed_size)
        
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_attention_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.attention_norm = nn.LayerNorm(embed_size)
        
        self.res_blocks = nn.Sequential(*[
            EnhancedResidualBlock(embed_size, dropout_rate) for _ in range(num_blocks)
        ])
        
        self.output_head = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2), nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_size // 2, MAX_ANTENNAS * 2)
        )

    def forward(self, csi):
        x = self.embedding(csi)
        x_reshaped = x.unsqueeze(1)
        x_attn, attn_weights = self.channel_attention(x_reshaped, x_reshaped, x_reshaped)
        x_attn = self.attention_dropout(x_attn.squeeze(1))
        x = self.attention_norm(x + x_attn)
        x = self.res_blocks(x)
        
        vector_components = self.output_head(x)
        vector_components = torch.clamp(vector_components, min=-1e6, max=1e6)

        vector_complex_real_imag = vector_components.view(-1, MAX_ANTENNAS, 2)
        norm = torch.norm(vector_complex_real_imag, p=2, dim=[1, 2], keepdim=True)
        vector_normalized = vector_complex_real_imag / (norm + 1e-9)
        
        return torch.complex(vector_normalized[..., 0], vector_normalized[..., 1])

class EnhancedResidualBlock(nn.Module):
    def __init__(self, size, dropout_rate=0.1):
        super(EnhancedResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(size, size), nn.PReLU(),
            nn.Dropout(dropout_rate), nn.LayerNorm(size) # Changed to LayerNorm
        )
        self.layer2 = nn.Sequential(
            nn.Linear(size, size), nn.PReLU(), 
            nn.Dropout(dropout_rate), nn.LayerNorm(size) # Changed to LayerNorm
        )
        self.skip_gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)
        return identity + self.skip_gate * out

class EnhancedTransmitterAI(nn.Module):
    def __init__(self):
        super(EnhancedTransmitterAI, self).__init__()
        self.constellation_encoder = NeuralConstellationEncoder()
        
        self.precoders = nn.ModuleList([
            LowRatePrecoder(),
            LowRatePrecoder(),
            LowRatePrecoder(),
            EnhancedHighRatePrecoder(num_blocks=3, embed_size=256, num_attention_heads=4),
            EnhancedHighRatePrecoder(num_blocks=4, embed_size=384, num_attention_heads=6),
            EnhancedHighRatePrecoder(num_blocks=5, embed_size=448, num_attention_heads=7),
            EnhancedHighRatePrecoder(num_blocks=6, embed_size=512, num_attention_heads=8),
            EnhancedHighRatePrecoder(num_blocks=7, embed_size=576, num_attention_heads=8)
        ])
        
        self.ae_heads = nn.ModuleList()
        for bps in RATES:
            latent_dim = RATE_SPECIFIC_CONFIG[bps]['latent_dim']
            self.ae_heads.append(nn.Sequential(
                nn.Linear(CSI_VECTOR_SIZE, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, latent_dim),
                nn.LayerNorm(latent_dim) # Keeps latent vectors standardized
            ))
        
        self.tau = 1.0
        embed_size = 128
        num_heads = 4
        self.rate_head_csi_embedding = nn.Linear(CSI_VECTOR_SIZE, embed_size)
        self.rate_head_noise_embedding = nn.Linear(1, embed_size)
        self.rate_head_transformer = ReceiverTransformerBlock(embed_size, num_heads)
        self.rate_head_decoder = nn.Sequential(
            nn.Linear(embed_size * 2, 64), 
            nn.PReLU(),
            nn.Linear(64, len(RATES))
        )

    def set_tau(self, new_tau):
        self.tau = new_tau

    def forward(self, data_chunk, csi, noise_power_context, force_rate_index=None, override_rate_one_hot=None):
        if override_rate_one_hot is not None:
            rate_one_hot = override_rate_one_hot        
        elif force_rate_index is None:
            csi_emb = self.rate_head_csi_embedding(csi)
            noise_emb = self.rate_head_noise_embedding(noise_power_context.unsqueeze(1))
            x = torch.stack([csi_emb, noise_emb], dim=1)
            x = self.rate_head_transformer(x)
            x = x.view(x.shape[0], -1)
            rate_logits = self.rate_head_decoder(x)
            if self.training:
                rate_one_hot = F.gumbel_softmax(rate_logits, tau=self.tau, hard=True)
            else:
                chosen_rate_index = torch.argmax(rate_logits, dim=1)
                rate_one_hot = F.one_hot(chosen_rate_index, num_classes=len(RATES)).float()
        else:
            rate_one_hot = F.one_hot(
                torch.full((data_chunk.shape[0],), force_rate_index, device=data_chunk.device),
                num_classes=len(RATES)
            ).float()

        chosen_rate_index = torch.argmax(rate_one_hot, dim=1)
        beamforming_vector = torch.zeros((csi.shape[0], MAX_ANTENNAS), dtype=torch.cfloat, device=csi.device)

        for i in range(len(RATES)):
            mask = (chosen_rate_index == i)
            if mask.any():
                beamforming_vector[mask] = self.precoders[i](csi[mask])
        
        encoded_symbols = self.constellation_encoder(data_chunk, rate_one_hot)
        complex_symbols = torch.complex(encoded_symbols[:, 0], encoded_symbols[:, 1])
        
        max_latent_dim = 0
        for bps in RATES:
            max_latent_dim = max(max_latent_dim, RATE_SPECIFIC_CONFIG[bps]['latent_dim'])
            
        with torch.no_grad():
            sample_output = self.ae_heads[0](csi[:1])
            target_dtype = sample_output.dtype
        
        original_latent_vec = torch.zeros(
            (csi.shape[0], max_latent_dim), 
            device=csi.device,
            dtype=target_dtype
        )
        
        for i in range(len(RATES)):
            mask = (chosen_rate_index == i)
            if mask.any():
                output_from_head = self.ae_heads[i](csi[mask])
                current_latent_dim = output_from_head.shape[1]
                original_latent_vec[mask, :current_latent_dim] = output_from_head
        
        tx_signal_complex = beamforming_vector * complex_symbols.unsqueeze(-1)  
        
        return tx_signal_complex, encoded_symbols, rate_one_hot, original_latent_vec, beamforming_vector

class ChannelAutoencoderDecoder(nn.Module):
    def __init__(self):
        super(ChannelAutoencoderDecoder, self).__init__()
        self.decoder_heads = nn.ModuleList()
        for bps in RATES:
            latent_dim = RATE_SPECIFIC_CONFIG[bps]['latent_dim']
            input_dim = RealisticConfig.NUM_RX_ANTENNAS * 2 + CSI_VECTOR_SIZE + 1            
            head = nn.Sequential(
                nn.Linear(input_dim, 128), nn.PReLU(),
                nn.Linear(128, 64), nn.PReLU(),
                nn.Linear(64, latent_dim),
                nn.LayerNorm(latent_dim) # Keeps latent vectors standardized
            )
            self.decoder_heads.append(head)

    def forward(self, equalized_symbol, csi_context, noise_power, rate_one_hot):
        combined_input = torch.cat([equalized_symbol, csi_context, noise_power.unsqueeze(1)], dim=1)
        max_latent_dim = 0
        for bps in RATES:
            max_latent_dim = max(max_latent_dim, RATE_SPECIFIC_CONFIG[bps]['latent_dim'])

        reconstructed_latent_vec = torch.zeros(
            (csi_context.shape[0], max_latent_dim),
            device=csi_context.device,
            dtype=combined_input.dtype
        )
        
        batch_rate_indices = torch.argmax(rate_one_hot, dim=1)
        for i in range(len(self.decoder_heads)):
            mask = (batch_rate_indices == i)
            if mask.any():
                input_slice = combined_input[mask]
                output_from_head = self.decoder_heads[i](input_slice)
                current_latent_dim = output_from_head.shape[1]
                reconstructed_latent_vec[mask, :current_latent_dim] = output_from_head.to(reconstructed_latent_vec.dtype)

        return reconstructed_latent_vec

class EnhancedReceiverAI(nn.Module):
    def __init__(self, num_heads=4, num_blocks=4, use_neural_equalizer=True):
        super(EnhancedReceiverAI, self).__init__()
        
        self.use_neural_equalizer = use_neural_equalizer
        num_rx_features = RealisticConfig.NUM_RX_ANTENNAS*2
        
        self.input_stabilizers = nn.ModuleList()
        # 🚨 HYBRID STRATEGY UPDATED
        # Rates 1, 2, 3, 4 (BPSK -> 16-QAM) uses RMSNorm for Maximum Stability.
        # Rates 5, 6, 7, 8 (32-QAM -> 256-QAM) uses SimpleScaler for Geometry Preservation.
        for i in range(len(RATES)):
            bps = RATES[i]
            
            # Check if bps is 1, 2, 3, or 4
            if bps <= 4:
                # Stability Mode
                self.input_stabilizers.append(RMSNorm(num_rx_features))
            else:
                # Geometry Mode (High Rate)
                #target_power = CONSTELLATION_POWER_TARGETS[bps]
                #ideal_scale = 1.0 / (np.sqrt(target_power) + 1e-9)
                #ideal_scale = np.clip(ideal_scale, 0.05, 1.0)
                #self.input_stabilizers.append(SimpleScaler(num_rx_features, init_scale=ideal_scale))  
                # 🚨 FIX: Geometry Mode (High Rates)
                # Since we added AGC, the input is already ~1.0.
                # Initialize scale to 1.0 so we don't shrink it to death.
                self.input_stabilizers.append(SimpleScaler(num_rx_features, init_scale=1.0))                
        self.decoders = nn.ModuleList([
            self._create_mlp_decoder(RATES[0]),
            self._create_mlp_decoder(RATES[1]),
            self._create_mlp_decoder(RATES[2]),
            self._create_transformer_decoder(embed_size=256, num_heads=4, num_blocks=3, bps=RATES[3]),
            self._create_transformer_decoder(embed_size=384, num_heads=6, num_blocks=4, bps=RATES[4]),
            self._create_transformer_decoder(embed_size=448, num_heads=7, num_blocks=5, bps=RATES[5]),
            self._create_transformer_decoder(embed_size=512, num_heads=8, num_blocks=6, bps=RATES[6]),
            self._create_transformer_decoder(embed_size=576, num_heads=8, num_blocks=7, bps=RATES[7])
        ])
        
        if self.use_neural_equalizer:
            self.equalization_correctors = nn.ModuleList([
                EnhancedNeuralEqualizationCorrection(hidden_size=128, num_layers=3),
                EnhancedNeuralEqualizationCorrection(hidden_size=192, num_layers=3),
                EnhancedNeuralEqualizationCorrection(hidden_size=256, num_layers=4),
                EnhancedNeuralEqualizationCorrection(hidden_size=320, num_layers=4),
                EnhancedNeuralEqualizationCorrection(hidden_size=384, num_layers=5)
            ])
        else:
            self.equalization_correctors = nn.ModuleList([
                LearnedEqualizationCorrection() for _ in range(5)
            ])
            
        self.rate_specific_equalizers = nn.ModuleList([
            EnhancedNeuralEqualizationCorrection(hidden_size=64, num_layers=2) if bps <= 3 else
            EnhancedNeuralEqualizationCorrection(hidden_size=128, num_layers=3) if bps <= 5 else
            EnhancedNeuralEqualizationCorrection(hidden_size=256, num_layers=4)
            for bps in RATES
        ])
        
        # 🚨 OUTPUT NORM HYBRID STRATEGY (Match Inputs)
        for i in range(len(RATES)):
            bps = RATES[i]
            if bps <= 4:
                setattr(self, f'output_norm_{i}', RMSNorm(num_rx_features))
            else:
                setattr(self, f'output_norm_{i}', SimpleScaler(num_rx_features, init_scale=0.5))
        
    def _create_mlp_decoder(self,bps):
        dropout_rate = 0.05
        input_dim = RealisticConfig.NUM_RX_ANTENNAS*2  + CSI_VECTOR_SIZE + 1
        return nn.Sequential(
            nn.Linear(input_dim, 256), nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, bps)
        )
    def _create_transformer_decoder(self, embed_size, num_heads, num_blocks, bps):
            dropout_rate = 0.05
            input_dim = RealisticConfig.NUM_RX_ANTENNAS * 2
            
            # Fourier Mapping Size (half of embed_size because we cat sin+cos)
            fourier_dim = embed_size // 2
            
            # Scale Heuristic: 
            # For High Rates (6, 7, 8), we need finer resolution -> Higher Scale.
            # Rate 4 (16QAM) -> Scale 1.0
            # Rate 8 (256QAM) -> Scale 2.0 or 3.0
            # Let's set a robust default of 2.0 for high rates.
            fourier_scale = 2.0 if bps >= 6 else 1.0

            return nn.ModuleDict({
                # 🚨 UPGRADE: Fourier Feature Embedding
                'symbol_processor': nn.Sequential(
                    # 1. Expand 8 inputs -> 512 Fourier Features (Sin/Cos)
                    GaussianFourierProjection(input_dim, mapping_size=fourier_dim, scale=fourier_scale),
                    
                    # 2. Process with Linear to mix frequencies
                    nn.Linear(fourier_dim * 2, embed_size),
                    nn.PReLU(),
                    nn.Dropout(dropout_rate)
                ),
                
                # CSI and Noise processors remain the same...
                'csi_processor': nn.Sequential(
                    nn.Linear(CSI_VECTOR_SIZE, 512), nn.PReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, embed_size)
                ),
                'noise_processor': nn.Linear(1, embed_size),
                'transformer_blocks': nn.Sequential(*[ReceiverTransformerBlock(embed_size, num_heads) for _ in range(num_blocks)]),
                'decoder_head': nn.Sequential(
                    nn.Linear(embed_size * 3, 256), nn.PReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, bps)
                )
            })

    def apply_neural_equalization(self, equalized_symbol, csi_context, noise_power, rate_index, return_mag=False):
        # 1. Input power monitoring
        input_std = equalized_symbol.std().item()
        if input_std < 0.001:
            emergency_gain = 0.1 / (input_std + 1e-9)
            emergency_gain = min(emergency_gain, 1000.0)
            equalized_symbol = equalized_symbol * emergency_gain
        
        # 2. Physics Context
        channel_power = torch.norm(csi_context, dim=1)**2
        
        if torch.all(noise_power == 0):
            kT_dbm_hz = -174
            bandwidth_hz = RealisticConfig.BANDWIDTH_HZ
            noise_figure_db = RealisticConfig.NOISE_FIGURE_DB
            noise_power_dbm = kT_dbm_hz + 10 * np.log10(bandwidth_hz) + noise_figure_db
            noise_power_val = 10**((noise_power_dbm - 30) / 10.0)
            noise_power = torch.full_like(channel_power, noise_power_val)
        else:
            noise_power = torch.clamp(noise_power, min=1e-12)
        
        snr_linear = channel_power / (noise_power + 1e-9)
        snr_db = 10 * torch.log10(snr_linear + 1e-9)
        snr_normalized = torch.sigmoid((snr_db - 10.0) / 5.0)
        
        channel_power_db = 10 * torch.log10(channel_power + 1e-9)
        channel_power_normalized = torch.sigmoid((channel_power_db - 20.0) / 10.0)
        
        # 3. Expert Selection
        if rate_index < 3:
            if self.use_neural_equalizer:
                raw_output = self.rate_specific_equalizers[rate_index](
                    equalized_symbol, channel_power_normalized, snr_normalized
                )
            else:
                raw_output = equalized_symbol
        else:
            corrector_index = rate_index - 3
            if corrector_index < len(self.equalization_correctors):
                raw_output = self.equalization_correctors[corrector_index](
                    equalized_symbol, channel_power_normalized, snr_normalized
                )
            else:
                raw_output = equalized_symbol
        
        # 4. Output Normalization (Now SimpleScaler)
        output_norm = getattr(self, f'output_norm_{rate_index}')
        normalized_output = output_norm(raw_output)
        
        # 5. Safety Clamp (Loose bounds)
        final_output = torch.clamp(normalized_output, min=-30.0, max=30.0)

        # 🚨 NEW: Calculate Correction Magnitude
        if return_mag:
            # Measure: ||NetworkOutput - Input||
            # We use 'raw_output' vs 'equalized_symbol' because they are in the same scaling domain.
            diff = raw_output - equalized_symbol
            magnitude = torch.norm(diff, p=2, dim=1) # Returns [Batch_Size]
            return final_output, magnitude

        return final_output

    def forward(self, equalized_symbol, csi_context, noise_power, rate_one_hot):
        # Input stabilization
        input_std = equalized_symbol.std().item()
        if input_std < 0.01:
            emergency_gain = 0.1 / (input_std + 1e-9)
            emergency_gain = min(emergency_gain, 1000.0)
            equalized_symbol = equalized_symbol * emergency_gain
            
        if equalized_symbol.dtype == torch.complex64 or equalized_symbol.dtype == torch.complex128:
            equalized_symbol_real = torch.view_as_real(equalized_symbol)
            equalized_symbol = equalized_symbol_real.view(equalized_symbol_real.shape[0], -1)
        
        chosen_rate_index = torch.argmax(rate_one_hot, dim=1)
        max_bps = max(RATES)
        final_logits = torch.zeros((csi_context.shape[0], max_bps), 
                                 dtype=csi_context.dtype, device=csi_context.device)
        # 🚨 NEW: Tensor to store correction magnitudes for the batch
        correction_magnitudes = torch.zeros(csi_context.shape[0], device=csi_context.device)
        for i in range(len(self.decoders)):
            mask = (chosen_rate_index == i)
            if not mask.any(): continue
    
            symbol_slice = equalized_symbol[mask]
            csi_slice = csi_context[mask]
            noise_slice = noise_power[mask]
            
            stabilized_symbol_slice = self.input_stabilizers[i](symbol_slice)
            # 🚨 NEW: Call with return_mag=True
            enhanced_symbol_slice, mag_slice = self.apply_neural_equalization(
                stabilized_symbol_slice, csi_slice, noise_slice, i, return_mag=True
            )            
            # Store magnitudes in the correct positions
            correction_magnitudes[mask] = mag_slice
            if i < 3:
                expert = self.decoders[i]
                combined_input = torch.cat([enhanced_symbol_slice, csi_slice, noise_slice.unsqueeze(1)], dim=1)
                output_logits = expert(combined_input)
            else:
                expert = self.decoders[i]
                symbol_emb = expert['symbol_processor'](enhanced_symbol_slice)
                csi_emb = expert['csi_processor'](csi_slice)
                noise_emb = expert['noise_processor'](noise_slice.unsqueeze(1))
                
                x_seq = torch.stack([symbol_emb, csi_emb, noise_emb], dim=1)
                x_tf = expert['transformer_blocks'](x_seq)
                x_tf = x_tf.view(x_tf.shape[0], -1)
                output_logits = expert['decoder_head'](x_tf)

            current_bps = RATES[i]
            final_logits[mask, :current_bps] = output_logits.to(final_logits.dtype)
        
        return final_logits, correction_magnitudes

class ManagerAI(nn.Module):
    def __init__(self, embed_size=128, num_heads=4):
        super(ManagerAI, self).__init__()
        self.rate_head_csi_embedding = nn.Linear(CSI_VECTOR_SIZE, embed_size)
        self.rate_head_noise_embedding = nn.Linear(1, embed_size)
        self.rate_head_transformer = ReceiverTransformerBlock(embed_size, num_heads)
        
        self.rate_head_decoder = nn.Sequential(
            nn.Linear(embed_size * 2, 64),
            nn.LayerNorm(64),
            nn.PReLU(),
            nn.Linear(64, len(RATES))
        )

    def forward(self, csi, snr_feature):
        csi_emb = self.rate_head_csi_embedding(csi)
        snr_emb = self.rate_head_noise_embedding(snr_feature)
        x = torch.stack([csi_emb, snr_emb], dim=1)
        x = self.rate_head_transformer(x)
        x = x.view(x.shape[0], -1)
        return self.rate_head_decoder(x)

class Critic(nn.Module):
    def __init__(self, embed_size=128, num_heads=4):
        super(Critic, self).__init__()
        self.csi_embedding = nn.Linear(CSI_VECTOR_SIZE, embed_size)
        self.snr_embedding = nn.Linear(1, embed_size)
        self.transformer = ReceiverTransformerBlock(embed_size, num_heads)
        
        self.value_head = nn.Sequential(
            nn.Linear(embed_size * 2, 64), nn.PReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, csi, snr_feature):
        csi_emb = self.csi_embedding(csi)
        snr_emb = self.snr_embedding(snr_feature)
        x = torch.stack([csi_emb, snr_emb], dim=1)
        x = self.transformer(x)
        x = x.view(x.shape[0], -1)
        return self.value_head(x)
        
class EmergencyOutputConstraint(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_limits = {
            1: 1.1, 2: 1.2, 3: 1.3, 4: 1.4, 
            5: 1.5, 6: 1.6, 7: 1.7, 8: 1.8 
        }
    
    def forward(self, x, rate_one_hot):
        rate_indices = torch.argmax(rate_one_hot, dim=1)
        constrained_output = x.clone()
        for i, rate_idx in enumerate(rate_indices):
            bps = RATES[rate_idx]
            max_limit = self.max_limits.get(bps, 1.5)
            current_norm = torch.norm(x[i])
            if current_norm > max_limit:
                scale_factor = max_limit / current_norm
                constrained_output[i] = x[i] * scale_factor
        return constrained_output 
