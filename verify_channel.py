import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from channel import RealisticMIMOChannel
from config import RealisticConfig
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = True    
class MIMOChannelDiagnostic:

    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
    
    def run_distance_sweep(self, verbose=False):
        """
        Distance：from 10m to 2000m
        - 10-100m：every 10m,10 samples
        - 100-2000m：every 50m,10 samples
        """
        print("🚀 start distance scanning...")
        

        distances_short = range(10, 101, 10)  
        distances_long = range(100, 2001, 50)  
        all_distances = list(distances_short) + list(distances_long)
        
        results = {
            'distance': [],
            'snr_mean': [],
            'snr_std': [],
            'path_loss_db': [],
            'rx_power_dbm': [],
            'tx_power_scale': []
        }
        
        for distance in all_distances:
            if verbose:
                print(f"\n📏 test distance: {distance}m")
            
            snr_samples = []
            path_loss_samples = []
            rx_power_samples = []
            tx_scale_samples = []
            

            for sample_idx in range(10):
）
                channel = RealisticMIMOChannel(
                    batch_size=1,
                    device=self.device,
                    use_fading=True,
                    distance_meters=distance,
                    fixed_snr_db=None,  
                    verbose=False
                )
                

                num_tx_antennas = RealisticConfig.NUM_TX_ANTENNAS
                tx_signal = torch.randn(1, num_tx_antennas, dtype=torch.cfloat, device=self.device)
                

                tx_power = torch.mean(torch.abs(tx_signal)**2).item()
                if tx_power > 0:
                    tx_signal = tx_signal / np.sqrt(tx_power)


                assert tx_signal.shape[1] == num_tx_antennas, f"antenna error: {tx_signal.shape[1]} != {num_tx_antennas}"
         

                rx_signal = channel.apply(tx_signal, add_noise=True)
                

                snr_samples.append(channel.actual_snr_db)
                path_loss_samples.append(channel.effective_path_loss_db)
                rx_power_samples.append(10*np.log10(channel.expected_rx_power_watts)+30)
                tx_scale_samples.append(channel.tx_power_scale)
            

            results['distance'].append(distance)
            results['snr_mean'].append(np.mean(snr_samples))
            results['snr_std'].append(np.std(snr_samples))
            results['path_loss_db'].append(np.mean(path_loss_samples))
            results['rx_power_dbm'].append(np.mean(rx_power_samples))
            results['tx_power_scale'].append(np.mean(tx_scale_samples))
            
            if verbose:
                print(f"   SNR: {np.mean(snr_samples):.1f} ± {np.std(snr_samples):.1f} dB")
                print(f"   path loss: {np.mean(path_loss_samples):.1f} dB")
                print(f"   rx power: {np.mean(rx_power_samples):.1f} dBm")
        
        self.results['distance_sweep'] = results
        return results
    
    def run_fixed_snr_sweep(self, snr_values=None, distance=500, verbose=False):

        print("🚀 FIX SNR scanning...")
        
        if snr_values is None:
            snr_values = [-10, -5, 0, 5, 10, 15, 20, 25, 30]  
        
        results = {
            'target_snr': [],
            'actual_snr_mean': [],
            'actual_snr_std': [],
            'snr_error_mean': [],
            'snr_error_std': [],
            'tx_power_scale_mean': [],
            'tx_power_scale_std': []
        }
        
        for target_snr in snr_values:
            if verbose:
                print(f"\n🎯 target SNR: {target_snr} dB")
            
            actual_snr_samples = []
            snr_error_samples = []
            tx_scale_samples = []
            

            for sample_idx in range(10):

                channel = RealisticMIMOChannel(
                    batch_size=1,
                    device=self.device,
                    use_fading=True,
                    distance_meters=distance,
                    fixed_snr_db=target_snr,  
                    verbose=False
                )
                

                num_tx_antennas = RealisticConfig.NUM_TX_ANTENNAS
                tx_signal = torch.randn(1, num_tx_antennas, dtype=torch.cfloat, device=self.device)
                

                tx_power = torch.mean(torch.abs(tx_signal)**2).item()
                if tx_power > 0:
                    tx_signal = tx_signal / np.sqrt(tx_power)
                

                assert tx_signal.shape[1] == num_tx_antennas, f"信号天线数错误: {tx_signal.shape[1]} != {num_tx_antennas}"
                

                rx_signal = channel.apply(tx_signal, add_noise=True)
                

                actual_snr_samples.append(channel.actual_snr_db)
                snr_error_samples.append(channel.actual_snr_db - target_snr)
                tx_scale_samples.append(channel.tx_power_scale)
            

            results['target_snr'].append(target_snr)
            results['actual_snr_mean'].append(np.mean(actual_snr_samples))
            results['actual_snr_std'].append(np.std(actual_snr_samples))
            results['snr_error_mean'].append(np.mean(snr_error_samples))
            results['snr_error_std'].append(np.std(snr_error_samples))
            results['tx_power_scale_mean'].append(np.mean(tx_scale_samples))
            results['tx_power_scale_std'].append(np.std(tx_scale_samples))
            
            if verbose:
                print(f"   real SNR: {np.mean(actual_snr_samples):.1f} ± {np.std(actual_snr_samples):.1f} dB")
                print(f"   SNR error: {np.mean(snr_error_samples):.1f} ± {np.std(snr_error_samples):.1f} dB")
                print(f"   tx power scaler: {np.mean(tx_scale_samples):.6f}")
        
        self.results['fixed_snr_sweep'] = results
        return results
    
    def run_comprehensive_test(self, verbose=True):

        print("🔍 system test...")
        


        try:
            channel = RealisticMIMOChannel(
                batch_size=2, 
                device=self.device,
                distance_meters=100,
                verbose=False
            )
            

            print(f"   config:")
            print(f"     - tx antenna: {RealisticConfig.NUM_TX_ANTENNAS}")
            print(f"     - rx antenna: {RealisticConfig.NUM_RX_ANTENNAS}")
            print(f"     - batchsize: 2")
            

            print(f"   2Dtest signal...")
            tx_2d = torch.randn(2, RealisticConfig.NUM_TX_ANTENNAS, dtype=torch.cfloat, device=self.device)
            print(f"     2D shape: {tx_2d.shape}")
            
            print(f"   apply ..")
            rx_2d = channel.apply(tx_2d, add_noise=True)
            print(f"     2D output shape: {rx_2d.shape}")
            
            expected_2d_shape = (2, RealisticConfig.NUM_RX_ANTENNAS)
            assert rx_2d.shape == expected_2d_shape, f"2D output error: {rx_2d.shape} != {expected_2d_shape}"
            print("   ✅ 2D MIMO successful")
            

            print(f"   generate 3D test signal...")
            tx_3d = torch.randn(2, RealisticConfig.NUM_TX_ANTENNAS, 2, dtype=torch.cfloat, device=self.device)
            print(f"     3D signal shape: {tx_3d.shape}")
            
            print(f"   apply channel to 3D...")
            rx_3d = channel.apply(tx_3d, add_noise=True)
            print(f"     3D output shape: {rx_3d.shape}")
            
            expected_3d_shape = (2, RealisticConfig.NUM_RX_ANTENNAS, 2)
            assert rx_3d.shape == expected_3d_shape, f"3D output error: {rx_3d.shape} != {expected_3d_shape}"
            print("   ✅ 3DMIMO successful")
            
        except Exception as e:
            print(f"   ❌ dimention error: {e}")

            print(f"   debug:")
            print(f"     - channel shape: {channel.H_fading.shape if 'channel' in locals() else 'N/A'}")
            return False
        

        print("\n2. test SNR control mode...")
        try:
            target_snr = 20.0
            channel = RealisticMIMOChannel(
                batch_size=1,
                device=self.device,
                distance_meters=500,
                fixed_snr_db=target_snr,
                verbose=False
            )
            
            tx_signal = torch.randn(1, RealisticConfig.NUM_TX_ANTENNAS, dtype=torch.cfloat, device=self.device)
            tx_power = torch.mean(torch.abs(tx_signal)**2).item()
            if tx_power > 0:
                tx_signal = tx_signal / np.sqrt(tx_power)
            
            rx_signal = channel.apply(tx_signal, add_noise=True)

            snr_error = abs(channel.actual_snr_db - target_snr)
            if snr_error < 5.0:  
                print(f"   ✅ SNR test pass (deviation: {snr_error:.1f} dB)")
            else:
                print(f"   ⚠️ SNR test no pass: {snr_error:.1f} dB")
                
        except Exception as e:
            print(f"   ❌ SNR test failed: {e}")
            return False
        

        print("\n3. test physics mode...")
        try:
            channel = RealisticMIMOChannel(
                batch_size=1,
                device=self.device,
                distance_meters=100,
                fixed_snr_db=None,  
                verbose=False
            )
            
            tx_signal = torch.randn(1, RealisticConfig.NUM_TX_ANTENNAS, dtype=torch.cfloat, device=self.device)
            tx_power = torch.mean(torch.abs(tx_signal)**2).item()
            if tx_power > 0:
                tx_signal = tx_signal / np.sqrt(tx_power)
            
            rx_signal = channel.apply(tx_signal, add_noise=True)
            
            print(f"   ✅ physics mode pass")
            print(f"      path loss: {channel.effective_path_loss_db:.1f} dB")
            print(f"      real SNR: {channel.actual_snr_db:.1f} dB")
            
        except Exception as e:
            print(f"   ❌ physics mode failed: {e}")
            return False
        
        print("\n🎉 all pass！")
        return True

    def plot_distance_analysis(self):

        if 'distance_sweep' not in self.results:
            print("❌ Please run distance sweep first")
            return
        
        data = self.results['distance_sweep']
        distances = data['distance']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # SNR vs Distance
        ax1.errorbar(distances, data['snr_mean'], yerr=data['snr_std'], 
                    fmt='o-', capsize=5, color='blue', alpha=0.7)
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('SNR (dB)')
        ax1.set_title('SNR vs Distance')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Path Loss vs Distance
        ax2.plot(distances, data['path_loss_db'], 'ro-', alpha=0.7)
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Path Loss (dB)')
        ax2.set_title('Path Loss vs Distance')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Received Power vs Distance
        ax3.plot(distances, data['rx_power_dbm'], 'go-', alpha=0.7)
        ax3.set_xlabel('Distance (m)')
        ax3.set_ylabel('Received Power (dBm)')
        ax3.set_title('Received Power vs Distance')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Tx Power Scaling vs Distance
        ax4.plot(distances, data['tx_power_scale'], 'mo-', alpha=0.7)
        ax4.set_xlabel('Distance (m)')
        ax4.set_ylabel('Tx Power Scale')
        ax4.set_title('Tx Power Scaling vs Distance')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('distance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_snr_analysis(self):

        if 'fixed_snr_sweep' not in self.results:
            print("❌ Please run fixed SNR sweep first")
            return
        
        data = self.results['fixed_snr_sweep']
        target_snrs = data['target_snr']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Actual SNR vs Target SNR
        ax1.errorbar(target_snrs, data['actual_snr_mean'], yerr=data['actual_snr_std'],
                    fmt='o-', capsize=5, color='blue', alpha=0.7, label='Actual SNR')
        ax1.plot(target_snrs, target_snrs, 'k--', alpha=0.5, label='Ideal')
        ax1.set_xlabel('Target SNR (dB)')
        ax1.set_ylabel('Actual SNR (dB)')
        ax1.set_title('Actual SNR vs Target SNR')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # SNR Error
        ax2.errorbar(target_snrs, data['snr_error_mean'], yerr=data['snr_error_std'],
                    fmt='o-', capsize=5, color='red', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Target SNR (dB)')
        ax2.set_ylabel('SNR Error (dB)')
        ax2.set_title('SNR Error vs Target SNR')
        ax2.grid(True, alpha=0.3)
        
        # Tx Power Scaling
        ax3.errorbar(target_snrs, data['tx_power_scale_mean'], yerr=data['tx_power_scale_std'],
                    fmt='o-', capsize=5, color='green', alpha=0.7)
        ax3.set_xlabel('Target SNR (dB)')
        ax3.set_ylabel('Tx Power Scale')
        ax3.set_title('Tx Power Scaling vs Target SNR')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # SNR Error Distribution
        ax4.hist(np.concatenate([np.array(data['snr_error_mean'])]), bins=20, alpha=0.7, color='orange')
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('SNR Error (dB)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('SNR Error Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('snr_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    def generate_report(self):

        print("\n" + "="*60)
        print("📊 MIMO Channel Model Diagnostic Report")
        print("="*60)
        
        if 'distance_sweep' in self.results:
            print("\n📏 Distance Sweep Results:")
            dist_data = self.results['distance_sweep']
            print(f"   Test Distance Range: {min(dist_data['distance'])} - {max(dist_data['distance'])} m")
            print(f"   SNR Range: {min(dist_data['snr_mean']):.1f} - {max(dist_data['snr_mean']):.1f} dB")
            print(f"   Path Loss Range: {min(dist_data['path_loss_db']):.1f} - {max(dist_data['path_loss_db']):.1f} dB")
        
        if 'fixed_snr_sweep' in self.results:
            print("\n🎯 Fixed SNR Sweep Results:")
            snr_data = self.results['fixed_snr_sweep']
            avg_error = np.mean(np.abs(snr_data['snr_error_mean']))
            max_error = np.max(np.abs(snr_data['snr_error_mean']))
            print(f"   SNR Test Range: {min(snr_data['target_snr'])} - {max(snr_data['target_snr'])} dB")
            print(f"   Average SNR Error: {avg_error:.2f} dB")
            print(f"   Maximum SNR Error: {max_error:.2f} dB")
            

            if max_error < 1.0:
                print("   ✅ SNR Control: Excellent")
            elif max_error < 3.0:
                print("   ✅ SNR Control: Good")
            elif max_error < 5.0:
                print("   ⚠️ SNR Control: Fair")
            else:
                print("   ❌ SNR Control: Needs Improvement")
        
        print("\n" + "="*60)

if __name__ == "__main__":

    diagnostic = MIMOChannelDiagnostic(device='cuda' if torch.cuda.is_available() else 'cpu')
    

    print("="*60)
    print("🔍 MIMO channel test")
    print("="*60)
    

    basic_test_passed = diagnostic.run_comprehensive_test(verbose=True)
    
    if basic_test_passed:

        print("\n" + "="*60)
        print("🚀 start scanning...")
        distance_results = diagnostic.run_distance_sweep(verbose=True)
        

        print("\n" + "="*60)
        print("🚀 start fix SNR scanning...")
        snr_results = diagnostic.run_fixed_snr_sweep(verbose=True)
        

        diagnostic.plot_distance_analysis()
        diagnostic.plot_snr_analysis()
        

        diagnostic.generate_report()
        

        if 'distance_sweep' in diagnostic.results:
            pd.DataFrame(diagnostic.results['distance_sweep']).to_csv('distance_sweep_results.csv', index=False)
            print("✅ distance save to  distance_sweep_results.csv")
        
        if 'fixed_snr_sweep' in diagnostic.results:
            pd.DataFrame(diagnostic.results['fixed_snr_sweep']).to_csv('snr_sweep_results.csv', index=False)
            print("✅ SNR result save to snr_sweep_results.csv")
        
        print("\n🎉 pass！")
    else:
        print("\n❌ no pass")