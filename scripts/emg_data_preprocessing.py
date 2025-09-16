#!/usr/bin/env python3
"""
EMG Data Preprocessing Pipeline
Comprehensive preprocessing for all three architectures
Tuned for 8-channel EMG data with 6 gesture classes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import stft, welch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EMGDataPreprocessor:
    """Comprehensive EMG data preprocessor for all architectures"""
    
    def __init__(self, data_path, fs=1000, n_channels=8, gesture_classes=[1, 2, 3, 4, 5, 6]):
        self.data_path = Path(data_path)
        self.fs = fs
        self.n_channels = n_channels
        self.gesture_classes = gesture_classes
        self.scaler = StandardScaler()
        
    def load_subject_data(self, subject_id):
        """Load data for a specific subject"""
        print(f"📊 Loading subject {subject_id} data...")
        
        subject_folder = self.data_path / f"{subject_id:02d}"
        files = list(subject_folder.glob("*.txt"))
        
        if not files:
            raise FileNotFoundError(f"No data files found for subject {subject_id} in {subject_folder}")
        
        print(f"   📁 Found {len(files)} files: {[f.name for f in files]}")
        
        all_data = []
        for file in files:
            try:
                print(f"   📄 Loading {file.name}...")
                df = pd.read_csv(file, sep='\t')
                print(f"      Shape: {df.shape}, Columns: {list(df.columns)}")
                
                if df.empty:
                    print(f"      ⚠️  Warning: {file.name} is empty, skipping...")
                    continue
                    
                df['file'] = file.name
                all_data.append(df)
            except Exception as e:
                print(f"      ❌ Error loading {file.name}: {e}")
                continue
        
        if not all_data:
            raise ValueError(f"No valid data loaded for subject {subject_id}")
        
        print(f"   ✅ Successfully loaded {len(all_data)} files")
        data = pd.concat(all_data, ignore_index=True)
        
        # Extract EMG channels and labels
        emg_data = data.iloc[:, 1:9].values  # channels 1-8
        labels = data['class'].values
        time = data['time'].values
        
        # Filter out unmarked samples (class 0)
        valid_mask = labels != 0
        emg_data = emg_data[valid_mask]
        labels = labels[valid_mask]
        time = time[valid_mask]
        
        print(f"   ✅ Loaded {len(emg_data)} samples")
        print(f"   ✅ Gesture classes: {np.unique(labels)}")
        print(f"   ✅ Time range: {time[0]} - {time[-1]} ms")
        
        return emg_data, labels, time
    
    def resample_data(self, emg_data, time, target_fs=1000):
        """Resample data to regular sampling rate"""
        print(f"🔄 Resampling to {target_fs} Hz...")
        
        # Create regular time grid
        start_time = time[0]
        end_time = time[-1]
        regular_time = np.arange(start_time, end_time, 1000/target_fs)  # 1ms intervals
        
        # Interpolate EMG data
        emg_resampled = np.zeros((len(regular_time), self.n_channels))
        for ch in range(self.n_channels):
            emg_resampled[:, ch] = np.interp(regular_time, time, emg_data[:, ch])
        
        # Interpolate labels (nearest neighbor)
        labels_resampled = np.interp(regular_time, time, labels)
        labels_resampled = np.round(labels_resampled).astype(int)
        
        print(f"   ✅ Resampled to {len(regular_time)} samples")
        
        return emg_resampled, labels_resampled, regular_time
    
    def apply_filters(self, emg_data, lowcut=20, highcut=450):
        """Apply bandpass filter to EMG data"""
        print(f"🔧 Applying bandpass filter ({lowcut}-{highcut} Hz)...")
        
        # Design bandpass filter
        nyquist = self.fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter to each channel
        emg_filtered = np.zeros_like(emg_data)
        for ch in range(self.n_channels):
            emg_filtered[:, ch] = signal.filtfilt(b, a, emg_data[:, ch])
        
        print(f"   ✅ Filter applied successfully")
        
        return emg_filtered
    
    def remove_artifacts(self, emg_data, threshold=3):
        """Remove artifacts using statistical thresholding"""
        print(f"🧹 Removing artifacts (threshold: {threshold}σ)...")
        
        # Calculate threshold for each channel
        channel_std = np.std(emg_data, axis=0)
        channel_mean = np.mean(emg_data, axis=0)
        
        # Identify artifacts
        artifacts = np.any(np.abs(emg_data - channel_mean) > threshold * channel_std, axis=1)
        
        # Replace artifacts with interpolated values
        emg_clean = emg_data.copy()
        for ch in range(self.n_channels):
            if np.any(artifacts):
                valid_indices = np.where(~artifacts)[0]
                if len(valid_indices) > 1:
                    emg_clean[artifacts, ch] = np.interp(
                        np.where(artifacts)[0], 
                        valid_indices, 
                        emg_clean[valid_indices, ch]
                    )
        
        artifact_ratio = np.sum(artifacts) / len(artifacts)
        print(f"   ✅ Removed {artifact_ratio:.3f} artifacts")
        
        return emg_clean
    
    def normalize_data(self, emg_data, method='zscore'):
        """Normalize EMG data"""
        print(f"📏 Normalizing data using {method}...")
        
        if method == 'zscore':
            # Z-score normalization per channel
            emg_normalized = np.zeros_like(emg_data)
            for ch in range(self.n_channels):
                channel_data = emg_data[:, ch]
                emg_normalized[:, ch] = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)
        
        elif method == 'minmax':
            # Min-max normalization per channel
            emg_normalized = np.zeros_like(emg_data)
            for ch in range(self.n_channels):
                channel_data = emg_data[:, ch]
                emg_normalized[:, ch] = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data) + 1e-8)
        
        elif method == 'global':
            # Global normalization
            emg_normalized = (emg_data - np.mean(emg_data)) / (np.std(emg_data) + 1e-8)
        
        print(f"   ✅ Normalization complete")
        
        return emg_normalized
    
    def create_windows(self, emg_data, labels, window_size=1000, overlap=0.5):
        """Create overlapping windows from EMG data"""
        print(f"🪟 Creating windows (size: {window_size}, overlap: {overlap})...")
        
        hop_length = int(window_size * (1 - overlap))
        windows = []
        window_labels = []
        
        for i in range(0, len(emg_data) - window_size + 1, hop_length):
            window = emg_data[i:i + window_size]
            window_label = labels[i:i + window_size]
            
            # Only use windows with consistent labels (no transitions)
            if len(np.unique(window_label)) == 1 and window_label[0] in self.gesture_classes:
                windows.append(window)
                window_labels.append(window_label[0])
        
        windows = np.array(windows)
        window_labels = np.array(window_labels)
        
        print(f"   ✅ Created {len(windows)} windows")
        print(f"   ✅ Window distribution: {np.bincount(window_labels - 1)}")
        
        return windows, window_labels
    
    def create_stft_representation(self, windows, nperseg=64, noverlap=32):
        """Create 2D STFT representation for Vision Transformer"""
        print(f"📊 Creating STFT representation (nperseg: {nperseg}, noverlap: {noverlap})...")
        
        stft_windows = []
        
        for window in windows:
            stft_2d = []
            for ch in range(self.n_channels):
                f, t, stft_ch = stft(window[:, ch], fs=self.fs, nperseg=nperseg, noverlap=noverlap)
                stft_2d.append(np.abs(stft_ch))
            
            # Stack channels: (n_channels, freq_bins, time_bins)
            stft_2d = np.stack(stft_2d, axis=0)
            
            # Normalize
            stft_2d = (stft_2d - np.mean(stft_2d)) / (np.std(stft_2d) + 1e-8)
            
            stft_windows.append(stft_2d)
        
        stft_windows = np.array(stft_windows)
        
        print(f"   ✅ STFT shape: {stft_windows.shape}")
        
        return stft_windows
    
    def create_wavelet_representation(self, windows, wavelet='db4', levels=4):
        """Create wavelet representation"""
        print(f"🌊 Creating wavelet representation (wavelet: {wavelet}, levels: {levels})...")
        
        import pywt
        
        wavelet_windows = []
        
        for window in windows:
            wavelet_2d = []
            for ch in range(self.n_channels):
                coeffs = pywt.wavedec(window[:, ch], wavelet, level=levels)
                # Reconstruct approximation and details
                reconstructed = pywt.waverec(coeffs, wavelet)
                # Pad or truncate to match original length
                if len(reconstructed) > len(window[:, ch]):
                    reconstructed = reconstructed[:len(window[:, ch])]
                elif len(reconstructed) < len(window[:, ch]):
                    reconstructed = np.pad(reconstructed, (0, len(window[:, ch]) - len(reconstructed)))
                
                wavelet_2d.append(reconstructed)
            
            # Stack channels
            wavelet_2d = np.stack(wavelet_2d, axis=0)
            
            # Normalize
            wavelet_2d = (wavelet_2d - np.mean(wavelet_2d)) / (np.std(wavelet_2d) + 1e-8)
            
            wavelet_windows.append(wavelet_2d)
        
        wavelet_windows = np.array(wavelet_windows)
        
        print(f"   ✅ Wavelet shape: {wavelet_windows.shape}")
        
        return wavelet_windows
    
    def preprocess_for_architecture(self, subject_id, architecture='all'):
        """Preprocess data for specific architecture"""
        print(f"🔧 Preprocessing for {architecture} architecture...")
        
        # Load raw data
        emg_data, labels, time = self.load_subject_data(subject_id)
        
        # Resample to regular sampling rate
        emg_data, labels, time = self.resample_data(emg_data, time, self.fs)
        
        # Apply filters
        emg_data = self.apply_filters(emg_data)
        
        # Remove artifacts
        emg_data = self.remove_artifacts(emg_data)
        
        # Normalize data
        emg_data = self.normalize_data(emg_data, method='zscore')
        
        # Create windows
        windows, window_labels = self.create_windows(emg_data, labels)
        
        results = {
            'raw_data': emg_data,
            'labels': labels,
            'time': time,
            'windows': windows,
            'window_labels': window_labels
        }
        
        if architecture in ['vit', 'all']:
            # Create STFT representation for Vision Transformer
            stft_windows = self.create_stft_representation(windows)
            results['stft_windows'] = stft_windows
        
        if architecture in ['wavelet', 'all']:
            # Create wavelet representation
            wavelet_windows = self.create_wavelet_representation(windows)
            results['wavelet_windows'] = wavelet_windows
        
        print(f"   ✅ Preprocessing complete for {architecture}")
        
        return results
    
    def create_visualizations(self, results, subject_id=1):
        """Create preprocessing visualizations"""
        print("📊 Creating preprocessing visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # 1. Raw EMG signals
        ax1 = axes[0, 0]
        time_sec = results['time'] / 1000
        for i in range(min(3, self.n_channels)):
            ax1.plot(time_sec, results['raw_data'][:, i] + i * 0.0001, label=f'Channel {i+1}', alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (offset)')
        ax1.set_title(f'Raw EMG Signals - Subject {subject_id}')
        ax1.legend()
        
        # 2. Gesture labels
        ax2 = axes[0, 1]
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.gesture_classes)))
        for i, gesture in enumerate(self.gesture_classes):
            mask = results['labels'] == gesture
            if np.any(mask):
                ax2.scatter(time_sec[mask], results['labels'][mask], c=[colors[i]], 
                           label=f'Gesture {gesture}', alpha=0.6, s=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Gesture Class')
        ax2.set_title('Gesture Labels Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Window distribution
        ax3 = axes[0, 2]
        window_counts = np.bincount(results['window_labels'] - 1)
        ax3.bar(range(len(window_counts)), window_counts, color=colors[:len(window_counts)])
        ax3.set_xlabel('Gesture Class')
        ax3.set_ylabel('Number of Windows')
        ax3.set_title('Window Distribution')
        ax3.set_xticks(range(len(window_counts)))
        ax3.set_xticklabels([f'G{i+1}' for i in range(len(window_counts))])
        
        # 4. Signal amplitude distribution
        ax4 = axes[1, 0]
        ax4.hist(np.abs(results['raw_data']).flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Absolute Amplitude')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Signal Amplitude Distribution')
        ax4.set_yscale('log')
        
        # 5. Channel correlation
        ax5 = axes[1, 1]
        channel_corr = np.corrcoef(results['raw_data'].T)
        sns.heatmap(channel_corr, annot=True, cmap='coolwarm', center=0,
                   xticklabels=[f'Ch{i+1}' for i in range(self.n_channels)],
                   yticklabels=[f'Ch{i+1}' for i in range(self.n_channels)],
                   ax=ax5)
        ax5.set_title('Channel Correlation Matrix')
        
        # 6. Frequency spectrum
        ax6 = axes[1, 2]
        f, psd = welch(results['raw_data'][:, 0], fs=self.fs, nperseg=min(1024, len(results['raw_data'])//4))
        ax6.semilogy(f, psd)
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Power Spectral Density')
        ax6.set_title('Frequency Spectrum - Channel 1')
        ax6.set_xlim(0, 500)
        
        # 7. STFT representation (if available)
        if 'stft_windows' in results:
            ax7 = axes[2, 0]
            stft_sample = results['stft_windows'][0, 0, :, :]  # First window, first channel
            im = ax7.imshow(stft_sample, aspect='auto', origin='lower', cmap='viridis')
            ax7.set_xlabel('Time')
            ax7.set_ylabel('Frequency')
            ax7.set_title('STFT Representation (Ch1)')
            plt.colorbar(im, ax=ax7)
        
        # 8. Wavelet representation (if available)
        if 'wavelet_windows' in results:
            ax8 = axes[2, 1]
            wavelet_sample = results['wavelet_windows'][0, 0, :]  # First window, first channel
            ax8.plot(wavelet_sample)
            ax8.set_xlabel('Time')
            ax8.set_ylabel('Amplitude')
            ax8.set_title('Wavelet Representation (Ch1)')
            ax8.grid(True, alpha=0.3)
        
        # 9. Preprocessing summary
        ax9 = axes[2, 2]
        summary_text = f"""
        Preprocessing Summary:
        • Raw samples: {len(results['raw_data']):,}
        • Windows: {len(results['windows']):,}
        • Channels: {self.n_channels}
        • Gesture classes: {len(self.gesture_classes)}
        • Sampling rate: {self.fs} Hz
        • Window size: {results['windows'].shape[1]} samples
        """
        ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax9.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'emg_preprocessing_subject_{subject_id}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Visualizations saved as 'emg_preprocessing_subject_{subject_id}.png'")

def main():
    """Demo preprocessing pipeline"""
    print("🚀 EMG Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Initialize preprocessor - data is in parent directory
    data_path = "../EMG_data"
    preprocessor = EMGDataPreprocessor(data_path)
    
    # Preprocess for all architectures
    results = preprocessor.preprocess_for_architecture(subject_id=1, architecture='all')
    
    # Create visualizations
    preprocessor.create_visualizations(results, subject_id=1)
    
    print("\n🎉 Preprocessing complete!")
    print("✅ Data ready for all three architectures:")
    print("   • Vision Transformer (STFT representation)")
    print("   • Attention CNN-LSTM (windowed data)")
    print("   • Domain Adaptive Transformer (multi-subject)")

if __name__ == "__main__":
    main()


