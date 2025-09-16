#!/usr/bin/env python3
"""
Modern EMG Feature Engineering
Based on 2024-2025 research for comprehensive feature extraction
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
import warnings
warnings.filterwarnings('ignore')

class ModernEMGFeatureExtractor:
    def __init__(self, fs=1000):
        self.fs = fs
        self.feature_names = []
        
    def extract_all_features(self, emg_data, labels=None):
        """Extract comprehensive feature set based on latest research"""
        print("🔧 Extracting Modern EMG Features...")
        
        features = {}
        
        # Time Domain Features (8 features per channel)
        print("   📊 Time Domain Features...")
        features.update(self._extract_time_domain_features(emg_data))
        
        # Frequency Domain Features (6 features per channel)
        print("   📊 Frequency Domain Features...")
        features.update(self._extract_frequency_domain_features(emg_data))
        
        # Time-Frequency Features (4 features per channel)
        print("   📊 Time-Frequency Features...")
        features.update(self._extract_time_frequency_features(emg_data))
        
        # Advanced Features (2+ features per channel)
        print("   📊 Advanced Features...")
        features.update(self._extract_advanced_features(emg_data))
        
        # Cross-Channel Features
        print("   📊 Cross-Channel Features...")
        features.update(self._extract_cross_channel_features(emg_data))
        
        # Convert to feature matrix
        feature_matrix = self._create_feature_matrix(features)
        
        print(f"   ✅ Extracted {feature_matrix.shape[1]} features from {emg_data.shape[1]} channels")
        
        return feature_matrix, self.feature_names
    
    def _extract_time_domain_features(self, emg_data):
        """Extract 8 time domain features per channel"""
        features = {}
        n_channels = emg_data.shape[1]
        
        for ch in range(n_channels):
            signal = emg_data[:, ch]
            
            # 1. Mean Absolute Value (MAV)
            features[f'MAV_ch{ch+1}'] = np.mean(np.abs(signal))
            
            # 2. Root Mean Square (RMS)
            features[f'RMS_ch{ch+1}'] = np.sqrt(np.mean(signal**2))
            
            # 3. Variance
            features[f'VAR_ch{ch+1}'] = np.var(signal)
            
            # 4. Standard Deviation
            features[f'STD_ch{ch+1}'] = np.std(signal)
            
            # 5. Wavelength (WL)
            features[f'WL_ch{ch+1}'] = np.sum(np.abs(np.diff(signal)))
            
            # 6. Zero Crossing Rate (ZC)
            features[f'ZC_ch{ch+1}'] = self._zero_crossing_rate(signal)
            
            # 7. Slope Sign Changes (SSC)
            features[f'SSC_ch{ch+1}'] = self._slope_sign_changes(signal)
            
            # 8. Integrated EMG (IEMG)
            features[f'IEMG_ch{ch+1}'] = np.sum(np.abs(signal))
        
        return features
    
    def _extract_frequency_domain_features(self, emg_data):
        """Extract 6 frequency domain features per channel"""
        features = {}
        n_channels = emg_data.shape[1]
        
        for ch in range(n_channels):
            signal = emg_data[:, ch]
            
            # Calculate power spectral density
            f, psd = signal.welch(signal, fs=self.fs, nperseg=min(256, len(signal)//4))
            
            # 1. Mean Power Spectral Density
            features[f'PSD_MEAN_ch{ch+1}'] = np.mean(psd)
            
            # 2. Peak Power Spectral Density
            features[f'PSD_PEAK_ch{ch+1}'] = np.max(psd)
            
            # 3. Spectral Centroid
            features[f'SPECTRAL_CENTROID_ch{ch+1}'] = np.sum(f * psd) / np.sum(psd)
            
            # 4. Spectral Bandwidth
            centroid = features[f'SPECTRAL_CENTROID_ch{ch+1}']
            features[f'SPECTRAL_BANDWIDTH_ch{ch+1}'] = np.sqrt(
                np.sum(((f - centroid)**2) * psd) / np.sum(psd)
            )
            
            # 5. Spectral Rolloff (85% energy)
            cumulative_energy = np.cumsum(psd)
            total_energy = cumulative_energy[-1]
            rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
            features[f'SPECTRAL_ROLLOFF_ch{ch+1}'] = f[rolloff_idx[0]] if len(rolloff_idx) > 0 else f[-1]
            
            # 6. Spectral Flux
            features[f'SPECTRAL_FLUX_ch{ch+1}'] = self._calculate_spectral_flux(psd)
        
        return features
    
    def _extract_time_frequency_features(self, emg_data):
        """Extract 4 time-frequency features per channel"""
        features = {}
        n_channels = emg_data.shape[1]
        
        for ch in range(n_channels):
            signal = emg_data[:, ch]
            
            # Short-Time Fourier Transform
            f, t, stft = signal.stft(signal, fs=self.fs, nperseg=64)
            stft_magnitude = np.abs(stft)
            
            # 1. Mean STFT Magnitude
            features[f'STFT_MEAN_ch{ch+1}'] = np.mean(stft_magnitude)
            
            # 2. STD STFT Magnitude
            features[f'STFT_STD_ch{ch+1}'] = np.std(stft_magnitude)
            
            # 3. STFT Energy
            features[f'STFT_ENERGY_ch{ch+1}'] = np.sum(stft_magnitude**2)
            
            # 4. STFT Entropy
            features[f'STFT_ENTROPY_ch{ch+1}'] = self._calculate_entropy(stft_magnitude)
        
        return features
    
    def _extract_advanced_features(self, emg_data):
        """Extract advanced features per channel"""
        features = {}
        n_channels = emg_data.shape[1]
        
        for ch in range(n_channels):
            signal = emg_data[:, ch]
            
            # 1. Skewness
            features[f'SKEW_ch{ch+1}'] = skew(signal)
            
            # 2. Kurtosis
            features[f'KURT_ch{ch+1}'] = kurtosis(signal)
            
            # 3. Sample Entropy
            features[f'SAMP_ENTROPY_ch{ch+1}'] = self._sample_entropy(signal)
            
            # 4. Detrended Fluctuation Analysis
            features[f'DFA_ch{ch+1}'] = self._detrended_fluctuation_analysis(signal)
        
        return features
    
    def _extract_cross_channel_features(self, emg_data):
        """Extract cross-channel features"""
        features = {}
        n_channels = emg_data.shape[1]
        
        # Channel correlation features
        corr_matrix = np.corrcoef(emg_data.T)
        
        # 1. Mean correlation between channels
        upper_tri = np.triu(corr_matrix, k=1)
        features['MEAN_CORRELATION'] = np.mean(upper_tri[upper_tri != 0])
        
        # 2. Max correlation between channels
        features['MAX_CORRELATION'] = np.max(upper_tri[upper_tri != 0])
        
        # 3. Channel activation balance
        channel_means = np.mean(np.abs(emg_data), axis=0)
        features['ACTIVATION_BALANCE'] = np.std(channel_means) / np.mean(channel_means)
        
        # 4. Spatial coherence
        features['SPATIAL_COHERENCE'] = self._calculate_spatial_coherence(emg_data)
        
        return features
    
    def _zero_crossing_rate(self, signal):
        """Calculate zero crossing rate"""
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        return len(zero_crossings) / len(signal)
    
    def _slope_sign_changes(self, signal):
        """Calculate slope sign changes"""
        diff = np.diff(signal)
        slope_changes = np.where(np.diff(np.sign(diff)))[0]
        return len(slope_changes) / len(signal)
    
    def _calculate_spectral_flux(self, psd):
        """Calculate spectral flux"""
        if len(psd) < 2:
            return 0
        return np.sum(np.abs(np.diff(psd)))
    
    def _calculate_entropy(self, data):
        """Calculate entropy of data"""
        # Normalize to probabilities
        data_norm = data / (np.sum(data) + 1e-10)
        # Remove zeros
        data_norm = data_norm[data_norm > 0]
        # Calculate entropy
        return -np.sum(data_norm * np.log2(data_norm + 1e-10))
    
    def _sample_entropy(self, signal, m=2, r=0.2):
        """Calculate sample entropy"""
        N = len(signal)
        if N < m + 1:
            return 0
        
        # Normalize signal
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _get_matches(signal, m, r):
            N = len(signal)
            patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
            matches = np.zeros(N - m + 1)
            for i in range(N - m + 1):
                for j in range(i + 1, N - m + 1):
                    if _maxdist(patterns[i], patterns[j], m) <= r:
                        matches[i] += 1
                        matches[j] += 1
            return matches
        
        matches_m = _get_matches(signal, m, r)
        matches_m1 = _get_matches(signal, m + 1, r)
        
        phi_m = np.mean(matches_m) / (N - m)
        phi_m1 = np.mean(matches_m1) / (N - m - 1)
        
        return -np.log(phi_m1 / phi_m) if phi_m1 > 0 and phi_m > 0 else 0
    
    def _detrended_fluctuation_analysis(self, signal, min_n=4, max_n=None):
        """Calculate DFA scaling exponent"""
        if max_n is None:
            max_n = len(signal) // 4
        
        # Integrate signal
        y = np.cumsum(signal - np.mean(signal))
        
        # Calculate fluctuation for different window sizes
        n_values = np.logspace(np.log10(min_n), np.log10(max_n), 10).astype(int)
        fluctuations = []
        
        for n in n_values:
            # Divide signal into windows
            n_windows = len(y) // n
            if n_windows < 2:
                continue
            
            # Calculate local trend and fluctuation
            local_fluctuations = []
            for i in range(n_windows):
                start = i * n
                end = start + n
                window = y[start:end]
                
                # Fit linear trend
                x = np.arange(n)
                coeffs = np.polyfit(x, window, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean((window - trend)**2))
                local_fluctuations.append(fluctuation)
            
            # Average fluctuation for this window size
            fluctuations.append(np.mean(local_fluctuations))
        
        if len(fluctuations) < 2:
            return 0
        
        # Calculate scaling exponent
        log_n = np.log(n_values[:len(fluctuations)])
        log_f = np.log(fluctuations)
        
        # Linear regression
        coeffs = np.polyfit(log_n, log_f, 1)
        return coeffs[0]
    
    def _calculate_spatial_coherence(self, emg_data):
        """Calculate spatial coherence across channels"""
        n_channels = emg_data.shape[1]
        if n_channels < 2:
            return 0
        
        # Calculate coherence between all channel pairs
        coherences = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                f, coherence = signal.coherence(emg_data[:, i], emg_data[:, j], fs=self.fs)
                coherences.append(np.mean(coherence))
        
        return np.mean(coherences) if coherences else 0
    
    def _create_feature_matrix(self, features):
        """Convert features dictionary to matrix"""
        feature_names = list(features.keys())
        feature_values = list(features.values())
        
        self.feature_names = feature_names
        return np.array(feature_values).reshape(1, -1)

class FeatureSelector:
    def __init__(self):
        self.selected_features = None
        self.feature_scores = None
    
    def select_features(self, X, y, method='mutual_info', n_features=50):
        """Select best features using specified method"""
        print(f"🔍 Selecting features using {method}...")
        
        if method == 'mutual_info':
            scores = mutual_info_classif(X, y)
        elif method == 'f_score':
            scores, _ = f_classif(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Select top features
        top_indices = np.argsort(scores)[-n_features:]
        self.selected_features = top_indices
        self.feature_scores = scores[top_indices]
        
        print(f"   ✅ Selected {len(top_indices)} features")
        print(f"   ✅ Score range: {scores[top_indices].min():.3f} - {scores[top_indices].max():.3f}")
        
        return top_indices, scores[top_indices]

def main():
    """Demo feature extraction"""
    print("🚀 Modern EMG Feature Engineering Demo")
    print("=" * 50)
    
    # Create sample EMG data
    np.random.seed(42)
    n_samples = 1000
    n_channels = 8
    
    # Generate synthetic EMG-like data
    emg_data = np.random.randn(n_samples, n_channels) * 0.0001
    labels = np.random.randint(0, 7, n_samples)
    
    print(f"📊 Sample Data: {n_samples} samples, {n_channels} channels")
    
    # Initialize feature extractor
    extractor = ModernEMGFeatureExtractor(fs=1000)
    
    # Extract features
    feature_matrix, feature_names = extractor.extract_all_features(emg_data, labels)
    
    print(f"\n📈 Feature Matrix Shape: {feature_matrix.shape}")
    print(f"📈 Feature Names: {len(feature_names)} features")
    
    # Feature selection
    selector = FeatureSelector()
    selected_indices, scores = selector.select_features(
        feature_matrix, labels, method='mutual_info', n_features=20
    )
    
    print(f"\n🎯 Selected Features:")
    for i, idx in enumerate(selected_indices[:10]):  # Show top 10
        print(f"   {i+1:2d}. {feature_names[idx]:30s} (score: {scores[i]:.3f})")
    
    print(f"\n✅ Feature engineering complete!")
    print(f"✅ Ready for model training with {len(selected_indices)} selected features")

if __name__ == "__main__":
    main()
