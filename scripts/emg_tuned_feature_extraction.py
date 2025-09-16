#!/usr/bin/env python3
"""
EMG Feature Extraction - Tuned for Your Specific Dataset
Properly configured for 8-channel EMG data with 6 gesture classes
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, cwt, morlet2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import mutual_info_classif, f_classif
import pywt
import warnings
warnings.filterwarnings('ignore')

class EMGTunedFeatureExtractor:
    def __init__(self, fs=1000, n_channels=8, gesture_classes=[1, 2, 3, 4, 5, 6]):
        self.fs = fs
        self.n_channels = n_channels
        self.gesture_classes = gesture_classes
        self.feature_names = []
        
    def extract_tuned_features(self, emg_data, labels=None):
        """Extract features tuned for your specific EMG data"""
        print("🔧 Extracting Tuned EMG Features...")
        print(f"   📊 Data shape: {emg_data.shape}")
        print(f"   📊 Channels: {self.n_channels}")
        print(f"   📊 Gesture classes: {self.gesture_classes}")
        
        features = {}
        
        # Time Domain Features (10 features per channel)
        print("   📊 Time Domain Features...")
        features.update(self._extract_time_domain_features(emg_data))
        
        # Frequency Domain Features (8 features per channel)
        print("   📊 Frequency Domain Features...")
        features.update(self._extract_frequency_domain_features(emg_data))
        
        # Time-Frequency Features (6 features per channel)
        print("   📊 Time-Frequency Features...")
        features.update(self._extract_time_frequency_features(emg_data))
        
        # Advanced Features (4 features per channel)
        print("   📊 Advanced Features...")
        features.update(self._extract_advanced_features(emg_data))
        
        # Cross-Channel Features (10 features)
        print("   📊 Cross-Channel Features...")
        features.update(self._extract_cross_channel_features(emg_data))
        
        # Convert to feature matrix
        feature_matrix = self._create_feature_matrix(features)
        
        print(f"   ✅ Extracted {feature_matrix.shape[1]} features")
        print(f"   ✅ Features per channel: {feature_matrix.shape[1] // self.n_channels}")
        
        return feature_matrix, self.feature_names
    
    def _extract_time_domain_features(self, emg_data):
        """Extract 10 time domain features per channel"""
        features = {}
        
        for ch in range(self.n_channels):
            channel_signal = emg_data[:, ch]
            
            # Basic features
            features[f'MAV_ch{ch+1}'] = np.mean(np.abs(channel_channel_signal))
            features[f'RMS_ch{ch+1}'] = np.sqrt(np.mean(channel_channel_signal**2))
            features[f'VAR_ch{ch+1}'] = np.var(channel_channel_signal)
            features[f'STD_ch{ch+1}'] = np.std(channel_channel_signal)
            features[f'WL_ch{ch+1}'] = np.sum(np.abs(np.diff(channel_channel_signal)))
            features[f'ZC_ch{ch+1}'] = self._zero_crossing_rate(channel_channel_signal)
            features[f'SSC_ch{ch+1}'] = self._slope_sign_changes(channel_channel_signal)
            features[f'IEMG_ch{ch+1}'] = np.sum(np.abs(channel_channel_signal))
            
            # Advanced time features
            features[f'ARV_ch{ch+1}'] = np.mean(np.abs(channel_signal))  # Average Rectified Value
            features[f'SSI_ch{ch+1}'] = np.sum(channel_signal**2)  # Simple Square Integral
        
        return features
    
    def _extract_frequency_domain_features(self, emg_data):
        """Extract 8 frequency domain features per channel"""
        features = {}
        
        for ch in range(self.n_channels):
            channel_signal = emg_data[:, ch]
            
            # Power spectral density
            f, psd = welch(channel_signal, fs=self.fs, nperseg=min(256, len(channel_signal)//4))
            
            # Basic frequency features
            features[f'PSD_MEAN_ch{ch+1}'] = np.mean(psd)
            features[f'PSD_PEAK_ch{ch+1}'] = np.max(psd)
            features[f'SPECTRAL_CENTROID_ch{ch+1}'] = np.sum(f * psd) / np.sum(psd)
            features[f'SPECTRAL_BANDWIDTH_ch{ch+1}'] = np.sqrt(
                np.sum(((f - features[f'SPECTRAL_CENTROID_ch{ch+1}'])**2) * psd) / np.sum(psd)
            )
            
            # Advanced frequency features
            features[f'SPECTRAL_ROLLOFF_ch{ch+1}'] = self._spectral_rolloff(f, psd)
            features[f'SPECTRAL_FLUX_ch{ch+1}'] = self._spectral_flux(psd)
            features[f'SPECTRAL_SLOPE_ch{ch+1}'] = self._spectral_slope(f, psd)
            features[f'SPECTRAL_VARIATION_ch{ch+1}'] = np.var(psd)
        
        return features
    
    def _extract_time_frequency_features(self, emg_data):
        """Extract 6 time-frequency features per channel"""
        features = {}
        
        for ch in range(self.n_channels):
            channel_signal = emg_data[:, ch]
            
            # Short-Time Fourier Transform
            f, t, stft = signal.stft(channel_signal, fs=self.fs, nperseg=64)
            stft_magnitude = np.abs(stft)
            
            # STFT features
            features[f'STFT_MEAN_ch{ch+1}'] = np.mean(stft_magnitude)
            features[f'STFT_STD_ch{ch+1}'] = np.std(stft_magnitude)
            features[f'STFT_ENERGY_ch{ch+1}'] = np.sum(stft_magnitude**2)
            features[f'STFT_ENTROPY_ch{ch+1}'] = self._calculate_entropy(stft_magnitude)
            
            # Wavelet features
            coeffs = pywt.wavedec(channel_signal, 'db4', level=4)
            features[f'WAVELET_ENERGY_ch{ch+1}'] = np.sum([np.sum(c**2) for c in coeffs])
            features[f'WAVELET_ENTROPY_ch{ch+1}'] = self._wavelet_entropy(coeffs)
        
        return features
    
    def _extract_advanced_features(self, emg_data):
        """Extract 4 advanced features per channel"""
        features = {}
        
        for ch in range(self.n_channels):
            channel_signal = emg_data[:, ch]
            
            # Statistical features
            features[f'SKEW_ch{ch+1}'] = skew(channel_signal)
            features[f'KURT_ch{ch+1}'] = kurtosis(channel_signal)
            
            # Nonlinear features
            features[f'SAMP_ENTROPY_ch{ch+1}'] = self._sample_entropy(channel_signal)
            features[f'DFA_ch{ch+1}'] = self._detrended_fluctuation_analysis(channel_signal)
        
        return features
    
    def _extract_cross_channel_features(self, emg_data):
        """Extract 10 cross-channel features"""
        features = {}
        
        # Channel correlation features
        corr_matrix = np.corrcoef(emg_data.T)
        upper_tri = np.triu(corr_matrix, k=1)
        valid_corrs = upper_tri[upper_tri != 0]
        
        features['MEAN_CORRELATION'] = np.mean(valid_corrs)
        features['MAX_CORRELATION'] = np.max(valid_corrs)
        features['MIN_CORRELATION'] = np.min(valid_corrs)
        features['STD_CORRELATION'] = np.std(valid_corrs)
        
        # Channel energy distribution
        channel_energies = np.sum(emg_data**2, axis=0)
        features['ENERGY_ENTROPY'] = entropy(channel_energies / np.sum(channel_energies))
        features['ENERGY_BALANCE'] = np.std(channel_energies) / np.mean(channel_energies)
        
        # Spatial features
        features['SPATIAL_COHERENCE'] = self._spatial_coherence(emg_data)
        features['CHANNEL_ACTIVATION_BALANCE'] = self._channel_activation_balance(emg_data)
        
        # Channel dominance
        features['DOMINANT_CHANNEL'] = np.argmax(channel_energies)
        features['CHANNEL_VARIANCE'] = np.var(channel_energies)
        
        return features
    
    def _zero_crossing_rate(self, channel_signal):
        """Calculate zero crossing rate"""
        zero_crossings = np.where(np.diff(np.signbit(channel_signal)))[0]
        return len(zero_crossings) / len(channel_signal)
    
    def _slope_sign_changes(self, channel_signal):
        """Calculate slope sign changes"""
        diff = np.diff(channel_signal)
        slope_changes = np.where(np.diff(np.sign(diff)))[0]
        return len(slope_changes) / len(channel_signal)
    
    def _spectral_rolloff(self, f, psd, rolloff=0.85):
        """Calculate spectral rolloff"""
        cumulative_energy = np.cumsum(psd)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= rolloff * total_energy)[0]
        return f[rolloff_idx[0]] if len(rolloff_idx) > 0 else f[-1]
    
    def _spectral_flux(self, psd):
        """Calculate spectral flux"""
        if len(psd) < 2:
            return 0
        return np.sum(np.abs(np.diff(psd)))
    
    def _spectral_slope(self, f, psd):
        """Calculate spectral slope"""
        if len(f) < 2:
            return 0
        log_f = np.log(f[1:] + 1e-10)
        log_psd = np.log(psd[1:] + 1e-10)
        slope, _ = np.polyfit(log_f, log_psd, 1)
        return slope
    
    def _calculate_entropy(self, data):
        """Calculate entropy"""
        data_norm = data / (np.sum(data) + 1e-10)
        data_norm = data_norm[data_norm > 0]
        return -np.sum(data_norm * np.log2(data_norm + 1e-10))
    
    def _wavelet_entropy(self, coeffs):
        """Calculate wavelet entropy"""
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = sum(energies)
        probabilities = [e / total_energy for e in energies]
        return -sum(p * np.log2(p + 1e-10) for p in probabilities)
    
    def _sample_entropy(self, channel_signal, m=2, r=0.2):
        """Calculate sample entropy"""
        N = len(channel_signal)
        if N < m + 1:
            return 0
        
        # Normalize channel_signal
        channel_signal = (channel_signal - np.mean(channel_signal)) / np.std(channel_signal)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _get_matches(channel_signal, m, r):
            N = len(channel_signal)
            patterns = np.array([channel_signal[i:i + m] for i in range(N - m + 1)])
            matches = np.zeros(N - m + 1)
            for i in range(N - m + 1):
                for j in range(i + 1, N - m + 1):
                    if _maxdist(patterns[i], patterns[j], m) <= r:
                        matches[i] += 1
                        matches[j] += 1
            return matches
        
        matches_m = _get_matches(channel_signal, m, r)
        matches_m1 = _get_matches(channel_signal, m + 1, r)
        
        phi_m = np.mean(matches_m) / (N - m)
        phi_m1 = np.mean(matches_m1) / (N - m - 1)
        
        return -np.log(phi_m1 / phi_m) if phi_m1 > 0 and phi_m > 0 else 0
    
    def _detrended_fluctuation_analysis(self, channel_signal, min_n=4, max_n=None):
        """Calculate DFA scaling exponent"""
        if max_n is None:
            max_n = len(channel_signal) // 4
        
        # Integrate channel_signal
        y = np.cumsum(channel_signal - np.mean(channel_signal))
        
        # Calculate fluctuation for different window sizes
        n_values = np.logspace(np.log10(min_n), np.log10(max_n), 10).astype(int)
        fluctuations = []
        
        for n in n_values:
            n_windows = len(y) // n
            if n_windows < 2:
                continue
            
            local_fluctuations = []
            for i in range(n_windows):
                start = i * n
                end = start + n
                window = y[start:end]
                
                x = np.arange(n)
                coeffs = np.polyfit(x, window, 1)
                trend = np.polyval(coeffs, x)
                
                fluctuation = np.sqrt(np.mean((window - trend)**2))
                local_fluctuations.append(fluctuation)
            
            fluctuations.append(np.mean(local_fluctuations))
        
        if len(fluctuations) < 2:
            return 0
        
        log_n = np.log(n_values[:len(fluctuations)])
        log_f = np.log(fluctuations)
        
        coeffs = np.polyfit(log_n, log_f, 1)
        return coeffs[0]
    
    def _spatial_coherence(self, emg_data):
        """Calculate spatial coherence across channels"""
        coherences = []
        for i in range(self.n_channels):
            for j in range(i + 1, self.n_channels):
                f, coherence = signal.coherence(emg_data[:, i], emg_data[:, j], fs=self.fs)
                coherences.append(np.mean(coherence))
        
        return np.mean(coherences) if coherences else 0
    
    def _channel_activation_balance(self, emg_data):
        """Calculate channel activation balance"""
        channel_means = np.mean(np.abs(emg_data), axis=0)
        return np.std(channel_means) / (np.mean(channel_means) + 1e-10)
    
    def _create_feature_matrix(self, features):
        """Convert features dictionary to matrix"""
        feature_names = list(features.keys())
        feature_values = list(features.values())
        
        self.feature_names = feature_names
        return np.array(feature_values).reshape(1, -1)

def main():
    """Demo tuned feature extraction"""
    print("🚀 EMG Feature Extraction - Tuned for Your Dataset")
    print("=" * 60)
    
    # Create sample EMG data matching your format
    np.random.seed(42)
    n_samples = 1000
    n_channels = 8
    
    # Generate synthetic EMG-like data
    emg_data = np.random.randn(n_samples, n_channels) * 0.0001
    labels = np.random.choice([1, 2, 3, 4, 5, 6], n_samples)
    
    print(f"📊 Sample Data: {n_samples} samples, {n_channels} channels")
    print(f"📊 Gesture classes: {np.unique(labels)}")
    
    # Initialize tuned feature extractor
    extractor = EMGTunedFeatureExtractor(
        fs=1000, 
        n_channels=8, 
        gesture_classes=[1, 2, 3, 4, 5, 6]
    )
    
    # Extract features
    feature_matrix, feature_names = extractor.extract_tuned_features(emg_data, labels)
    
    print(f"\n📈 Feature Matrix Shape: {feature_matrix.shape}")
    print(f"📈 Total Features: {len(feature_names)}")
    print(f"📈 Features per Channel: {len(feature_names) // n_channels}")
    
    # Show feature categories
    categories = {
        'Time Domain': [f for f in feature_names if any(x in f for x in ['MAV', 'RMS', 'VAR', 'STD', 'WL', 'ZC', 'SSC', 'IEMG', 'ARV', 'SSI'])],
        'Frequency Domain': [f for f in feature_names if any(x in f for x in ['PSD', 'SPECTRAL'])],
        'Time-Frequency': [f for f in feature_names if any(x in f for x in ['STFT', 'WAVELET'])],
        'Advanced': [f for f in feature_names if any(x in f for x in ['SKEW', 'KURT', 'SAMP_ENTROPY', 'DFA'])],
        'Cross-Channel': [f for f in feature_names if any(x in f for x in ['CORRELATION', 'ENERGY', 'SPATIAL', 'CHANNEL', 'DOMINANT'])]
    }
    
    print(f"\n📊 Feature Categories:")
    for category, features in categories.items():
        print(f"   {category}: {len(features)} features")
    
    print(f"\n✅ Tuned feature extraction complete!")
    print(f"✅ Ready for your specific EMG data")

if __name__ == "__main__":
    main()
