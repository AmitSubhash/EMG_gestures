#!/usr/bin/env python3
"""
Advanced EMG Feature Extraction (2024-2025 SOTA)
Based on latest research: Wavelet transforms, Sample entropy, DFA, etc.
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

class AdvancedEMGFeatureExtractor:
    def __init__(self, fs=1000):
        self.fs = fs
        self.feature_names = []
        
    def extract_comprehensive_features(self, emg_data, labels=None):
        """Extract comprehensive feature set based on 2024-2025 research"""
        print("🔧 Extracting Advanced EMG Features (2024-2025 SOTA)...")
        
        features = {}
        
        # Time Domain Features (12 features per channel)
        print("   📊 Time Domain Features...")
        features.update(self._extract_advanced_time_domain_features(emg_data))
        
        # Frequency Domain Features (10 features per channel)
        print("   📊 Frequency Domain Features...")
        features.update(self._extract_advanced_frequency_domain_features(emg_data))
        
        # Time-Frequency Features (8 features per channel)
        print("   📊 Time-Frequency Features...")
        features.update(self._extract_advanced_time_frequency_features(emg_data))
        
        # Advanced Statistical Features (6 features per channel)
        print("   📊 Advanced Statistical Features...")
        features.update(self._extract_advanced_statistical_features(emg_data))
        
        # Nonlinear Features (4 features per channel)
        print("   📊 Nonlinear Features...")
        features.update(self._extract_nonlinear_features(emg_data))
        
        # Cross-Channel Features (8 features)
        print("   📊 Cross-Channel Features...")
        features.update(self._extract_cross_channel_features(emg_data))
        
        # Convert to feature matrix
        feature_matrix = self._create_feature_matrix(features)
        
        print(f"   ✅ Extracted {feature_matrix.shape[1]} features from {emg_data.shape[1]} channels")
        
        return feature_matrix, self.feature_names
    
    def _extract_advanced_time_domain_features(self, emg_data):
        """Extract 12 advanced time domain features per channel"""
        features = {}
        n_channels = emg_data.shape[1]
        
        for ch in range(n_channels):
            signal = emg_data[:, ch]
            
            # Basic features
            features[f'MAV_ch{ch+1}'] = np.mean(np.abs(signal))
            features[f'RMS_ch{ch+1}'] = np.sqrt(np.mean(signal**2))
            features[f'VAR_ch{ch+1}'] = np.var(signal)
            features[f'STD_ch{ch+1}'] = np.std(signal)
            features[f'WL_ch{ch+1}'] = np.sum(np.abs(np.diff(signal)))
            features[f'ZC_ch{ch+1}'] = self._zero_crossing_rate(signal)
            features[f'SSC_ch{ch+1}'] = self._slope_sign_changes(signal)
            features[f'IEMG_ch{ch+1}'] = np.sum(np.abs(signal))
            
            # Advanced features
            features[f'ARV_ch{ch+1}'] = np.mean(np.abs(signal))  # Average Rectified Value
            features[f'SSI_ch{ch+1}'] = np.sum(signal**2)  # Simple Square Integral
            features[f'TM_ch{ch+1}'] = np.mean(np.abs(signal**3))**(1/3)  # Third Moment
            features[f'LOG_ch{ch+1}'] = np.exp(np.mean(np.log(np.abs(signal) + 1e-10)))  # Log Detector
        
        return features
    
    def _extract_advanced_frequency_domain_features(self, emg_data):
        """Extract 10 advanced frequency domain features per channel"""
        features = {}
        n_channels = emg_data.shape[1]
        
        for ch in range(n_channels):
            signal = emg_data[:, ch]
            
            # Power spectral density
            f, psd = welch(signal, fs=self.fs, nperseg=min(256, len(signal)//4))
            
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
            features[f'SPECTRAL_CENTROID_SPREAD_ch{ch+1}'] = self._spectral_centroid_spread(f, psd)
            features[f'SPECTRAL_SLOPE_ch{ch+1}'] = self._spectral_slope(f, psd)
            features[f'SPECTRAL_DECREASE_ch{ch+1}'] = self._spectral_decrease(f, psd)
            features[f'SPECTRAL_VARIATION_ch{ch+1}'] = np.var(psd)
        
        return features
    
    def _extract_advanced_time_frequency_features(self, emg_data):
        """Extract 8 advanced time-frequency features per channel"""
        features = {}
        n_channels = emg_data.shape[1]
        
        for ch in range(n_channels):
            signal = emg_data[:, ch]
            
            # Short-Time Fourier Transform
            f, t, stft = signal.stft(signal, fs=self.fs, nperseg=64)
            stft_magnitude = np.abs(stft)
            
            # STFT features
            features[f'STFT_MEAN_ch{ch+1}'] = np.mean(stft_magnitude)
            features[f'STFT_STD_ch{ch+1}'] = np.std(stft_magnitude)
            features[f'STFT_ENERGY_ch{ch+1}'] = np.sum(stft_magnitude**2)
            features[f'STFT_ENTROPY_ch{ch+1}'] = self._calculate_entropy(stft_magnitude)
            
            # Wavelet features
            coeffs = pywt.wavedec(signal, 'db4', level=4)
            features[f'WAVELET_ENERGY_ch{ch+1}'] = np.sum([np.sum(c**2) for c in coeffs])
            features[f'WAVELET_ENTROPY_ch{ch+1}'] = self._wavelet_entropy(coeffs)
            features[f'WAVELET_VAR_ch{ch+1}'] = np.var([np.var(c) for c in coeffs])
            features[f'WAVELET_MEAN_ch{ch+1}'] = np.mean([np.mean(c) for c in coeffs])
        
        return features
    
    def _extract_advanced_statistical_features(self, emg_data):
        """Extract 6 advanced statistical features per channel"""
        features = {}
        n_channels = emg_data.shape[1]
        
        for ch in range(n_channels):
            signal = emg_data[:, ch]
            
            # Statistical features
            features[f'SKEW_ch{ch+1}'] = skew(signal)
            features[f'KURT_ch{ch+1}'] = kurtosis(signal)
            features[f'RANGE_ch{ch+1}'] = np.ptp(signal)
            features[f'PERCENTILE_25_ch{ch+1}'] = np.percentile(signal, 25)
            features[f'PERCENTILE_75_ch{ch+1}'] = np.percentile(signal, 75)
            features[f'IQR_ch{ch+1}'] = np.percentile(signal, 75) - np.percentile(signal, 25)
        
        return features
    
    def _extract_nonlinear_features(self, emg_data):
        """Extract 4 nonlinear features per channel"""
        features = {}
        n_channels = emg_data.shape[1]
        
        for ch in range(n_channels):
            signal = emg_data[:, ch]
            
            # Nonlinear features
            features[f'SAMP_ENTROPY_ch{ch+1}'] = self._sample_entropy(signal)
            features[f'DFA_ch{ch+1}'] = self._detrended_fluctuation_analysis(signal)
            features[f'LYAPUNOV_ch{ch+1}'] = self._largest_lyapunov_exponent(signal)
            features[f'CORRELATION_DIM_ch{ch+1}'] = self._correlation_dimension(signal)
        
        return features
    
    def _extract_cross_channel_features(self, emg_data):
        """Extract 8 cross-channel features"""
        features = {}
        n_channels = emg_data.shape[1]
        
        if n_channels < 2:
            return features
        
        # Channel correlation
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
        
        # Spatial coherence
        features['SPATIAL_COHERENCE'] = self._spatial_coherence(emg_data)
        features['CHANNEL_ACTIVATION_BALANCE'] = self._channel_activation_balance(emg_data)
        
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
    
    def _spectral_centroid_spread(self, f, psd):
        """Calculate spectral centroid spread"""
        centroid = np.sum(f * psd) / np.sum(psd)
        spread = np.sqrt(np.sum(((f - centroid)**2) * psd) / np.sum(psd))
        return spread
    
    def _spectral_slope(self, f, psd):
        """Calculate spectral slope"""
        if len(f) < 2:
            return 0
        # Linear regression on log-log scale
        log_f = np.log(f[1:] + 1e-10)
        log_psd = np.log(psd[1:] + 1e-10)
        slope, _ = np.polyfit(log_f, log_psd, 1)
        return slope
    
    def _spectral_decrease(self, f, psd):
        """Calculate spectral decrease"""
        if len(f) < 2:
            return 0
        numerator = np.sum((psd[1:] - psd[0]) / (np.arange(1, len(f)) + 1e-10))
        denominator = np.sum(psd[1:])
        return numerator / (denominator + 1e-10)
    
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
    
    def _largest_lyapunov_exponent(self, signal, max_iter=1000):
        """Calculate largest Lyapunov exponent"""
        # Simplified implementation
        N = len(signal)
        if N < 100:
            return 0
        
        # Phase space reconstruction
        m = 3  # Embedding dimension
        tau = 1  # Time delay
        
        # Create phase space vectors
        vectors = []
        for i in range(N - (m-1)*tau):
            vector = [signal[i + j*tau] for j in range(m)]
            vectors.append(vector)
        
        vectors = np.array(vectors)
        
        # Calculate divergence
        divergences = []
        for i in range(min(100, len(vectors)-1)):
            distances = np.linalg.norm(vectors[i+1:] - vectors[i], axis=1)
            if len(distances) > 0:
                divergences.append(np.mean(distances))
        
        if len(divergences) < 2:
            return 0
        
        # Calculate Lyapunov exponent
        time_steps = np.arange(len(divergences))
        log_div = np.log(divergences + 1e-10)
        
        # Linear regression
        coeffs = np.polyfit(time_steps, log_div, 1)
        return coeffs[0]
    
    def _correlation_dimension(self, signal, max_radius=0.1):
        """Calculate correlation dimension"""
        N = len(signal)
        if N < 100:
            return 0
        
        # Phase space reconstruction
        m = 3
        tau = 1
        
        vectors = []
        for i in range(N - (m-1)*tau):
            vector = [signal[i + j*tau] for j in range(m)]
            vectors.append(vector)
        
        vectors = np.array(vectors)
        
        # Calculate correlation sum
        radii = np.logspace(-3, np.log10(max_radius), 10)
        correlation_sums = []
        
        for r in radii:
            count = 0
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    if np.linalg.norm(vectors[i] - vectors[j]) < r:
                        count += 1
            
            correlation_sums.append(count)
        
        # Calculate correlation dimension
        log_radii = np.log(radii)
        log_sums = np.log(np.array(correlation_sums) + 1e-10)
        
        # Linear regression
        coeffs = np.polyfit(log_radii, log_sums, 1)
        return coeffs[0]
    
    def _spatial_coherence(self, emg_data):
        """Calculate spatial coherence across channels"""
        n_channels = emg_data.shape[1]
        if n_channels < 2:
            return 0
        
        coherences = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
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
    """Demo advanced feature extraction"""
    print("🚀 Advanced EMG Feature Extraction Demo (2024-2025 SOTA)")
    print("=" * 60)
    
    # Create sample EMG data
    np.random.seed(42)
    n_samples = 1000
    n_channels = 8
    
    # Generate synthetic EMG-like data
    emg_data = np.random.randn(n_samples, n_channels) * 0.0001
    labels = np.random.randint(0, 6, n_samples)
    
    print(f"📊 Sample Data: {n_samples} samples, {n_channels} channels")
    
    # Initialize feature extractor
    extractor = AdvancedEMGFeatureExtractor(fs=1000)
    
    # Extract features
    feature_matrix, feature_names = extractor.extract_comprehensive_features(emg_data, labels)
    
    print(f"\n📈 Feature Matrix Shape: {feature_matrix.shape}")
    print(f"📈 Total Features: {len(feature_names)}")
    
    # Show feature categories
    categories = {
        'Time Domain': [f for f in feature_names if any(x in f for x in ['MAV', 'RMS', 'VAR', 'STD', 'WL', 'ZC', 'SSC', 'IEMG', 'ARV', 'SSI', 'TM', 'LOG'])],
        'Frequency Domain': [f for f in feature_names if any(x in f for x in ['PSD', 'SPECTRAL'])],
        'Time-Frequency': [f for f in feature_names if any(x in f for x in ['STFT', 'WAVELET'])],
        'Statistical': [f for f in feature_names if any(x in f for x in ['SKEW', 'KURT', 'RANGE', 'PERCENTILE', 'IQR'])],
        'Nonlinear': [f for f in feature_names if any(x in f for x in ['SAMP_ENTROPY', 'DFA', 'LYAPUNOV', 'CORRELATION_DIM'])],
        'Cross-Channel': [f for f in feature_names if any(x in f for x in ['CORRELATION', 'ENERGY', 'SPATIAL', 'CHANNEL'])]
    }
    
    print(f"\n📊 Feature Categories:")
    for category, features in categories.items():
        print(f"   {category}: {len(features)} features")
    
    print(f"\n✅ Advanced feature extraction complete!")
    print(f"✅ Ready for state-of-the-art model training")

if __name__ == "__main__":
    main()
