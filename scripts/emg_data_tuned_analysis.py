#!/usr/bin/env python3
"""
EMG Data Analysis - Tuned for Your Specific Dataset
Properly configured for the actual EMG data structure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EMGDataTunedAnalyzer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.fs = 1000  # Target sampling frequency
        self.n_channels = 8  # Your data has 8 channels
        self.gesture_classes = [1, 2, 3, 4, 5, 6]  # Your actual gesture classes
        self.results = {}
        
    def load_subject_data(self, subject_id):
        """Load data for a specific subject - tuned for your data format"""
        subject_folder = self.data_path / f"{subject_id:02d}"
        files = list(subject_folder.glob("*.txt"))
        
        if not files:
            raise FileNotFoundError(f"No data files found for subject {subject_id} in {subject_folder}")
        
        print(f"   📁 Found {len(files)} files: {[f.name for f in files]}")
        
        all_data = []
        for file in files:
            try:
                print(f"   📄 Loading {file.name}...")
                # Your data is tab-separated, not comma-separated
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
        return pd.concat(all_data, ignore_index=True)
    
    def preprocess_data(self, data):
        """Preprocess data to handle irregular sampling and resample to 1000Hz"""
        print("🔧 Preprocessing EMG data...")
        
        # Extract time and EMG channels
        time_col = data['time'].values
        emg_channels = data.iloc[:, 1:9].values  # channels 1-8
        labels = data['class'].values
        
        # Handle irregular sampling - resample to 1000Hz
        print(f"   📊 Original data: {len(data)} samples")
        print(f"   📊 Time range: {time_col[0]} - {time_col[-1]} ms")
        print(f"   📊 Duration: {(time_col[-1] - time_col[0]) / 1000:.1f} seconds")
        
        # Create regular time grid at 1000Hz
        start_time = time_col[0]
        end_time = time_col[-1]
        regular_time = np.arange(start_time, end_time, 1)  # 1ms intervals
        
        # Interpolate EMG data to regular grid
        emg_resampled = np.zeros((len(regular_time), self.n_channels))
        for ch in range(self.n_channels):
            emg_resampled[:, ch] = np.interp(regular_time, time_col, emg_channels[:, ch])
        
        # Interpolate labels (nearest neighbor)
        labels_resampled = np.interp(regular_time, time_col, labels)
        labels_resampled = np.round(labels_resampled).astype(int)
        
        print(f"   ✅ Resampled to: {len(regular_time)} samples at 1000Hz")
        
        return emg_resampled, labels_resampled, regular_time
    
    def analyze_signal_quality(self, emg_data, labels):
        """Analyze signal quality - tuned for your data"""
        print("🔍 Analyzing Signal Quality...")
        
        quality_metrics = {}
        
        # 1. Signal-to-Noise Ratio
        signal_power = np.mean(emg_data**2, axis=0)
        noise_power = np.var(np.diff(emg_data, axis=0), axis=0)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        quality_metrics['snr_db'] = snr
        
        # 2. Artifact Detection
        threshold = 3 * np.std(emg_data, axis=0)
        artifacts = np.any(np.abs(emg_data) > threshold, axis=1)
        quality_metrics['artifact_ratio'] = np.sum(artifacts) / len(artifacts)
        
        # 3. Baseline Drift
        baseline_drift = np.max(emg_data, axis=0) - np.min(emg_data, axis=0)
        quality_metrics['baseline_drift'] = baseline_drift
        
        # 4. Channel Correlation
        channel_corr = np.corrcoef(emg_data.T)
        quality_metrics['channel_correlation'] = channel_corr
        
        # 5. Gesture Distribution
        gesture_counts = {}
        for gesture in self.gesture_classes:
            count = np.sum(labels == gesture)
            gesture_counts[gesture] = count
        
        quality_metrics['gesture_distribution'] = gesture_counts
        
        self.results['signal_quality'] = quality_metrics
        
        print(f"   ✅ SNR Range: {snr.min():.1f} - {snr.max():.1f} dB")
        print(f"   ✅ Artifact Ratio: {quality_metrics['artifact_ratio']:.3f}")
        print(f"   ✅ Gesture Distribution: {gesture_counts}")
        
        return quality_metrics
    
    def analyze_gesture_patterns(self, emg_data, labels):
        """Analyze gesture patterns - tuned for your 6 gesture classes"""
        print("🎯 Analyzing Gesture Patterns...")
        
        gesture_analysis = {}
        
        for gesture_class in self.gesture_classes:
            gesture_mask = labels == gesture_class
            gesture_data = emg_data[gesture_mask]
            
            if len(gesture_data) > 0:
                gesture_analysis[gesture_class] = {
                    'duration_seconds': len(gesture_data) / self.fs,
                    'samples_count': len(gesture_data),
                    'mean_amplitude': np.mean(np.abs(gesture_data), axis=0),
                    'amplitude_std': np.std(np.abs(gesture_data), axis=0),
                    'channel_activation': np.mean(np.abs(gesture_data), axis=0),
                    'activation_rank': np.argsort(np.mean(np.abs(gesture_data), axis=0))[::-1]
                }
        
        self.results['gesture_patterns'] = gesture_analysis
        
        print(f"   ✅ Found {len(gesture_analysis)} gesture classes")
        for gesture, metrics in gesture_analysis.items():
            print(f"   ✅ Gesture {gesture}: {metrics['duration_seconds']:.1f}s, "
                  f"{metrics['samples_count']} samples")
            print(f"      Top channels: {metrics['activation_rank'][:3]}")
        
        return gesture_analysis
    
    def create_visualizations(self, subject_id=1):
        """Create visualizations tuned for your data"""
        print("📊 Creating Visualizations...")
        
        # Load and preprocess data
        data = self.load_subject_data(subject_id)
        emg_data, labels, time = self.preprocess_data(data)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Raw EMG signals
        ax1 = plt.subplot(3, 3, 1)
        for i in range(self.n_channels):
            plt.plot(time/1000, emg_data[:, i] + i * 0.0001, label=f'Channel {i+1}', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (offset)')
        plt.title(f'Raw EMG Signals - Subject {subject_id}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Gesture labels over time
        ax2 = plt.subplot(3, 3, 2)
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.gesture_classes)))
        for i, gesture in enumerate(self.gesture_classes):
            mask = labels == gesture
            if np.any(mask):
                plt.scatter(time[mask]/1000, labels[mask], c=[colors[i]], 
                           label=f'Gesture {gesture}', alpha=0.6, s=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Gesture Class')
        plt.title('Gesture Labels Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Signal amplitude distribution
        ax3 = plt.subplot(3, 3, 3)
        plt.hist(np.abs(emg_data).flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Amplitude')
        plt.ylabel('Frequency')
        plt.title('Signal Amplitude Distribution')
        plt.yscale('log')
        
        # 4. Channel correlation heatmap
        ax4 = plt.subplot(3, 3, 4)
        channel_corr = np.corrcoef(emg_data.T)
        sns.heatmap(channel_corr, annot=True, cmap='coolwarm', center=0,
                   xticklabels=[f'Ch{i+1}' for i in range(self.n_channels)],
                   yticklabels=[f'Ch{i+1}' for i in range(self.n_channels)])
        plt.title('Channel Correlation Matrix')
        
        # 5. Gesture duration analysis
        ax5 = plt.subplot(3, 3, 5)
        gesture_durations = []
        gesture_classes = []
        
        for gesture_class in self.gesture_classes:
            gesture_mask = labels == gesture_class
            if np.any(gesture_mask):
                duration = np.sum(gesture_mask) / self.fs
                gesture_durations.append(duration)
                gesture_classes.append(gesture_class)
        
        plt.bar(gesture_classes, gesture_durations, color=colors[:len(gesture_classes)])
        plt.xlabel('Gesture Class')
        plt.ylabel('Duration (s)')
        plt.title('Gesture Duration Analysis')
        
        # 6. Frequency spectrum
        ax6 = plt.subplot(3, 3, 6)
        f, psd = signal.welch(emg_data[:, 0], fs=self.fs, nperseg=min(1024, len(emg_data)//4))
        plt.semilogy(f, psd)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Frequency Spectrum - Channel 1')
        plt.xlim(0, 500)
        
        # 7. Gesture amplitude profiles
        ax7 = plt.subplot(3, 3, 7)
        for i, gesture_class in enumerate(self.gesture_classes):
            gesture_mask = labels == gesture_class
            if np.any(gesture_mask):
                gesture_data = emg_data[gesture_mask]
                mean_amplitude = np.mean(np.abs(gesture_data), axis=0)
                plt.plot(mean_amplitude, 'o-', color=colors[i], label=f'Gesture {gesture_class}')
        plt.xlabel('Channel')
        plt.ylabel('Mean Absolute Amplitude')
        plt.title('Gesture Amplitude Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Signal quality metrics
        ax8 = plt.subplot(3, 3, 8)
        if 'signal_quality' in self.results:
            snr = self.results['signal_quality']['snr_db']
            plt.bar(range(self.n_channels), snr, color='skyblue')
            plt.xlabel('Channel')
            plt.ylabel('SNR (dB)')
            plt.title('Signal-to-Noise Ratio by Channel')
            plt.grid(True, alpha=0.3)
        
        # 9. Gesture class distribution
        ax9 = plt.subplot(3, 3, 9)
        gesture_counts = [np.sum(labels == g) for g in self.gesture_classes]
        plt.pie(gesture_counts, labels=[f'Gesture {g}' for g in self.gesture_classes], 
                autopct='%1.1f%%', colors=colors)
        plt.title('Gesture Class Distribution')
        
        plt.tight_layout()
        plt.savefig(f'emg_analysis_subject_{subject_id}_tuned.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Visualizations saved as 'emg_analysis_subject_{subject_id}_tuned.png'")
    
    def generate_analysis_report(self):
        """Generate analysis report tuned for your data"""
        print("📋 Generating Analysis Report...")
        
        report = []
        report.append("# EMG Data Analysis Report - Tuned for Your Dataset")
        report.append("=" * 60)
        report.append("")
        
        # Data characteristics
        report.append("## Dataset Characteristics")
        report.append(f"- **Channels**: {self.n_channels} EMG channels")
        report.append(f"- **Gesture Classes**: {self.gesture_classes}")
        report.append(f"- **Sampling Rate**: {self.fs} Hz (resampled)")
        report.append(f"- **Data Format**: Tab-separated text files")
        report.append("")
        
        # Signal Quality Summary
        if 'signal_quality' in self.results:
            sq = self.results['signal_quality']
            report.append("## Signal Quality Analysis")
            report.append(f"- **SNR Range**: {sq['snr_db'].min():.1f} - {sq['snr_db'].max():.1f} dB")
            report.append(f"- **Artifact Ratio**: {sq['artifact_ratio']:.3f}")
            report.append(f"- **Baseline Drift Range**: {sq['baseline_drift'].min():.2e} - {sq['baseline_drift'].max():.2e}")
            report.append("")
        
        # Gesture Patterns Summary
        if 'gesture_patterns' in self.results:
            gp = self.results['gesture_patterns']
            report.append("## Gesture Pattern Analysis")
            report.append(f"- **Number of Gesture Classes**: {len(gp)}")
            for gesture, metrics in gp.items():
                report.append(f"- **Gesture {gesture}**: {metrics['duration_seconds']:.1f}s duration, "
                            f"{metrics['samples_count']} samples")
                top_channels = metrics['activation_rank'][:3]
                report.append(f"  - Top channels: {top_channels}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations for Your Data")
        report.append("")
        report.append("1. **Data Preprocessing**: Resample to regular 1000Hz grid")
        report.append("2. **Feature Extraction**: Use 8-channel specific features")
        report.append("3. **Model Training**: Focus on 6 gesture classes (1-6)")
        report.append("4. **Cross-Validation**: Use subject-specific splits")
        report.append("5. **Real-time Processing**: Optimize for 1000Hz input")
        
        # Save report
        with open('emg_analysis_report_tuned.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("   ✅ Analysis report saved as 'emg_analysis_report_tuned.md'")
        
        return '\n'.join(report)

def main():
    """Run tuned analysis for your EMG data"""
    print("🚀 EMG Data Analysis - Tuned for Your Dataset")
    print("=" * 60)
    
    # Initialize analyzer
    data_path = "/Users/amitsubhash/Downloads/EMG/EMG_gestures/EMG_data"
    analyzer = EMGDataTunedAnalyzer(data_path)
    
    # Analyze first subject
    print(f"\n📊 Analyzing Subject 1")
    data = analyzer.load_subject_data(1)
    
    # Preprocess data
    emg_data, labels, time = analyzer.preprocess_data(data)
    
    # Run analyses
    analyzer.analyze_signal_quality(emg_data, labels)
    analyzer.analyze_gesture_patterns(emg_data, labels)
    
    # Create visualizations
    analyzer.create_visualizations(subject_id=1)
    
    # Generate report
    report = analyzer.generate_analysis_report()
    
    print("\n" + "=" * 60)
    print("🎉 Analysis Complete!")
    print("=" * 60)
    print("\nKey Files Generated:")
    print("- emg_analysis_subject_1_tuned.png (visualizations)")
    print("- emg_analysis_report_tuned.md (detailed report)")
    print("\nNext Steps:")
    print("1. Review the tuned visualizations")
    print("2. Check the analysis report for data-specific insights")
    print("3. Use this tuned approach for all other scripts")

if __name__ == "__main__":
    main()
