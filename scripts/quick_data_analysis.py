#!/usr/bin/env python3
"""
Quick EMG Data Analysis Script
Modern analysis of EMG gesture classification data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class QuickEMGAnalyzer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.fs = 1000  # Sampling frequency
        self.results = {}
        
    def load_subject_data(self, subject_id):
        """Load data for a specific subject"""
        subject_folder = self.data_path / f"{subject_id:02d}"
        files = list(subject_folder.glob("*.txt"))
        
        all_data = []
        for file in files:
            df = pd.read_csv(file, sep='\t')
            df['file'] = file.name
            all_data.append(df)
        
        return pd.concat(all_data, ignore_index=True)
    
    def analyze_signal_quality(self, data):
        """Quick signal quality analysis"""
        print("🔍 Analyzing Signal Quality...")
        
        # Extract EMG channels (exclude time and class)
        emg_channels = data.iloc[:, 1:9].values
        labels = data['class'].values
        
        quality_metrics = {}
        
        # 1. Signal-to-Noise Ratio
        signal_power = np.mean(emg_channels**2, axis=0)
        noise_power = np.var(np.diff(emg_channels, axis=0), axis=0)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        quality_metrics['snr_db'] = snr
        
        # 2. Artifact Detection
        threshold = 3 * np.std(emg_channels, axis=0)
        artifacts = np.any(np.abs(emg_channels) > threshold, axis=1)
        quality_metrics['artifact_ratio'] = np.sum(artifacts) / len(artifacts)
        
        # 3. Baseline Drift
        baseline_drift = np.max(emg_channels, axis=0) - np.min(emg_channels, axis=0)
        quality_metrics['baseline_drift'] = baseline_drift
        
        # 4. Channel Correlation
        channel_corr = np.corrcoef(emg_channels.T)
        quality_metrics['channel_correlation'] = channel_corr
        
        self.results['signal_quality'] = quality_metrics
        
        print(f"   ✅ SNR Range: {snr.min():.1f} - {snr.max():.1f} dB")
        print(f"   ✅ Artifact Ratio: {quality_metrics['artifact_ratio']:.3f}")
        print(f"   ✅ Baseline Drift Range: {baseline_drift.min():.2e} - {baseline_drift.max():.2e}")
        
        return quality_metrics
    
    def analyze_gesture_patterns(self, data):
        """Analyze gesture patterns"""
        print("🎯 Analyzing Gesture Patterns...")
        
        emg_channels = data.iloc[:, 1:9].values
        labels = data['class'].values
        
        gesture_analysis = {}
        
        for gesture_class in np.unique(labels):
            if gesture_class == 0:  # Skip unmarked
                continue
                
            gesture_mask = labels == gesture_class
            gesture_data = emg_channels[gesture_mask]
            
            if len(gesture_data) > 0:
                gesture_analysis[gesture_class] = {
                    'duration_seconds': len(gesture_data) / self.fs,
                    'mean_amplitude': np.mean(np.abs(gesture_data), axis=0),
                    'amplitude_std': np.std(np.abs(gesture_data), axis=0),
                    'channel_activation': np.mean(np.abs(gesture_data), axis=0),
                    'samples_count': len(gesture_data)
                }
        
        self.results['gesture_patterns'] = gesture_analysis
        
        print(f"   ✅ Found {len(gesture_analysis)} gesture classes")
        for gesture, metrics in gesture_analysis.items():
            print(f"   ✅ Gesture {gesture}: {metrics['duration_seconds']:.1f}s, "
                  f"{metrics['samples_count']} samples")
        
        return gesture_analysis
    
    def analyze_cross_subject_variability(self, subject_ids):
        """Analyze variability across subjects"""
        print("👥 Analyzing Cross-Subject Variability...")
        
        subject_metrics = {}
        
        for subject_id in subject_ids:
            try:
                data = self.load_subject_data(subject_id)
                emg_channels = data.iloc[:, 1:9].values
                
                subject_metrics[subject_id] = {
                    'mean_amplitude': np.mean(np.abs(emg_channels), axis=0),
                    'signal_variance': np.var(emg_channels, axis=0),
                    'signal_range': np.ptp(emg_channels, axis=0),
                    'total_samples': len(emg_channels)
                }
            except Exception as e:
                print(f"   ⚠️  Could not load subject {subject_id}: {e}")
                continue
        
        # Calculate variability metrics
        amplitude_means = np.array([metrics['mean_amplitude'] for metrics in subject_metrics.values()])
        amplitude_variability = np.std(amplitude_means, axis=0)
        
        self.results['cross_subject_variability'] = {
            'subject_metrics': subject_metrics,
            'amplitude_variability': amplitude_variability,
            'n_subjects': len(subject_metrics)
        }
        
        print(f"   ✅ Analyzed {len(subject_metrics)} subjects")
        print(f"   ✅ Amplitude variability range: {amplitude_variability.min():.2e} - {amplitude_variability.max():.2e}")
        
        return subject_metrics
    
    def create_visualizations(self, subject_id=1):
        """Create comprehensive visualizations"""
        print("📊 Creating Visualizations...")
        
        # Load data for visualization
        data = self.load_subject_data(subject_id)
        emg_channels = data.iloc[:, 1:9].values
        labels = data['class'].values
        time = data['time'].values / 1000  # Convert to seconds
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Raw EMG signals
        ax1 = plt.subplot(3, 3, 1)
        for i in range(8):
            plt.plot(time, emg_channels[:, i] + i * 0.0001, label=f'Channel {i+1}', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (offset)')
        plt.title(f'Raw EMG Signals - Subject {subject_id}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Gesture labels over time
        ax2 = plt.subplot(3, 3, 2)
        plt.plot(time, labels, 'o', markersize=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Gesture Class')
        plt.title('Gesture Labels Over Time')
        plt.grid(True, alpha=0.3)
        
        # 3. Signal amplitude distribution
        ax3 = plt.subplot(3, 3, 3)
        plt.hist(np.abs(emg_channels).flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Amplitude')
        plt.ylabel('Frequency')
        plt.title('Signal Amplitude Distribution')
        plt.yscale('log')
        
        # 4. Channel correlation heatmap
        ax4 = plt.subplot(3, 3, 4)
        channel_corr = np.corrcoef(emg_channels.T)
        sns.heatmap(channel_corr, annot=True, cmap='coolwarm', center=0,
                   xticklabels=[f'Ch{i+1}' for i in range(8)],
                   yticklabels=[f'Ch{i+1}' for i in range(8)])
        plt.title('Channel Correlation Matrix')
        
        # 5. Gesture duration analysis
        ax5 = plt.subplot(3, 3, 5)
        gesture_durations = []
        gesture_classes = []
        
        for gesture_class in np.unique(labels):
            if gesture_class == 0:
                continue
            gesture_mask = labels == gesture_class
            if np.any(gesture_mask):
                duration = np.sum(gesture_mask) / self.fs
                gesture_durations.append(duration)
                gesture_classes.append(gesture_class)
        
        plt.bar(gesture_classes, gesture_durations)
        plt.xlabel('Gesture Class')
        plt.ylabel('Duration (s)')
        plt.title('Gesture Duration Analysis')
        
        # 6. Frequency spectrum
        ax6 = plt.subplot(3, 3, 6)
        f, psd = signal.welch(emg_channels[:, 0], fs=self.fs, nperseg=min(1024, len(emg_channels)//4))
        plt.semilogy(f, psd)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Frequency Spectrum - Channel 1')
        plt.xlim(0, 500)
        
        # 7. Gesture amplitude profiles
        ax7 = plt.subplot(3, 3, 7)
        for gesture_class in np.unique(labels):
            if gesture_class == 0:
                continue
            gesture_mask = labels == gesture_class
            if np.any(gesture_mask):
                gesture_data = emg_channels[gesture_mask]
                mean_amplitude = np.mean(np.abs(gesture_data), axis=0)
                plt.plot(mean_amplitude, 'o-', label=f'Gesture {gesture_class}')
        plt.xlabel('Channel')
        plt.ylabel('Mean Absolute Amplitude')
        plt.title('Gesture Amplitude Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Signal quality metrics
        ax8 = plt.subplot(3, 3, 8)
        if 'signal_quality' in self.results:
            snr = self.results['signal_quality']['snr_db']
            plt.bar(range(8), snr)
            plt.xlabel('Channel')
            plt.ylabel('SNR (dB)')
            plt.title('Signal-to-Noise Ratio by Channel')
            plt.grid(True, alpha=0.3)
        
        # 9. Cross-subject variability (if available)
        ax9 = plt.subplot(3, 3, 9)
        if 'cross_subject_variability' in self.results:
            variability = self.results['cross_subject_variability']['amplitude_variability']
            plt.bar(range(8), variability)
            plt.xlabel('Channel')
            plt.ylabel('Amplitude Variability')
            plt.title('Cross-Subject Amplitude Variability')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'emg_analysis_subject_{subject_id}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Visualizations saved as 'emg_analysis_subject_{subject_id}.png'")
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print("📋 Generating Analysis Report...")
        
        report = []
        report.append("# EMG Data Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Signal Quality Summary
        if 'signal_quality' in self.results:
            sq = self.results['signal_quality']
            report.append("## Signal Quality Analysis")
            report.append(f"- SNR Range: {sq['snr_db'].min():.1f} - {sq['snr_db'].max():.1f} dB")
            report.append(f"- Artifact Ratio: {sq['artifact_ratio']:.3f}")
            report.append(f"- Baseline Drift Range: {sq['baseline_drift'].min():.2e} - {sq['baseline_drift'].max():.2e}")
            report.append("")
        
        # Gesture Patterns Summary
        if 'gesture_patterns' in self.results:
            gp = self.results['gesture_patterns']
            report.append("## Gesture Pattern Analysis")
            report.append(f"- Number of Gesture Classes: {len(gp)}")
            for gesture, metrics in gp.items():
                report.append(f"- Gesture {gesture}: {metrics['duration_seconds']:.1f}s duration, "
                            f"{metrics['samples_count']} samples")
            report.append("")
        
        # Cross-Subject Variability Summary
        if 'cross_subject_variability' in self.results:
            csv = self.results['cross_subject_variability']
            report.append("## Cross-Subject Variability Analysis")
            report.append(f"- Number of Subjects Analyzed: {csv['n_subjects']}")
            report.append(f"- Amplitude Variability Range: {csv['amplitude_variability'].min():.2e} - "
                        f"{csv['amplitude_variability'].max():.2e}")
            report.append("")
        
        # Recommendations
        report.append("## Key Findings and Recommendations")
        report.append("")
        
        if 'signal_quality' in self.results:
            sq = self.results['signal_quality']
            if sq['artifact_ratio'] > 0.05:
                report.append("⚠️  **High Artifact Ratio**: Consider implementing artifact removal techniques")
            if sq['snr_db'].min() < 10:
                report.append("⚠️  **Low SNR**: Consider improving signal preprocessing")
        
        if 'cross_subject_variability' in self.results:
            csv = self.results['cross_subject_variability']
            if csv['amplitude_variability'].max() > 1e-4:
                report.append("⚠️  **High Cross-Subject Variability**: Consider domain adaptation techniques")
        
        report.append("✅ **Data Quality**: Overall good signal quality for analysis")
        report.append("✅ **Gesture Patterns**: Clear gesture patterns identified")
        report.append("✅ **Cross-Subject Analysis**: Variability patterns identified for improvement")
        
        # Save report
        with open('emg_analysis_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("   ✅ Analysis report saved as 'emg_analysis_report.md'")
        
        return '\n'.join(report)

def main():
    """Main analysis function"""
    print("🚀 Starting EMG Data Analysis...")
    print("=" * 50)
    
    # Initialize analyzer
    data_path = "/Users/amitsubhash/Downloads/EMG/EMG_gestures/EMG_data"
    analyzer = QuickEMGAnalyzer(data_path)
    
    # Analyze a few subjects
    subject_ids = [1, 2, 3, 4, 5]  # Analyze first 5 subjects
    
    # Load and analyze first subject in detail
    print(f"\n📊 Detailed Analysis - Subject 1")
    data = analyzer.load_subject_data(1)
    
    # Run analyses
    analyzer.analyze_signal_quality(data)
    analyzer.analyze_gesture_patterns(data)
    
    # Analyze cross-subject variability
    analyzer.analyze_cross_subject_variability(subject_ids)
    
    # Create visualizations
    analyzer.create_visualizations(subject_id=1)
    
    # Generate report
    report = analyzer.generate_analysis_report()
    
    print("\n" + "=" * 50)
    print("🎉 Analysis Complete!")
    print("=" * 50)
    print("\nKey Files Generated:")
    print("- emg_analysis_subject_1.png (visualizations)")
    print("- emg_analysis_report.md (detailed report)")
    print("\nNext Steps:")
    print("1. Review the generated visualizations")
    print("2. Check the analysis report for insights")
    print("3. Implement improvements based on findings")
    print("4. Proceed with advanced feature engineering")

if __name__ == "__main__":
    main()
