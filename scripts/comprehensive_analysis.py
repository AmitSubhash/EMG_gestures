#!/usr/bin/env python3
"""
Comprehensive EMG Analysis Pipeline
Complete analysis from data loading to model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from quick_data_analysis import QuickEMGAnalyzer
from modern_feature_engineering import ModernEMGFeatureExtractor, FeatureSelector
from modern_models import ModernEMGModels, ModelComparison

class ComprehensiveEMGAnalysis:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.analyzer = QuickEMGAnalyzer(data_path)
        self.feature_extractor = ModernEMGFeatureExtractor()
        self.feature_selector = FeatureSelector()
        self.results = {}
        
    def run_complete_analysis(self, subject_ids=None, test_subjects=None):
        """Run complete analysis pipeline"""
        print("🚀 Starting Comprehensive EMG Analysis")
        print("=" * 60)
        
        if subject_ids is None:
            subject_ids = list(range(1, 6))  # First 5 subjects
        
        if test_subjects is None:
            test_subjects = [1, 2]  # Test on subjects 1 and 2
        
        # Phase 1: Data Quality Analysis
        print("\n📊 Phase 1: Data Quality Analysis")
        print("-" * 40)
        self._analyze_data_quality(subject_ids)
        
        # Phase 2: Feature Engineering
        print("\n🔧 Phase 2: Feature Engineering")
        print("-" * 40)
        self._extract_and_select_features(subject_ids)
        
        # Phase 3: Model Development
        print("\n🧠 Phase 3: Model Development")
        print("-" * 40)
        self._develop_models()
        
        # Phase 4: Cross-Subject Evaluation
        print("\n👥 Phase 4: Cross-Subject Evaluation")
        print("-" * 40)
        self._evaluate_cross_subject_performance(subject_ids, test_subjects)
        
        # Phase 5: Results Summary
        print("\n📋 Phase 5: Results Summary")
        print("-" * 40)
        self._generate_final_report()
        
        return self.results
    
    def _analyze_data_quality(self, subject_ids):
        """Analyze data quality across subjects"""
        print("   🔍 Analyzing data quality...")
        
        quality_results = {}
        
        for subject_id in subject_ids:
            try:
                data = self.analyzer.load_subject_data(subject_id)
                quality_metrics = self.analyzer.analyze_signal_quality(data)
                quality_results[subject_id] = quality_metrics
            except Exception as e:
                print(f"   ⚠️  Could not analyze subject {subject_id}: {e}")
                continue
        
        self.results['data_quality'] = quality_results
        print(f"   ✅ Analyzed {len(quality_results)} subjects")
        
        # Create quality summary
        if quality_results:
            snr_values = [metrics['snr_db'] for metrics in quality_results.values()]
            artifact_ratios = [metrics['artifact_ratio'] for metrics in quality_results.values()]
            
            print(f"   📈 SNR Range: {np.min(snr_values):.1f} - {np.max(snr_values):.1f} dB")
            print(f"   📈 Artifact Ratio Range: {np.min(artifact_ratios):.3f} - {np.max(artifact_ratios):.3f}")
    
    def _extract_and_select_features(self, subject_ids):
        """Extract and select features"""
        print("   🔧 Extracting features...")
        
        all_features = []
        all_labels = []
        subject_info = []
        
        for subject_id in subject_ids:
            try:
                data = self.analyzer.load_subject_data(subject_id)
                emg_channels = data.iloc[:, 1:9].values
                labels = data['class'].values
                
                # Extract features
                feature_matrix, feature_names = self.feature_extractor.extract_all_features(emg_channels, labels)
                
                # Filter out unmarked data (class 0)
                valid_mask = labels != 0
                if np.any(valid_mask):
                    valid_features = feature_matrix[valid_mask]
                    valid_labels = labels[valid_mask]
                    
                    all_features.append(valid_features)
                    all_labels.append(valid_labels)
                    subject_info.extend([subject_id] * len(valid_features))
                
            except Exception as e:
                print(f"   ⚠️  Could not extract features for subject {subject_id}: {e}")
                continue
        
        if not all_features:
            print("   ❌ No features extracted!")
            return
        
        # Combine all features
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        subject_info = np.array(subject_info)
        
        print(f"   ✅ Extracted {X.shape[1]} features from {X.shape[0]} samples")
        
        # Feature selection
        print("   🔍 Selecting best features...")
        selected_indices, scores = self.feature_selector.select_features(
            X, y, method='mutual_info', n_features=min(50, X.shape[1])
        )
        
        # Select features
        X_selected = X[:, selected_indices]
        selected_feature_names = [self.feature_extractor.feature_names[i] for i in selected_indices]
        
        self.results['features'] = {
            'X': X_selected,
            'y': y,
            'subject_info': subject_info,
            'feature_names': selected_feature_names,
            'selected_indices': selected_indices,
            'feature_scores': scores
        }
        
        print(f"   ✅ Selected {len(selected_indices)} best features")
    
    def _develop_models(self):
        """Develop and train models"""
        print("   🧠 Developing models...")
        
        if 'features' not in self.results:
            print("   ❌ No features available for model development!")
            return
        
        X = self.results['features']['X']
        y = self.results['features']['y']
        subject_info = self.results['features']['subject_info']
        
        # Convert labels to categorical
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        y_categorical = pd.get_dummies(y).values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
        )
        
        # Initialize model builder
        model_builder = ModernEMGModels(
            input_shape=(X.shape[1],),
            n_classes=n_classes
        )
        
        # Create models
        print("     Creating CNN-LSTM model...")
        cnn_lstm = model_builder.create_cnn_lstm_model()
        
        print("     Creating Transformer model...")
        transformer = model_builder.create_transformer_model()
        
        # Train models
        print("     Training CNN-LSTM...")
        history_cnn_lstm = model_builder.train_model(
            'cnn_lstm', X_train, y_train, X_val, y_val, epochs=20, verbose=0
        )
        
        print("     Training Transformer...")
        history_transformer = model_builder.train_model(
            'transformer', X_train, y_train, X_val, y_val, epochs=20, verbose=0
        )
        
        # Evaluate models
        print("     Evaluating models...")
        cnn_lstm_results = model_builder.evaluate_model('cnn_lstm', X_test, y_test)
        transformer_results = model_builder.evaluate_model('transformer', X_test, y_test)
        
        self.results['models'] = {
            'model_builder': model_builder,
            'cnn_lstm_results': cnn_lstm_results,
            'transformer_results': transformer_results,
            'history_cnn_lstm': history_cnn_lstm,
            'history_transformer': history_transformer,
            'n_classes': n_classes,
            'class_names': [f'Class {i}' for i in unique_labels]
        }
        
        print(f"     ✅ CNN-LSTM Accuracy: {cnn_lstm_results['accuracy']:.4f}")
        print(f"     ✅ Transformer Accuracy: {transformer_results['accuracy']:.4f}")
    
    def _evaluate_cross_subject_performance(self, subject_ids, test_subjects):
        """Evaluate cross-subject performance"""
        print("   👥 Evaluating cross-subject performance...")
        
        if 'models' not in self.results:
            print("   ❌ No models available for evaluation!")
            return
        
        model_builder = self.results['models']['model_builder']
        cross_subject_results = {}
        
        for test_subject in test_subjects:
            print(f"     Testing on subject {test_subject}...")
            
            try:
                # Load test subject data
                data = self.analyzer.load_subject_data(test_subject)
                emg_channels = data.iloc[:, 1:9].values
                labels = data['class'].values
                
                # Extract features
                feature_matrix, _ = self.feature_extractor.extract_all_features(emg_channels, labels)
                
                # Select same features
                selected_indices = self.results['features']['selected_indices']
                X_test = feature_matrix[:, selected_indices]
                
                # Filter valid data
                valid_mask = labels != 0
                X_test = X_test[valid_mask]
                y_test = labels[valid_mask]
                
                if len(X_test) == 0:
                    print(f"     ⚠️  No valid data for subject {test_subject}")
                    continue
                
                # Convert labels
                unique_labels = np.unique(y_test)
                y_test_categorical = pd.get_dummies(y_test).values
                
                # Evaluate models
                subject_results = {}
                
                for model_name in ['cnn_lstm', 'transformer']:
                    try:
                        results = model_builder.evaluate_model(model_name, X_test, y_test_categorical)
                        subject_results[model_name] = {
                            'accuracy': results['accuracy'],
                            'f1_macro': results['classification_report']['macro avg']['f1-score'],
                            'f1_weighted': results['classification_report']['weighted avg']['f1-score']
                        }
                    except Exception as e:
                        print(f"     ⚠️  Error evaluating {model_name} on subject {test_subject}: {e}")
                        continue
                
                cross_subject_results[test_subject] = subject_results
                
                # Print results
                for model_name, metrics in subject_results.items():
                    print(f"       {model_name}: Acc={metrics['accuracy']:.3f}, "
                          f"F1_macro={metrics['f1_macro']:.3f}")
                
            except Exception as e:
                print(f"     ⚠️  Could not evaluate subject {test_subject}: {e}")
                continue
        
        self.results['cross_subject'] = cross_subject_results
        print(f"   ✅ Evaluated {len(cross_subject_results)} test subjects")
    
    def _generate_final_report(self):
        """Generate final analysis report"""
        print("   📋 Generating final report...")
        
        report = []
        report.append("# Comprehensive EMG Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Data Quality Summary
        if 'data_quality' in self.results:
            report.append("## Data Quality Analysis")
            quality_results = self.results['data_quality']
            
            snr_values = [metrics['snr_db'] for metrics in quality_results.values()]
            artifact_ratios = [metrics['artifact_ratio'] for metrics in quality_results.values()]
            
            report.append(f"- **Subjects Analyzed**: {len(quality_results)}")
            report.append(f"- **SNR Range**: {np.min(snr_values):.1f} - {np.max(snr_values):.1f} dB")
            report.append(f"- **Artifact Ratio Range**: {np.min(artifact_ratios):.3f} - {np.max(artifact_ratios):.3f}")
            report.append("")
        
        # Feature Engineering Summary
        if 'features' in self.results:
            report.append("## Feature Engineering")
            features = self.results['features']
            
            report.append(f"- **Total Features Extracted**: {len(self.feature_extractor.feature_names)}")
            report.append(f"- **Selected Features**: {len(features['selected_indices'])}")
            report.append(f"- **Samples**: {features['X'].shape[0]}")
            report.append(f"- **Feature Score Range**: {features['feature_scores'].min():.3f} - {features['feature_scores'].max():.3f}")
            report.append("")
        
        # Model Performance Summary
        if 'models' in self.results:
            report.append("## Model Performance")
            models = self.results['models']
            
            cnn_lstm_acc = models['cnn_lstm_results']['accuracy']
            transformer_acc = models['transformer_results']['accuracy']
            
            report.append(f"- **CNN-LSTM Accuracy**: {cnn_lstm_acc:.4f}")
            report.append(f"- **Transformer Accuracy**: {transformer_acc:.4f}")
            report.append(f"- **Number of Classes**: {models['n_classes']}")
            report.append("")
        
        # Cross-Subject Performance Summary
        if 'cross_subject' in self.results:
            report.append("## Cross-Subject Performance")
            cross_subject = self.results['cross_subject']
            
            for subject_id, subject_results in cross_subject.items():
                report.append(f"### Subject {subject_id}")
                for model_name, metrics in subject_results.items():
                    report.append(f"- **{model_name}**: Accuracy={metrics['accuracy']:.3f}, "
                                f"F1_macro={metrics['f1_macro']:.3f}")
                report.append("")
        
        # Key Findings
        report.append("## Key Findings")
        report.append("")
        
        if 'cross_subject' in self.results:
            cross_subject = self.results['cross_subject']
            if cross_subject:
                # Calculate average cross-subject performance
                all_accuracies = []
                for subject_results in cross_subject.values():
                    for model_results in subject_results.values():
                        all_accuracies.append(model_results['accuracy'])
                
                avg_accuracy = np.mean(all_accuracies)
                report.append(f"- **Average Cross-Subject Accuracy**: {avg_accuracy:.3f}")
                
                if avg_accuracy < 0.5:
                    report.append("- ⚠️  **Low Cross-Subject Performance**: Consider domain adaptation techniques")
                elif avg_accuracy < 0.7:
                    report.append("- ⚠️  **Moderate Cross-Subject Performance**: Room for improvement")
                else:
                    report.append("- ✅ **Good Cross-Subject Performance**: Models generalize well")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        report.append("1. **Feature Engineering**: Continue exploring advanced features")
        report.append("2. **Model Architecture**: Experiment with ensemble methods")
        report.append("3. **Domain Adaptation**: Implement cross-subject adaptation techniques")
        report.append("4. **Data Augmentation**: Increase training data diversity")
        report.append("5. **Hyperparameter Tuning**: Optimize model parameters")
        
        # Save report
        with open('comprehensive_analysis_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("   ✅ Report saved as 'comprehensive_analysis_report.md'")
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print("\n📊 Key Results:")
        
        if 'models' in self.results:
            models = self.results['models']
            print(f"   • CNN-LSTM Accuracy: {models['cnn_lstm_results']['accuracy']:.3f}")
            print(f"   • Transformer Accuracy: {models['transformer_results']['accuracy']:.3f}")
        
        if 'cross_subject' in self.results and self.results['cross_subject']:
            cross_subject = self.results['cross_subject']
            all_accuracies = []
            for subject_results in cross_subject.values():
                for model_results in subject_results.values():
                    all_accuracies.append(model_results['accuracy'])
            avg_accuracy = np.mean(all_accuracies)
            print(f"   • Average Cross-Subject Accuracy: {avg_accuracy:.3f}")
        
        print(f"\n📁 Generated Files:")
        print(f"   • comprehensive_analysis_report.md")
        print(f"   • emg_analysis_subject_1.png")
        print(f"   • emg_analysis_report.md")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review the comprehensive analysis report")
        print(f"   2. Implement domain adaptation techniques")
        print(f"   3. Experiment with ensemble methods")
        print(f"   4. Optimize hyperparameters")
        print(f"   5. Deploy for real-time inference")

def main():
    """Run comprehensive analysis"""
    data_path = "../EMG_data"
    
    # Initialize analysis
    analysis = ComprehensiveEMGAnalysis(data_path)
    
    # Run complete analysis
    results = analysis.run_complete_analysis(
        subject_ids=[1, 2, 3, 4, 5],  # Analyze first 5 subjects
        test_subjects=[1, 2]  # Test on subjects 1 and 2
    )
    
    print("\n🎯 Analysis complete! Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
