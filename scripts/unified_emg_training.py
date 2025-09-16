#!/usr/bin/env python3
"""
Unified EMG Training Pipeline
Trains all three state-of-the-art architectures with proper evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from emg_data_preprocessing import EMGDataPreprocessor
from vision_transformer_emg import VisionTransformerEMG, EMGVisionTransformerTrainer
from attention_cnn_lstm_emg import AttentionCNNLSTM, EMGAttentionCNNLSTMTrainer
from domain_adaptive_transformer_emg import DomainAdaptiveTransformer, DomainAdaptiveTrainer

class UnifiedEMGTrainer:
    """Unified trainer for all three architectures"""
    
    def __init__(self, data_path, device='cpu'):
        self.data_path = Path(data_path)
        self.device = device
        self.preprocessor = EMGDataPreprocessor(data_path)
        self.results = {}
        
    def prepare_data(self, subject_ids=[1, 2, 3, 4, 5], test_subject=1):
        """Prepare data for all architectures"""
        print("🔧 Preparing data for all architectures...")
        
        # Prepare single-subject data for ViT and CNN-LSTM
        print(f"📊 Preparing single-subject data (subject {test_subject})...")
        single_subject_results = self.preprocessor.preprocess_for_architecture(
            subject_id=test_subject, architecture='all'
        )
        
        # Prepare multi-subject data for Domain Adaptive Transformer
        print(f"📊 Preparing multi-subject data (subjects {subject_ids})...")
        multi_subject_data = []
        multi_subject_labels = []
        multi_subject_ids = []
        
        for subject_id in subject_ids:
            results = self.preprocessor.preprocess_for_architecture(
                subject_id=subject_id, architecture='all'
            )
            multi_subject_data.append(results['raw_data'])
            multi_subject_labels.append(results['labels'])
            multi_subject_ids.extend([subject_id] * len(results['labels']))
        
        # Concatenate multi-subject data
        multi_subject_emg = np.vstack(multi_subject_data)
        multi_subject_labels = np.hstack(multi_subject_labels)
        multi_subject_ids = np.array(multi_subject_ids)
        
        # Create windows for multi-subject data
        multi_windows, multi_window_labels = self.preprocessor.create_windows(
            multi_subject_emg, multi_subject_labels
        )
        
        # Create multi-subject dataset
        multi_subject_dataset = self._create_multi_subject_dataset(
            multi_windows, multi_window_labels, multi_subject_ids
        )
        
        self.data = {
            'single_subject': single_subject_results,
            'multi_subject': {
                'windows': multi_windows,
                'labels': multi_window_labels,
                'subject_ids': multi_subject_ids,
                'dataset': multi_subject_dataset
            }
        }
        
        print("   ✅ Data preparation complete")
        
        return self.data
    
    def _create_multi_subject_dataset(self, windows, labels, subject_ids):
        """Create multi-subject dataset for domain adaptation"""
        from domain_adaptive_transformer_emg import EMGMultiSubjectDataset
        
        # Create subject IDs for windows
        window_subject_ids = []
        window_idx = 0
        for i, subject_id in enumerate(subject_ids):
            subject_windows = np.sum(labels == (i + 1))  # Assuming labels are 1-6
            window_subject_ids.extend([subject_id] * subject_windows)
            window_idx += subject_windows
        
        return EMGMultiSubjectDataset(
            windows, labels, np.array(window_subject_ids), 
            window_size=1000, overlap=0.5
        )
    
    def train_vision_transformer(self, epochs=50):
        """Train Vision Transformer"""
        print("\n🚀 Training Vision Transformer...")
        print("=" * 50)
        
        # Prepare data
        stft_windows = self.data['single_subject']['stft_windows']
        window_labels = self.data['single_subject']['window_labels']
        
        # Create dataset
        from vision_transformer_emg import EMGSTFTDataset
        dataset = EMGSTFTDataset(
            self.data['single_subject']['raw_data'], 
            self.data['single_subject']['labels'],
            window_size=1000, overlap=0.5
        )
        
        # Split data
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Create model
        model = VisionTransformerEMG(
            img_size=(8, 33, 63),
            patch_size=(1, 4, 4),
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            num_classes=6,
            dropout=0.1
        )
        
        # Train model
        trainer = EMGVisionTransformerTrainer(model, self.device)
        results = trainer.train(train_loader, val_loader, epochs=epochs)
        
        # Evaluate
        test_loss, test_acc, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
        
        self.results['vision_transformer'] = {
            'model': model,
            'trainer': trainer,
            'results': results,
            'test_metrics': {
                'loss': test_loss,
                'accuracy': test_acc,
                'f1_score': test_f1
            },
            'predictions': test_preds,
            'targets': test_targets
        }
        
        print(f"✅ Vision Transformer - F1: {test_f1:.4f}, Acc: {test_acc:.2f}%")
        
        return self.results['vision_transformer']
    
    def train_attention_cnn_lstm(self, epochs=50):
        """Train Attention CNN-LSTM"""
        print("\n🚀 Training Attention CNN-LSTM...")
        print("=" * 50)
        
        # Prepare data
        windows = self.data['single_subject']['windows']
        window_labels = self.data['single_subject']['window_labels']
        
        # Create dataset
        from attention_cnn_lstm_emg import EMGWindowDataset
        dataset = EMGWindowDataset(
            self.data['single_subject']['raw_data'],
            self.data['single_subject']['labels'],
            window_size=1000, overlap=0.5
        )
        
        # Split data
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = AttentionCNNLSTM(
            n_channels=8,
            window_size=1000,
            hidden_dim=128,
            num_lstm_layers=1,
            num_heads=8,
            num_classes=6,
            dropout=0.1
        )
        
        # Train model
        trainer = EMGAttentionCNNLSTMTrainer(model, self.device)
        results = trainer.train(train_loader, val_loader, epochs=epochs)
        
        # Evaluate
        test_loss, test_acc, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
        
        self.results['attention_cnn_lstm'] = {
            'model': model,
            'trainer': trainer,
            'results': results,
            'test_metrics': {
                'loss': test_loss,
                'accuracy': test_acc,
                'f1_score': test_f1
            },
            'predictions': test_preds,
            'targets': test_targets
        }
        
        print(f"✅ Attention CNN-LSTM - F1: {test_f1:.4f}, Acc: {test_acc:.2f}%")
        
        return self.results['attention_cnn_lstm']
    
    def train_domain_adaptive_transformer(self, epochs=50):
        """Train Domain Adaptive Transformer"""
        print("\n🚀 Training Domain Adaptive Transformer...")
        print("=" * 50)
        
        # Prepare data
        dataset = self.data['multi_subject']['dataset']
        
        # Split data
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = DomainAdaptiveTransformer(
            n_channels=8,
            window_size=1000,
            d_model=128,
            num_heads=8,
            num_layers=4,
            d_ff=512,
            num_classes=6,
            num_domains=5,  # Number of subjects
            dropout=0.1
        )
        
        # Train model
        trainer = DomainAdaptiveTrainer(model, self.device, lambda_domain=0.1)
        results = trainer.train(train_loader, val_loader, epochs=epochs)
        
        # Evaluate
        test_loss, test_acc, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
        
        self.results['domain_adaptive_transformer'] = {
            'model': model,
            'trainer': trainer,
            'results': results,
            'test_metrics': {
                'loss': test_loss,
                'accuracy': test_acc,
                'f1_score': test_f1
            },
            'predictions': test_preds,
            'targets': test_targets
        }
        
        print(f"✅ Domain Adaptive Transformer - F1: {test_f1:.4f}, Acc: {test_acc:.2f}%")
        
        return self.results['domain_adaptive_transformer']
    
    def compare_architectures(self):
        """Compare all three architectures"""
        print("\n📊 Comparing Architectures...")
        print("=" * 50)
        
        # Create comparison table
        comparison_data = []
        for arch_name, arch_results in self.results.items():
            metrics = arch_results['test_metrics']
            comparison_data.append({
                'Architecture': arch_name.replace('_', ' ').title(),
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.2f}%",
                'Loss': f"{metrics['loss']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. F1-Score comparison
        ax1 = axes[0, 0]
        architectures = [arch.replace('_', ' ').title() for arch in self.results.keys()]
        f1_scores = [arch_results['test_metrics']['f1_score'] for arch_results in self.results.values()]
        bars = ax1.bar(architectures, f1_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_ylabel('F1-Score')
        ax1.set_title('F1-Score Comparison')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. Accuracy comparison
        ax2 = axes[0, 1]
        accuracies = [arch_results['test_metrics']['accuracy'] for arch_results in self.results.values()]
        bars = ax2.bar(architectures, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Comparison')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # 3. Training curves comparison
        ax3 = axes[1, 0]
        for arch_name, arch_results in self.results.items():
            val_f1_scores = arch_results['results']['val_f1_scores']
            ax3.plot(val_f1_scores, label=arch_name.replace('_', ' ').title(), linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation F1-Score')
        ax3.set_title('Training Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Confusion matrices
        ax4 = axes[1, 1]
        # Create a combined confusion matrix
        all_targets = []
        all_preds = []
        arch_names = []
        
        for arch_name, arch_results in self.results.items():
            targets = arch_results['targets']
            preds = arch_results['predictions']
            all_targets.extend(targets)
            all_preds.extend(preds)
            arch_names.extend([arch_name] * len(targets))
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_title('Combined Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Comparison visualization saved as 'architecture_comparison.png'")
        
        return comparison_df
    
    def save_results(self, filename='emg_training_results.json'):
        """Save training results"""
        print(f"💾 Saving results to {filename}...")
        
        # Prepare results for JSON serialization
        json_results = {}
        for arch_name, arch_results in self.results.items():
            json_results[arch_name] = {
                'test_metrics': arch_results['test_metrics'],
                'training_history': {
                    'train_losses': arch_results['results']['train_losses'],
                    'val_losses': arch_results['results']['val_losses'],
                    'val_accuracies': arch_results['results']['val_accuracies'],
                    'val_f1_scores': arch_results['results']['val_f1_scores'],
                    'best_f1': arch_results['results']['best_f1']
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"   ✅ Results saved to {filename}")
    
    def run_full_training(self, subject_ids=[1, 2, 3, 4, 5], test_subject=1, epochs=50):
        """Run full training pipeline for all architectures"""
        print("🚀 Starting Full EMG Training Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Prepare data
        self.prepare_data(subject_ids, test_subject)
        
        # Train all architectures
        print(f"\n🎯 Training all architectures for {epochs} epochs each...")
        
        # Vision Transformer
        self.train_vision_transformer(epochs=epochs)
        
        # Attention CNN-LSTM
        self.train_attention_cnn_lstm(epochs=epochs)
        
        # Domain Adaptive Transformer
        self.train_domain_adaptive_transformer(epochs=epochs)
        
        # Compare architectures
        comparison_df = self.compare_architectures()
        
        # Save results
        self.save_results()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 Training Complete!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📊 Best performing architecture:")
        
        # Find best architecture
        best_arch = max(self.results.keys(), 
                       key=lambda x: self.results[x]['test_metrics']['f1_score'])
        best_f1 = self.results[best_arch]['test_metrics']['f1_score']
        best_acc = self.results[best_arch]['test_metrics']['accuracy']
        
        print(f"   🏆 {best_arch.replace('_', ' ').title()}")
        print(f"   📈 F1-Score: {best_f1:.4f}")
        print(f"   📈 Accuracy: {best_acc:.2f}%")
        
        return self.results, comparison_df

def main():
    """Main function to run unified training"""
    print("🚀 Unified EMG Training Pipeline")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Initialize trainer
    data_path = "/Users/amitsubhash/Downloads/EMG/EMG_gestures/EMG_data"
    trainer = UnifiedEMGTrainer(data_path, device)
    
    # Run full training
    results, comparison = trainer.run_full_training(
        subject_ids=[1, 2, 3, 4, 5],  # Multi-subject data
        test_subject=1,  # Single-subject data
        epochs=30  # Reduced for demo
    )
    
    print("\n🎯 Training Summary:")
    print("✅ All three architectures trained successfully")
    print("✅ Performance comparison completed")
    print("✅ Results saved to JSON")
    print("✅ Visualizations generated")

if __name__ == "__main__":
    main()


