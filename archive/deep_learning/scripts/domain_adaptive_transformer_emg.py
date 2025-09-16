#!/usr/bin/env python3
"""
Domain Adaptive Transformer for EMG Classification
Implements adversarial domain adaptation for cross-subject generalization
Tuned for 8-channel EMG data with 6 gesture classes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EMGMultiSubjectDataset(Dataset):
    """Dataset for multi-subject EMG data with domain labels"""
    def __init__(self, emg_data, labels, subject_ids, window_size=1000, overlap=0.5, fs=1000):
        self.emg_data = emg_data
        self.labels = labels
        self.subject_ids = subject_ids
        self.window_size = window_size
        self.overlap = overlap
        self.fs = fs
        self.hop_length = int(window_size * (1 - overlap))
        
        # Create windows
        self.windows, self.window_labels, self.window_subjects = self._create_windows()
        
    def _create_windows(self):
        """Create overlapping windows from EMG data"""
        windows = []
        window_labels = []
        window_subjects = []
        
        for i in range(0, len(self.emg_data) - self.window_size + 1, self.hop_length):
            window = self.emg_data[i:i + self.window_size]
            
            # Only use windows with consistent labels (no transitions)
            window_label = self.labels[i:i + self.window_size]
            if len(np.unique(window_label)) == 1 and window_label[0] != 0:
                windows.append(window)
                window_labels.append(window_label[0])
                window_subjects.append(self.subject_ids[i])
        
        return np.array(windows), np.array(window_labels), np.array(window_subjects)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.window_labels[idx]
        subject = self.window_subjects[idx]
        
        # Normalize window
        window = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-8)
        
        # Convert to tensor
        window_tensor = torch.FloatTensor(window).T  # (n_channels, window_size)
        label_tensor = torch.LongTensor([label - 1])  # Convert to 0-5 range
        subject_tensor = torch.LongTensor([subject])
        
        return window_tensor, label_tensor, subject_tensor

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output

class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial training"""
    def __init__(self, input_dim, hidden_dim=128, num_domains=36):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_domains)
        )
        
    def forward(self, x):
        return self.network(x)

class DomainAdaptiveTransformer(nn.Module):
    """Domain Adaptive Transformer for EMG Classification"""
    def __init__(self, n_channels=8, window_size=1000, d_model=128, num_heads=8, 
                 num_layers=4, d_ff=512, num_classes=6, num_domains=36, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(n_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, window_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(d_model, d_model, num_domains)
        
    def forward(self, x, return_features=False):
        # x: (batch_size, n_channels, window_size)
        batch_size, n_channels, seq_len = x.size()
        
        # Transpose and project
        x = x.transpose(1, 2)  # (batch_size, window_size, n_channels)
        x = self.input_projection(x)  # (batch_size, window_size, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (window_size, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, window_size, d_model)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Global average pooling
        features = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_domain_logits(self, x):
        """Get domain classification logits"""
        _, features = self.forward(x, return_features=True)
        domain_logits = self.domain_discriminator(features)
        return domain_logits

class DomainAdaptiveTrainer:
    """Trainer for Domain Adaptive Transformer"""
    def __init__(self, model, device='cpu', lambda_domain=0.1):
        self.model = model.to(device)
        self.device = device
        self.lambda_domain = lambda_domain
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        
        # Optimizers
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.domain_optimizer = torch.optim.AdamW(model.domain_discriminator.parameters(), 
                                                lr=1e-4, weight_decay=0.01)
        
        # Schedulers
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.domain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.domain_optimizer, T_max=100)
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_class_loss = 0
        total_domain_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target, subject) in enumerate(dataloader):
            data, target, subject = data.to(self.device), target.to(self.device), subject.to(self.device)
            target = target.squeeze()
            subject = subject.squeeze()
            
            # Forward pass
            class_logits, features = self.model(data, return_features=True)
            domain_logits = self.model.get_domain_logits(data)
            
            # Classification loss
            class_loss = self.classification_criterion(class_logits, target)
            
            # Domain adversarial loss (gradient reversal)
            domain_loss = self.domain_criterion(domain_logits, subject)
            
            # Total loss
            total_loss_batch = class_loss - self.lambda_domain * domain_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Domain discriminator training
            domain_logits_detached = self.model.get_domain_logits(data.detach())
            domain_loss_detached = self.domain_criterion(domain_logits_detached, subject)
            
            self.domain_optimizer.zero_grad()
            domain_loss_detached.backward()
            self.domain_optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            
            pred = class_logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)}, '
                      f'Total Loss: {total_loss_batch.item():.4f}, '
                      f'Class Loss: {class_loss.item():.4f}, '
                      f'Domain Loss: {domain_loss.item():.4f}')
        
        return (total_loss / len(dataloader), 
                total_class_loss / len(dataloader), 
                total_domain_loss / len(dataloader), 
                100. * correct / total)
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target, subject in dataloader:
                data, target, subject = data.to(self.device), target.to(self.device), subject.to(self.device)
                target = target.squeeze()
                
                class_logits = self.model(data)
                loss = self.classification_criterion(class_logits, target)
                
                total_loss += loss.item()
                pred = class_logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return total_loss / len(dataloader), accuracy, f1, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs=50):
        """Train the model"""
        best_f1 = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_f1_scores = []
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss, class_loss, domain_loss, train_acc = self.train_epoch(train_loader)
            print(f'Train Loss: {train_loss:.4f}, Class Loss: {class_loss:.4f}, '
                  f'Domain Loss: {domain_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # Validate
            val_loss, val_acc, val_f1, _, _ = self.evaluate(val_loader)
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
            
            # Update learning rates
            self.scheduler.step()
            self.domain_scheduler.step()
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 'best_domain_adaptive_transformer_emg_model.pth')
                print(f'New best F1: {best_f1:.4f}')
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_f1_scores.append(val_f1)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_f1_scores': val_f1_scores,
            'best_f1': best_f1
        }

def load_multi_subject_data(data_path, subject_ids=[1, 2, 3, 4, 5]):
    """Load EMG data from multiple subjects"""
    print(f"📊 Loading EMG data from subjects: {subject_ids}")
    
    all_emg_data = []
    all_labels = []
    all_subject_ids = []
    
    for subject_id in subject_ids:
        subject_folder = data_path / f"{subject_id:02d}"
        files = list(subject_folder.glob("*.txt"))
        
        subject_data = []
        for file in files:
            df = pd.read_csv(file, sep='\t')
            subject_data.append(df)
        
        if subject_data:
            data = pd.concat(subject_data, ignore_index=True)
            
            # Extract EMG channels and labels
            emg_data = data.iloc[:, 1:9].values  # channels 1-8
            labels = data['class'].values
            
            # Filter out unmarked samples (class 0)
            valid_mask = labels != 0
            emg_data = emg_data[valid_mask]
            labels = labels[valid_mask]
            
            # Add subject ID
            subject_ids_array = np.full(len(emg_data), subject_id)
            
            all_emg_data.append(emg_data)
            all_labels.append(labels)
            all_subject_ids.append(subject_ids_array)
            
            print(f"   ✅ Subject {subject_id}: {len(emg_data)} samples")
    
    # Concatenate all data
    emg_data = np.vstack(all_emg_data)
    labels = np.hstack(all_labels)
    subject_ids = np.hstack(all_subject_ids)
    
    print(f"   ✅ Total: {len(emg_data)} samples from {len(subject_ids)} subjects")
    print(f"   ✅ Gesture classes: {np.unique(labels)}")
    
    return emg_data, labels, subject_ids

def create_visualizations(results, model, test_loader, device):
    """Create visualizations for Domain Adaptive Transformer results"""
    print("📊 Creating Domain Adaptive Transformer visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training curves
    ax1 = axes[0, 0]
    ax1.plot(results['train_losses'], label='Train Loss', color='blue')
    ax1.plot(results['val_losses'], label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(results['val_accuracies'], label='Val Accuracy', color='green')
    ax2.plot(results['val_f1_scores'], label='Val F1-Score', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion matrix
    ax3 = axes[1, 0]
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target, subject in test_loader:
            data, target, subject = data.to(device), target.to(device), subject.to(device)
            target = target.squeeze()
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_targets)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix')
    
    # 4. Model summary
    ax4 = axes[1, 1]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    ax4.text(0.5, 0.7, f'Total Parameters: {total_params:,}', ha='center', va='center', fontsize=14)
    ax4.text(0.5, 0.5, f'Trainable Parameters: {trainable_params:,}', ha='center', va='center', fontsize=14)
    ax4.text(0.5, 0.3, f'Best F1-Score: {results["best_f1"]:.4f}', ha='center', va='center', fontsize=14)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Model Summary')
    
    plt.tight_layout()
    plt.savefig('domain_adaptive_transformer_emg_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Visualizations saved as 'domain_adaptive_transformer_emg_results.png'")

def main():
    """Main function to run Domain Adaptive Transformer on EMG data"""
    print("🚀 Domain Adaptive Transformer for EMG Classification")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Load multi-subject data
    data_path = Path("../EMG_data")
    emg_data, labels, subject_ids = load_multi_subject_data(data_path, subject_ids=[1, 2, 3, 4, 5])
    
    # Create dataset
    dataset = EMGMultiSubjectDataset(emg_data, labels, subject_ids, window_size=1000, overlap=0.5)
    print(f"📊 Created {len(dataset)} windows")
    
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
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
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
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer = DomainAdaptiveTrainer(model, device, lambda_domain=0.1)
    results = trainer.train(train_loader, val_loader, epochs=50)
    
    # Evaluate on test set
    print("\n📊 Final Test Evaluation:")
    test_loss, test_acc, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Create visualizations
    create_visualizations(results, model, test_loader, device)
    
    print("\n🎉 Domain Adaptive Transformer training complete!")
    print(f"✅ Best F1-Score: {results['best_f1']:.4f}")
    print("✅ Model saved as 'best_domain_adaptive_transformer_emg_model.pth'")

if __name__ == "__main__":
    main()


