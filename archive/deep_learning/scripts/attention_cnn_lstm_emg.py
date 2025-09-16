#!/usr/bin/env python3
"""
Attention CNN-LSTM for EMG Classification
Implements lightweight CNN + bidirectional LSTM + multi-head attention
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

class EMGWindowDataset(Dataset):
    """Dataset for EMG windowed data"""
    def __init__(self, emg_data, labels, window_size=1000, overlap=0.5, fs=1000):
        self.emg_data = emg_data
        self.labels = labels
        self.window_size = window_size
        self.overlap = overlap
        self.fs = fs
        self.hop_length = int(window_size * (1 - overlap))
        
        # Create windows
        self.windows, self.window_labels = self._create_windows()
        
    def _create_windows(self):
        """Create overlapping windows from EMG data"""
        windows = []
        window_labels = []
        
        for i in range(0, len(self.emg_data) - self.window_size + 1, self.hop_length):
            window = self.emg_data[i:i + self.window_size]
            
            # Only use windows with consistent labels (no transitions)
            window_label = self.labels[i:i + self.window_size]
            if len(np.unique(window_label)) == 1 and window_label[0] != 0:
                windows.append(window)
                window_labels.append(window_label[0])
        
        return np.array(windows), np.array(window_labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.window_labels[idx]
        
        # Normalize window
        window = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-8)
        
        # Convert to tensor
        window_tensor = torch.FloatTensor(window).T  # (n_channels, window_size)
        label_tensor = torch.LongTensor([label - 1])  # Convert to 0-5 range
        
        return window_tensor, label_tensor

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, n_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch_size, n_channels, seq_len)
        b, c, s = x.size()
        
        # Global average pooling and max pooling
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # Channel attention
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        
        attention = self.sigmoid(avg_out + max_out)
        attention = attention.view(b, c, 1)
        
        return x * attention

class TemporalAttention(nn.Module):
    """Temporal attention mechanism"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class CNNBlock(nn.Module):
    """CNN block with channel attention"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.channel_attention = ChannelAttention(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.channel_attention(x)
        return x

class AttentionCNNLSTM(nn.Module):
    """Attention CNN-LSTM for EMG Classification"""
    def __init__(self, n_channels=8, window_size=1000, hidden_dim=128, 
                 num_lstm_layers=1, num_heads=8, num_classes=6, dropout=0.1):
        super().__init__()
        
        # CNN feature extraction
        self.cnn_blocks = nn.ModuleList([
            CNNBlock(n_channels, 32, kernel_size=3, stride=1),
            CNNBlock(32, 64, kernel_size=3, stride=2),
            CNNBlock(64, 128, kernel_size=3, stride=2)
        ])
        
        # Calculate LSTM input size after CNN
        # After 3 CNN blocks with strides [1, 2, 2], sequence length becomes window_size // 4
        lstm_input_size = 128
        lstm_seq_len = window_size // 4
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim * 2, num_heads, dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x: (batch_size, n_channels, window_size)
        batch_size = x.size(0)
        
        # CNN feature extraction
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
        
        # Reshape for LSTM: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Temporal attention
        attended_out = self.temporal_attention(lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attended_out, dim=1)  # (batch_size, hidden_dim * 2)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

class EMGAttentionCNNLSTMTrainer:
    """Trainer for Attention CNN-LSTM on EMG data"""
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            target = target.squeeze()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
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
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # Validate
            val_loss, val_acc, val_f1, _, _ = self.evaluate(val_loader)
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
            
            # Update learning rate
            self.scheduler.step()
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 'best_attention_cnn_lstm_emg_model.pth')
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

def load_emg_data(data_path, subject_id=1):
    """Load and preprocess EMG data"""
    print(f"📊 Loading EMG data for subject {subject_id}")
    
    # Load data
    subject_folder = data_path / f"{subject_id:02d}"
    files = list(subject_folder.glob("*.txt"))
    
    all_data = []
    for file in files:
        df = pd.read_csv(file, sep='\t')
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    
    # Extract EMG channels and labels
    emg_data = data.iloc[:, 1:9].values  # channels 1-8
    labels = data['class'].values
    
    # Filter out unmarked samples (class 0)
    valid_mask = labels != 0
    emg_data = emg_data[valid_mask]
    labels = labels[valid_mask]
    
    print(f"   ✅ Loaded {len(emg_data)} samples")
    print(f"   ✅ Gesture classes: {np.unique(labels)}")
    
    return emg_data, labels

def create_visualizations(results, model, test_loader, device):
    """Create visualizations for Attention CNN-LSTM results"""
    print("📊 Creating Attention CNN-LSTM visualizations...")
    
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
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
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
    
    # 4. Model architecture summary
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
    plt.savefig('attention_cnn_lstm_emg_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Visualizations saved as 'attention_cnn_lstm_emg_results.png'")

def main():
    """Main function to run Attention CNN-LSTM on EMG data"""
    print("🚀 Attention CNN-LSTM for EMG Classification")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Load data
    data_path = Path("../EMG_data")
    emg_data, labels = load_emg_data(data_path, subject_id=1)
    
    # Create dataset
    dataset = EMGWindowDataset(emg_data, labels, window_size=1000, overlap=0.5)
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
    model = AttentionCNNLSTM(
        n_channels=8,
        window_size=1000,
        hidden_dim=128,
        num_lstm_layers=1,
        num_heads=8,
        num_classes=6,
        dropout=0.1
    )
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer = EMGAttentionCNNLSTMTrainer(model, device)
    results = trainer.train(train_loader, val_loader, epochs=50)
    
    # Evaluate on test set
    print("\n📊 Final Test Evaluation:")
    test_loss, test_acc, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Create visualizations
    create_visualizations(results, model, test_loader, device)
    
    print("\n🎉 Attention CNN-LSTM training complete!")
    print(f"✅ Best F1-Score: {results['best_f1']:.4f}")
    print("✅ Model saved as 'best_attention_cnn_lstm_emg_model.pth'")

if __name__ == "__main__":
    main()


