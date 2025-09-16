#!/usr/bin/env python3
"""
Vision Transformer for EMG Classification
Implements ViT with 2D STFT representation and patch-based processing
Tuned for 8-channel EMG data with 6 gesture classes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import stft
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

class EMGSTFTDataset(Dataset):
    """Dataset for EMG STFT representation"""
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
    
    def _compute_stft_2d(self, window):
        """Compute 2D STFT representation for all channels"""
        # window shape: (window_size, n_channels)
        stft_2d = []
        
        for ch in range(window.shape[1]):
            f, t, stft_ch = stft(window[:, ch], fs=self.fs, nperseg=64, noverlap=32)
            stft_2d.append(np.abs(stft_ch))
        
        # Stack channels: (n_channels, freq_bins, time_bins)
        stft_2d = np.stack(stft_2d, axis=0)
        
        # Normalize
        stft_2d = (stft_2d - np.mean(stft_2d)) / (np.std(stft_2d) + 1e-8)
        
        return stft_2d
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.window_labels[idx]
        
        # Compute 2D STFT
        stft_2d = self._compute_stft_2d(window)
        
        # Convert to tensor
        stft_tensor = torch.FloatTensor(stft_2d)
        label_tensor = torch.LongTensor([label - 1])  # Convert to 0-5 range
        
        return stft_tensor, label_tensor

class PatchEmbedding(nn.Module):
    """Patch embedding for 2D STFT representation"""
    def __init__(self, img_size=(8, 33, 63), patch_size=(1, 4, 4), in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
        # Convolutional patch embedding
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                             kernel_size=(1, patch_size[1], patch_size[2]), 
                             stride=(1, patch_size[1], patch_size[2]))
        
    def forward(self, x):
        # x: (batch_size, n_channels, freq_bins, time_bins)
        # Add channel dimension for conv3d
        x = x.unsqueeze(1)  # (batch_size, 1, n_channels, freq_bins, time_bins)
        
        # Apply convolution
        x = self.proj(x)  # (batch_size, embed_dim, n_channels, freq_bins//patch_size, time_bins//patch_size)
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
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

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x))
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformerEMG(nn.Module):
    """Vision Transformer for EMG Classification"""
    def __init__(self, img_size=(8, 33, 63), patch_size=(1, 4, 4), in_channels=1, 
                 embed_dim=256, num_heads=8, num_layers=6, num_classes=6, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Initialize with max expected size, will be adjusted dynamically
        self.max_patches = 1000  # Large enough for any reasonable input
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding (adjust size dynamically)
        n_patches = x.shape[1]
        if n_patches > self.pos_embed.shape[1]:
            # If we need more patches than expected, extend the positional embedding
            additional_patches = n_patches - self.pos_embed.shape[1]
            additional_pos_embed = torch.zeros(1, additional_patches, self.pos_embed.shape[2], 
                                            device=self.pos_embed.device, dtype=self.pos_embed.dtype)
            self.pos_embed = nn.Parameter(torch.cat([self.pos_embed, additional_pos_embed], dim=1))
        
        # Use only the needed portion of positional embedding
        pos_embed = self.pos_embed[:, :n_patches, :]
        x = x + pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Classification head
        cls_output = x[:, 0]  # CLS token
        logits = self.head(cls_output)
        
        return logits

class EMGVisionTransformerTrainer:
    """Trainer for Vision Transformer on EMG data"""
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
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
                torch.save(self.model.state_dict(), 'best_vit_emg_model.pth')
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
    """Load and preprocess EMG data for ViT"""
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
    """Create visualizations for ViT results"""
    print("📊 Creating ViT visualizations...")
    
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
    
    # 4. Attention visualization (simplified)
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, f'Best F1-Score: {results["best_f1"]:.4f}', 
             ha='center', va='center', fontsize=16, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Model Performance Summary')
    
    plt.tight_layout()
    plt.savefig('vision_transformer_emg_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Visualizations saved as 'vision_transformer_emg_results.png'")

def main():
    """Main function to run Vision Transformer on EMG data"""
    print("🚀 Vision Transformer for EMG Classification")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Load data from multiple subjects for better performance
    data_path = Path("../EMG_data")
    print("📊 Loading data from multiple subjects...")
    
    all_windows = []
    all_labels = []
    
    # Load data from first 3 subjects
    for subject_id in [1, 2, 3]:
        try:
            emg_data, labels = load_emg_data(data_path, subject_id=subject_id)
            dataset = EMGSTFTDataset(emg_data, labels, window_size=1000, overlap=0.5)
            all_windows.extend(dataset.windows)
            all_labels.extend(dataset.window_labels)
            print(f"   ✅ Subject {subject_id}: {len(dataset)} windows")
        except Exception as e:
            print(f"   ⚠️  Subject {subject_id}: {e}")
            continue
    
    # Create combined dataset
    dataset = EMGSTFTDataset(np.array(all_windows), np.array(all_labels), window_size=1000, overlap=0.5)
    print(f"📊 Total windows: {len(dataset)}")
    
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
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = VisionTransformerEMG(
        img_size=(8, 33, 63),  # 8 channels, ~33 freq bins, ~63 time bins
        patch_size=(1, 4, 4),
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        num_classes=6,
        dropout=0.1
    )
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer = EMGVisionTransformerTrainer(model, device)
    results = trainer.train(train_loader, val_loader, epochs=50)
    
    # Evaluate on test set
    print("\n📊 Final Test Evaluation:")
    test_loss, test_acc, test_f1, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Create visualizations
    create_visualizations(results, model, test_loader, device)
    
    print("\n🎉 Vision Transformer training complete!")
    print(f"✅ Best F1-Score: {results['best_f1']:.4f}")
    print("✅ Model saved as 'best_vit_emg_model.pth'")
    
    # Save training history
    import json
    with open('vit_training_history.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ Training history saved as 'vit_training_history.json'")
    
    # Organize outputs
    print("📁 Organizing outputs...")
    import subprocess
    subprocess.run(['python3', 'organize_outputs.py'], check=True)

if __name__ == "__main__":
    main()


