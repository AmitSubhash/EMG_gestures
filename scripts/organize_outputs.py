#!/usr/bin/env python3
"""
Output Organization Script
Automatically organizes EMG analysis outputs with timestamps
"""

import os
import shutil
import glob
from datetime import datetime
from pathlib import Path

def organize_outputs():
    """Organize outputs into structured folders with timestamps"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped directories
    results_dir = Path("results")
    timestamp_dir = results_dir / f"run_{timestamp}"
    
    # Create subdirectories
    subdirs = ["data_analysis", "vision_transformer", "attention_cnn_lstm", "domain_adaptive", "models", "logs"]
    for subdir in subdirs:
        (timestamp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created organized structure: {timestamp_dir}")
    
    # Move files to appropriate directories
    moves = [
        # Data analysis outputs
        ("emg_analysis_subject_*.png", "data_analysis"),
        ("emg_preprocessing_subject_*.png", "data_analysis"),
        ("emg_tuned_feature_extraction_*.png", "data_analysis"),
        
        # Vision Transformer outputs
        ("vision_transformer_emg_results.png", "vision_transformer"),
        ("best_vit_emg_model.pth", "models"),
        ("vit_training_history.json", "vision_transformer"),
        
        # CNN-LSTM outputs
        ("attention_cnn_lstm_results.png", "attention_cnn_lstm"),
        ("best_cnn_lstm_emg_model.pth", "models"),
        ("cnn_lstm_training_history.json", "attention_cnn_lstm"),
        
        # Domain Adaptive outputs
        ("domain_adaptive_transformer_results.png", "domain_adaptive"),
        ("best_domain_adaptive_emg_model.pth", "models"),
        ("domain_adaptive_training_history.json", "domain_adaptive"),
        
        # Log files
        ("../logs/*.out", "logs"),
        ("../logs/*.err", "logs"),
    ]
    
    for pattern, subdir in moves:
        files = glob.glob(pattern)
        for file in files:
            if os.path.exists(file):
                dest = timestamp_dir / subdir / os.path.basename(file)
                shutil.move(file, dest)
                print(f"   📄 Moved {file} → {dest}")
    
    # Create summary file
    summary_file = timestamp_dir / "run_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"EMG Analysis Run Summary\n")
        f.write(f"========================\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run ID: {timestamp}\n\n")
        f.write(f"Directory Structure:\n")
        for subdir in subdirs:
            files = list((timestamp_dir / subdir).glob("*"))
            f.write(f"  {subdir}/: {len(files)} files\n")
    
    print(f"📊 Created summary: {summary_file}")
    print(f"✅ Outputs organized in: {timestamp_dir}")
    
    return timestamp_dir

if __name__ == "__main__":
    organize_outputs()
