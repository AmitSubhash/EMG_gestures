#!/bin/bash
# Vision Transformer EMG Classification
# Run this after the preprocessing pipeline

echo "🤖 Vision Transformer EMG Classification"
echo "========================================"

# Load required modules
echo "📦 Loading required modules..."
module load python/3.9
module load gcc/9.3.0

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install PyTorch if not already installed
echo "📦 Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Run Vision Transformer
echo "🚀 Running Vision Transformer..."
python3 vision_transformer_emg.py

echo "✅ Vision Transformer training complete!"
echo "📊 Check the results and model files"
