#!/bin/bash
# EMG Analysis Pipeline for IU Quartz System
# This script runs the complete EMG analysis pipeline

echo "🚀 EMG Analysis Pipeline - IU Quartz System"
echo "=============================================="

# Load required modules (adjust based on what's available on Quartz)
echo "📦 Loading required modules..."
module load python/3.9
module load gcc/9.3.0

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "📦 Installing required packages..."
pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install PyWavelets

# Run data analysis
echo "📊 Running data analysis..."
python3 emg_data_tuned_analysis.py

# Run feature extraction
echo "🔧 Running feature extraction..."
python3 emg_tuned_feature_extraction.py

# Run preprocessing
echo "🔄 Running data preprocessing..."
python3 emg_data_preprocessing.py

echo "✅ Analysis pipeline complete!"
echo "📁 Check the generated files and visualizations"
