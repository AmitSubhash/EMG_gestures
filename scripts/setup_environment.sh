#!/bin/bash
# EMG Environment Setup Script
# Run this to set up the environment before submitting jobs

echo "🚀 Setting up EMG Analysis Environment"
echo "======================================"

# Load Python module
echo "📦 Loading Python module..."
module load python/3.12.4

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install packages
echo "📦 Installing required packages..."
pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn PyWavelets

echo "✅ Environment setup complete!"
echo "📁 Virtual environment: $(pwd)/venv"
echo "🐍 Python version: $(python3 --version)"
echo "📦 Installed packages:"
pip list | grep -E "(numpy|pandas|matplotlib|scipy|scikit-learn|PyWavelets)"

echo ""
echo "🚀 Ready to submit jobs!"
echo "   cd jobs && sbatch emg_analysis_job.slurm"
