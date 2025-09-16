#!/bin/bash
# One-time virtual environment setup
# Run this once to create and configure the virtual environment

echo "🚀 Setting up EMG Virtual Environment (One-time setup)"
echo "====================================================="

# Load Python module
echo "📦 Loading Python module..."
module load python/3.12.4

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install all required packages
echo "📦 Installing packages..."
pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn PyWavelets
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify installation
echo "✅ Verifying installation..."
python3 -c "import torch, numpy, pandas, matplotlib, scipy, sklearn, pywt; print('✅ All packages installed successfully!')"

echo ""
echo "🎉 Virtual environment setup complete!"
echo "📁 Location: $(pwd)/venv"
echo "🐍 Python: $(python3 --version)"
echo "🔥 PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""
echo "🚀 Ready to submit jobs!"
echo "   cd jobs && sbatch emg_analysis_job.slurm"
