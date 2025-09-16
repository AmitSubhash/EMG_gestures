#!/bin/bash
#SBATCH --job-name=emg_cnn_lstm               # Job name
#SBATCH --output=../logs/emg_cnn_lstm_%j.out  # Standard output file (with job id in file name)
#SBATCH --error=../logs/emg_cnn_lstm_%j.err   # Standard error file (with job id in file name)
#SBATCH --mail-type=ALL                       # Email notifications for all job events
#SBATCH --mail-user=atsubhas@iu.edu           # Email address for notifications
#SBATCH -p gpu                                # Partition name
#SBATCH --gpus-per-node=1                     # Request one GPU per node
#SBATCH -A r00602                             # Project/account number
#SBATCH --mem=8G                              # Memory per node
#SBATCH --time=03:30:00                       # Maximum walltime (hh:mm:ss)
#SBATCH --cpus-per-task=4                     # Number of CPU cores per task

# -------------------- Module Loads and Environment Setup --------------------

# Load required modules
module load python/gpu/3.11.5
module load cudatoolkit/11.7

# Change to scripts directory first
cd "$(dirname "$0")/.."
echo "📁 Current directory: $(pwd)"

# Activate existing virtual environment
echo "🔧 Activating existing virtual environment..."
source venv/bin/activate

# Verify packages are available
echo "📦 Verifying packages..."
python3 -c "import torch, numpy, pandas, matplotlib, scipy, sklearn, pywt; print('✅ All packages available')"

# -------------------- Run Python Script --------------------

# Run the Python script with full path
echo "🚀 Running Attention CNN-LSTM..."
python3 scripts/attention_cnn_lstm_emg.py

# Organize outputs with timestamp
echo "📁 Organizing outputs..."
python3 scripts/organize_outputs.py

echo "✅ Attention CNN-LSTM training complete!"
echo "📊 Check the results and model files"
echo "Job finished at: $(date)"
