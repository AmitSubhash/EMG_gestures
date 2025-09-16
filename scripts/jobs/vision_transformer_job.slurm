#!/bin/bash
#SBATCH --job-name=emg_vit
#SBATCH --account=r00602
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=../logs/emg_vit_%j.out
#SBATCH --error=../logs/emg_vit_%j.err

echo "🤖 Vision Transformer EMG Classification - Quartz System"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Time: $(date)"

# Change to scripts directory
cd "$(dirname "$0")/.."
echo "📁 Working directory: $(pwd)"

# Load required modules
echo "📦 Loading modules..."
module load python/3.9
module load gcc/9.3.0

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install PyTorch (CPU version for Quartz)
echo "📦 Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Run Vision Transformer
echo "🚀 Running Vision Transformer..."
python3 vision_transformer_emg.py

echo "✅ Vision Transformer training complete!"
echo "📊 Check the results and model files"
echo "Job finished at: $(date)"
