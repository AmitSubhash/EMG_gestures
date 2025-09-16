#!/bin/bash
#SBATCH --job-name=emg_vit                    # Job name
#SBATCH --output=../logs/emg_vit_%j.out       # Standard output file (with job id in file name)
#SBATCH --error=../logs/emg_vit_%j.err        # Standard error file (with job id in file name)
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
module load cuda/11.7

# Activate your virtual environment
source venv/bin/activate

# -------------------- Change Directory and Run Python Script --------------------

# Change directory to where your Python scripts are located
cd "$(dirname "$0")/.."

# Run the Python script
python3 vision_transformer_emg.py

echo "✅ Vision Transformer training complete!"
echo "📊 Check the results and model files"
echo "Job finished at: $(date)"
