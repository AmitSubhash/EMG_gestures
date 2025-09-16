#!/bin/bash
#SBATCH --job-name=emg_analysis               # Job name
#SBATCH --output=../logs/emg_analysis_%j.out  # Standard output file (with job id in file name)
#SBATCH --error=../logs/emg_analysis_%j.err   # Standard error file (with job id in file name)
#SBATCH --mail-type=ALL                       # Email notifications for all job events
#SBATCH --mail-user=atsubhas@iu.edu           # Email address for notifications
#SBATCH -p gpu                        # Partition name
#SBATCH -A r00602                             # Project/account number
#SBATCH --mem=8G                              # Memory per node
#SBATCH --time=02:00:00                       # Maximum walltime (hh:mm:ss)
#SBATCH --cpus-per-task=4                     # Number of CPU cores per task

# -------------------- Module Loads and Environment Setup --------------------

# Load required modules
module load python/3.12.4

# Change to scripts directory first
cd "$(dirname "$0")/.."

# Activate existing virtual environment
echo "🔧 Activating existing virtual environment..."
source venv/bin/activate

# Verify packages are available
echo "📦 Verifying packages..."
python3 -c "import numpy, pandas, matplotlib, scipy, sklearn, pywt; print('✅ All packages available')"

# -------------------- Change Directory and Run Python Scripts --------------------

# Change directory to where your Python scripts are located
cd "$(dirname "$0")/.."

# Run the Python scripts
python3 emg_data_tuned_analysis.py
python3 emg_tuned_feature_extraction.py
python3 emg_data_preprocessing.py

# Organize outputs with timestamp
echo "📁 Organizing outputs..."
python3 organize_outputs.py

echo "✅ Analysis pipeline complete!"
echo "📁 Check the organized results in results/run_*/"
echo "Job finished at: $(date)"
