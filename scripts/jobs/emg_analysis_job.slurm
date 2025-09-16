#!/bin/bash
#SBATCH --job-name=emg_analysis
#SBATCH --account=r00602
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/emg_analysis_%j.out
#SBATCH --error=../logs/emg_analysis_%j.err

echo "🚀 EMG Analysis Job - Quartz System"
echo "===================================="
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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "📦 Installing packages..."
pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn
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
echo "Job finished at: $(date)"
