#!/bin/bash
#SBATCH --job-name=emg_complete
#SBATCH --account=r00602
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=../logs/emg_complete_%j.out
#SBATCH --error=../logs/emg_complete_%j.err

echo "🚀 Complete EMG Analysis and Model Training Pipeline - Quartz System"
echo "===================================================================="
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

# Create and activate virtual environment
echo "🔧 Setting up environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install all required packages
echo "📦 Installing packages..."
pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn
pip install PyWavelets
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 1: Data Analysis
echo "📊 Step 1: Data Analysis"
echo "========================"
python3 emg_data_tuned_analysis.py

# Step 2: Feature Extraction
echo "🔧 Step 2: Feature Extraction"
echo "============================="
python3 emg_tuned_feature_extraction.py

# Step 3: Data Preprocessing
echo "🔄 Step 3: Data Preprocessing"
echo "============================"
python3 emg_data_preprocessing.py

# Step 4: Vision Transformer
echo "🤖 Step 4: Vision Transformer"
echo "============================"
python3 vision_transformer_emg.py

# Step 5: Attention CNN-LSTM
echo "🧠 Step 5: Attention CNN-LSTM"
echo "============================"
python3 attention_cnn_lstm_emg.py

# Step 6: Domain Adaptive Transformer
echo "🌐 Step 6: Domain Adaptive Transformer"
echo "====================================="
python3 domain_adaptive_transformer_emg.py

# Step 7: Unified Training and Comparison
echo "📊 Step 7: Model Comparison"
echo "=========================="
python3 unified_emg_training.py

echo "🎉 Complete pipeline finished!"
echo "📁 Check all generated files, models, and results"
echo "📊 Look for visualization files and performance metrics"
echo "Job finished at: $(date)"
