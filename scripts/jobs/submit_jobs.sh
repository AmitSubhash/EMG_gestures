#!/bin/bash
# EMG Job Submission Script for Quartz
# This script helps you submit different types of EMG analysis jobs

echo "🚀 EMG Job Submission Helper - Quartz System"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "../emg_data_tuned_analysis.py" ]; then
    echo "❌ Error: Please run this script from the jobs directory"
    echo "   cd scripts/jobs && ./submit_jobs.sh"
    exit 1
fi

echo "Available job types:"
echo "1. Data Analysis Only (2 hours, 8GB RAM)"
echo "2. Vision Transformer (3.5 hours, 8GB RAM, 1 GPU)"
echo "3. Attention CNN-LSTM (3.5 hours, 8GB RAM, 1 GPU)"
echo "4. Domain Adaptive Transformer (3.5 hours, 8GB RAM, 1 GPU)"
echo "5. Interactive Session (for testing)"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "📊 Submitting data analysis job..."
        sbatch emg_analysis_job.slurm
        echo "✅ Job submitted! Check status with: squeue -u $USER"
        ;;
    2)
        echo "🤖 Submitting Vision Transformer job..."
        sbatch vision_transformer_job.slurm
        echo "✅ Job submitted! Check status with: squeue -u $USER"
        ;;
    3)
        echo "🧠 Submitting Attention CNN-LSTM job..."
        sbatch attention_cnn_lstm_job.slurm
        echo "✅ Job submitted! Check status with: squeue -u $USER"
        ;;
    4)
        echo "🌐 Submitting Domain Adaptive Transformer job..."
        sbatch domain_adaptive_job.slurm
        echo "✅ Job submitted! Check status with: squeue -u $USER"
        ;;
    5)
        echo "🖥️  Starting interactive session..."
        echo "   This will give you a shell to test commands interactively"
        srun -A r00602 --partition=normal --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=8G --time=02:00:00 --pty bash
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "📋 Useful commands:"
echo "   squeue -u $USER          # Check your job status"
echo "   scancel <job_id>         # Cancel a job"
echo "   cat <output_file>        # View job output"
echo "   ls -la *.out *.err       # List output files"
