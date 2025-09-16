# EMG Analysis Jobs - Quartz System

This directory contains Slurm job scripts for running EMG analysis on IU's Quartz system.

## Directory Structure
```
EMG_gestures/
├── scripts/           # Python analysis scripts
├── jobs/             # Slurm job files (this directory)
├── logs/             # Job output and error logs
├── results/          # Analysis results and visualizations
├── models/           # Trained model files
└── EMG_data/         # Raw EMG data
```

## Available Jobs

### 1. Data Analysis Only (`emg_analysis_job.slurm`)
- **Duration**: 2 hours
- **Resources**: 4 CPUs, 8GB RAM
- **Purpose**: Data exploration, feature extraction, preprocessing
- **Output**: Analysis reports and visualizations

### 2. Vision Transformer (`vision_transformer_job.slurm`)
- **Duration**: 3.5 hours
- **Resources**: 4 CPUs, 8GB RAM, 1 GPU
- **Purpose**: Train Vision Transformer on STFT spectrograms
- **Output**: Trained model and performance metrics

### 3. Attention CNN-LSTM (`attention_cnn_lstm_job.slurm`)
- **Duration**: 3.5 hours
- **Resources**: 4 CPUs, 8GB RAM, 1 GPU
- **Purpose**: Train Attention CNN-LSTM on time-series data
- **Output**: Trained model and performance metrics

### 4. Domain Adaptive Transformer (`domain_adaptive_job.slurm`)
- **Duration**: 3.5 hours
- **Resources**: 4 CPUs, 8GB RAM, 1 GPU
- **Purpose**: Train Domain Adaptive Transformer for cross-subject generalization
- **Output**: Trained model and performance metrics

## How to Submit Jobs

### Option 1: Use the submission helper
```bash
cd scripts/jobs
./submit_jobs.sh
```

### Option 2: Submit directly
```bash
cd scripts/jobs
sbatch emg_analysis_job.slurm
sbatch vision_transformer_job.slurm
sbatch complete_emg_pipeline_job.slurm
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job output
cat ../logs/emg_analysis_<job_id>.out

# View job errors
cat ../logs/emg_analysis_<job_id>.err

# Cancel a job
scancel <job_id>
```

## Account Information
- **Account**: r00602 (Mouse Optical Research)
- **Partition**: normal
- **Available Resources**: Big Red 200, Quartz

## Tips for Success
1. Start with the data analysis job to ensure everything works
2. Check logs regularly for any errors
3. Use the complete pipeline job for final results
4. Save important results to the results/ directory
5. Keep models in the models/ directory for future use
