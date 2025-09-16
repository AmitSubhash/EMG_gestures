# 🚀 EMG Analysis - Quick Start Guide for Quartz

## Project Structure
```
EMG_gestures/
├── scripts/           # Python analysis scripts
│   ├── jobs/         # Slurm job files
│   ├── logs/         # Job output logs
│   ├── results/      # Analysis results
│   └── models/       # Trained models
└── EMG_data/         # Your raw EMG data
```

## 🎯 Quick Start (3 Steps)

### Step 1: Navigate to Jobs Directory
```bash
cd scripts/jobs
```

### Step 2: Submit Your First Job
```bash
./submit_jobs.sh
```
Choose option 1 for data analysis (safest start)

### Step 3: Monitor Progress
```bash
# Check job status
squeue -u $USER

# View results when complete
ls -la ../logs/
ls -la ../results/
```

## 📊 Available Jobs

| Job Type | Duration | Resources | Purpose |
|----------|----------|-----------|---------|
| **Data Analysis** | 2 hours | 4 CPUs, 8GB | Explore data, extract features |
| **Vision Transformer** | 4 hours | 8 CPUs, 16GB | Train ViT on spectrograms |
| **Complete Pipeline** | 8 hours | 8 CPUs, 32GB | Run everything |

## 🔍 What Each Job Does

### Data Analysis Job
- ✅ Loads and validates your EMG data
- ✅ Creates comprehensive visualizations
- ✅ Extracts 234 features per window
- ✅ Generates 2D representations (STFT, Wavelets)
- 📁 **Output**: `logs/emg_analysis_*.out`, `results/` folder

### Vision Transformer Job
- ✅ Trains state-of-the-art ViT model
- ✅ Uses STFT spectrograms as 2D images
- ✅ Implements attention mechanisms
- ✅ Cross-validates performance
- 📁 **Output**: `models/vit_*.pth`, performance metrics

### Complete Pipeline Job
- ✅ Runs all analysis steps
- ✅ Trains all three models (ViT, CNN-LSTM, Domain Adaptive)
- ✅ Compares model performance
- ✅ Generates comprehensive results
- 📁 **Output**: Complete analysis package

## 🚨 Troubleshooting

### Job Fails to Start
```bash
# Check if you're in the right directory
pwd  # Should show: .../EMG_gestures/scripts/jobs

# Check account access
sacct -A r00602
```

### Job Runs But Fails
```bash
# Check error logs
cat ../logs/emg_analysis_<job_id>.err

# Check output logs
cat ../logs/emg_analysis_<job_id>.out
```

### Need More Resources
Edit the job files in `scripts/jobs/` to increase:
- `--cpus-per-task` (more CPUs)
- `--mem` (more memory)
- `--time` (longer duration)

## 📈 Expected Results

After running the complete pipeline, you'll have:
- **Performance metrics** for all three models
- **Trained models** ready for inference
- **Visualizations** showing data patterns
- **Feature importance** analysis
- **Cross-subject generalization** results

## 🎓 Learning Outcomes

You'll understand:
- How EMG signals are processed
- How deep learning works with biosignals
- How attention mechanisms improve performance
- How to handle cross-subject generalization
- How to compare different architectures

## 🚀 Ready to Start?

```bash
cd scripts/jobs
./submit_jobs.sh
```

**Choose option 1** to start with data analysis and see your EMG data come to life! 🎉
