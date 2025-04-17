# Parameter‑Efficient Distillation of RoBERTa with LoRA

This project implements knowledge‐distillation fine‑tuning of `roberta-base` on the AG News dataset using LoRA adapters. It includes:

- **Data**: AG News (via `datasets`) + `data/test_unlabelled.pkl`
- **Teacher**: Full `roberta-base` for logit generation
- **Student**: `roberta-base` wrapped with LoRA (rank=8, α=16, dropout=0.1)  
- **Distillation**: Temperature=2.0, α=0.7  
- **Training**: 5 epochs, train‐batch=16, eval‐batch=64, cosine LR scheduler  
- **Results**: ≥ 90% accuracy in validation set, ≥ 84% accuracy in test set.

## 🚀 Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/<your-org>/deep-learning-spring2025-lora.git
cd deep-learning-spring2025-lora

# 2. Install dependencies
bash setup.sh

# 3. Download data from Kaggle into data/
kaggle competitions download -c deep-learning-spring-2025-project-2 -p data/
unzip data/deep-learning-spring-2025-project-2.zip -d data/

# 4. Train with distillation
python scripts/train_distill.py \
  --data_dir data \
  --output_dir results/model_checkpoint

# 5. Inspect training/validation curves in the notebook
jupyter notebook notebooks/distillation_pipeline.ipynb

# 6. Generate Kaggle submission
python scripts/inference.py \
  --checkpoint results/model_checkpoint \
  --test_file data/test_unlabelled.pkl \
  --output_csv submission.csv
