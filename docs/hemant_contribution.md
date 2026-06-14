# MLOps Pipeline Project - Hemant Contribution

## Improved Kaggle Experiment

This contribution adds a robust Kaggle experiment notebook for emotion classification.

- Dataset: `dair-ai/emotion`
- Dataset access: Hugging Face Datasets Server API
- Model: `distilbert-base-uncased`
- W&B project: `mlops-assignment3`
- Hugging Face model: `VaibhavG25AIT2122/mlops-emotion-classifier`

## Improvements Added

- Raw dataset inspection
- Missing value and duplicate checks
- Class distribution analysis
- Text cleaning
- `id2label.json` and `label2id.json`
- Two experiment versions with W&B tracking
- Manual PyTorch training loop to avoid Kaggle package conflicts
- Best model selection using weighted F1
- Hugging Face model push
- Docker inference files
- GitHub Actions CI and inference workflows
