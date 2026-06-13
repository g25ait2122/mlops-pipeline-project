# MLOps Emotion Classifier Pipeline

An end-to-end MLOps pipeline that fine-tunes DistilBERT for emotion classification. Built for the PGD AI Program at IIT Jodhpur, this project demonstrates automated training, testing, and deployment workflows using GitHub Actions, Docker, and Weights & Biases.

## What Does This Thing Do?

Simple - it takes text and tells you what emotion it expresses. Think "I love machine learning!" → joy, or "This bug won't fix itself" → anger. We fine-tuned DistilBERT on the emotion dataset to classify text into 6 categories:

- Sadness
- Joy  
- Love
- Anger
- Fear
- Surprise

## Project Structure

```
mlops-pipeline-project/
├── .github/workflows/
│   ├── ci.yml              # Linting on every push to develop
│   └── inference.yml       # Manual inference workflow
├── src/
│   ├── data_prep.py        # Dataset loading and preprocessing
│   └── inference.py        # Model inference script
├── Dockerfile              # Container setup for deployment
├── requirements.txt        # Python dependencies
├── id2label.json          # Label mapping (0→sadness, etc.)
└── README.md              # You're reading it
```

## Pipeline Architecture

### Training Pipeline (Kaggle Notebook)
```
┌─────────────────────────────────────────────────────────────┐
│                   Kaggle Notebook (GPU)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Load Dataset   │
                    │ dair-ai/emotion │
                    │  (2000 samples) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Preprocessing  │
                    │  - Drop nulls   │
                    │  - Lowercase    │
                    │  - Tokenize     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Fine-tune     │
                    │   DistilBERT    │
                    │   2 epochs      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  W&B Tracking   │
                    │  - Loss curves  │
                    │  - Metrics      │
                    └────────┬────────┘
                             │
                             ▼
            ┌──────────────────────────────────┐
            │          Push to HF Hub          │
            │ VaibhavG25AIT/emotion-classifier │
            └──────────────────────────────────┘
```

### CI/CD Pipeline (GitHub Actions)
```
┌──────────────────────────────────────────────────────────────┐
│                    Developer Push to develop                 │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   GitHub Action │
                    │   (CI Workflow) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Checkout Code  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Setup Python   │
                    │     3.11        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Install Deps   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Run Flake8     │
                    │  Lint Check     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                  PASS              FAIL
                    │                 │
                    ▼                 ▼
               ✅ Merge           ❌ Fix Code
```

### Inference Pipeline (Manual Trigger)
```
┌──────────────────────────────────────────────────────────────┐
│            User Triggers Workflow from GitHub UI             │
│                   (Provides input text)                      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  GitHub Action  │
                    │   (Inference)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Checkout Code  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │                 │
                    │  Install Deps   │
                    │                 │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Load Model    │
                    │   From HF Hub   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │                 │
                    │  Run Inference  │
                    │                 │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Output Result  │
                    │  emotion: joy   │
                    └─────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11
- HuggingFace account + token (for model access)
- Docker (optional, for containerized deployment)

### Installation

Clone the repo:
```bash
git clone https://github.com/g25ait2122/mlops-pipeline-project.git
cd mlops-pipeline-project
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Inference Locally

Set your environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export HF_MODEL_NAME="VaibhavG25AIT2122/mlops-emotion-classifier"
export INPUT_TEXT="I'm so excited about this project!"
```

Run inference:
```bash
python src/inference.py
```

Expected output:
```
Loading model: VaibhavG25AIT2122/mlops-emotion-classifier
Input: I'm so excited about this project!
Prediction: [{'label': 'joy', 'score': 0.8234}]
```

### Using the Model Directly

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="VaibhavG25AIT2122/mlops-emotion-classifier")
result = classifier("Rush hour traffic is driving me crazy!")
print(result)  # [{'label': 'anger', 'score': 0.XX}]
```

### Docker Deployment

Build the image:
```bash
docker build -t emotion-classifier \
  --build-arg HF_MODEL_NAME="VaibhavG25AIT2122/mlops-emotion-classifier" .
```

Run the container:
```bash
docker run -e HF_TOKEN="your_token" \
           -e INPUT_TEXT="Docker makes deployment easy!" \
           emotion-classifier
```

## GitHub Actions Workflows

### CI Workflow
**Trigger:** Push to `develop` or PR to `main`  
**What it does:** Runs flake8 linting on all Python files in `src/`

### Inference Workflow
**Trigger:** Manual (workflow_dispatch)  
**What it does:** Takes user input text and runs inference on the deployed model

To trigger manually:
1. Go to **Actions** tab
2. Select **Inference** workflow
3. Click **Run workflow**
4. Enter your text and click **Run**

## Model Details

### Training Setup
- **Base Model:** distilbert-base-uncased
- **Dataset:** dair-ai/emotion (2000 samples for speed)
- **Epochs:** 2
- **Learning Rate:** 5e-5
- **Batch Size:** 16
- **Hardware:** Kaggle T4 GPU (< 3 hours)

### Performance
- **Accuracy:** ~82%
- **F1 Score:** ~0.81

Not production-ready, but good enough to show the pipeline works!

### Training Procedure

We used `data_prep.py` to:
1. Load 5000 samples from emotion dataset
2. Drop null values
3. Lowercase all text
4. Create label mappings (saved to `id2label.json`)

Fine-tuning was done in a Kaggle notebook with W&B tracking for:
- Training/validation loss
- Accuracy metrics
- Model checkpoints

## Known Issues & Fixes

### DistilBERT token_type_ids Error
DistilBERT doesn't use token_type_ids. Fixed by configuring tokenizer:
```python
tokenizer.model_input_names = ["input_ids", "attention_mask"]
```

### NumPy 2.x Compatibility
PyTorch/transformers libraries aren't compatible with NumPy 2.x yet. Pinned to `numpy<2` in requirements.

## Development

### Branch Strategy
- `main` - Production-ready code
- `develop` - Active development

CI runs on pushes to `develop`. PRs required to merge into `main`.

### Code Quality
We use flake8 with max line length 120. Run locally:
```bash
flake8 src/ --max-line-length=120
```

## Why This Project Exists

This was built for an MLOps assignment to demonstrate:
- ✅ Model training with experiment tracking (W&B)
- ✅ Version control with Git
- ✅ CI/CD with GitHub Actions
- ✅ Containerization with Docker
- ✅ Model deployment to HuggingFace Hub
- ✅ Automated testing and linting

It's not meant for production use - the model was trained on a tiny dataset to fit within Kaggle's free GPU limits. But it shows how a real ML pipeline should work.

## License

MIT License

## Team

**Group 15**  
PGD AI Program, IIT Jodhpur

---

**Model Card:** [VaibhavG25AIT2122/mlops-emotion-classifier](https://huggingface.co/VaibhavG25AIT2122/mlops-emotion-classifier)  
**Issues?** Feel free to create an issue