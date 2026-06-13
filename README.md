# MLOps Emotion Classifier Pipeline

An end-to-end MLOps pipeline demonstrating deployment and inference for emotion classification using a fine-tuned DistilBERT model. Built for the PGD AI Program at IIT Jodhpur, this project shows automated testing, deployment workflows, and CI/CD practices with GitHub Actions and Docker.

## What Does This Thing Do?

Simple - it takes text and tells you what emotion it expresses. Think "I love machine learning!" → joy, or "This bug won't fix itself" → anger. We're using a fine-tuned DistilBERT model that classifies text into 6 categories:

- sadness
- joy  
- love
- anger
- fear
- surprise

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
├── id2label.json           # Label mapping (0→sadness, etc.)
└── README.md               # You're reading it
```

## Pipeline Architecture

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

### Inference Workflow in Action

The workflow now appears in GitHub Actions and can be triggered manually:

![Inference Workflow](images/inference.png)

Example inference run showing successful emotion classification:

![Inference Run Output](images/output.png)

The workflow successfully:
- Loads the model from HuggingFace Hub
- Processes the input text
- Returns emotion predictions with confidence scores

## Model Details

### Model Information
- **Base Model:** distilbert-base-uncased (fine-tuned by model creator)
- **Training Data:** dair-ai/emotion dataset
- **Performance:** ~82% accuracy, ~0.81 F1 score
- **Model Source:** HuggingFace Hub

### Data Preparation

The `data_prep.py` script demonstrates how to:
1. Load samples from the emotion dataset
2. Drop null values
3. Lowercase text for preprocessing
4. Create label mappings (saved to `id2label.json`)

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
- Model deployment and inference workflows
- Version control with Git
- CI/CD with GitHub Actions
- Containerization with Docker
- Integration with HuggingFace models
- Automated testing and linting

We're focusing on the deployment and operations side of MLOps - showing how to take a model and build production-ready infrastructure around it.

## License

MIT License

## Team

**Group 15**  
PGD AI Program, IIT Jodhpur

---

**Model Card:** [VaibhavG25AIT2122/mlops-emotion-classifier](https://huggingface.co/VaibhavG25AIT2122/mlops-emotion-classifier)  
**Issues?** Feel free to create an issue