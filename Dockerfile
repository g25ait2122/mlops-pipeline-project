FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Take HF model name as argument
ARG HF_MODEL_NAME="distilbert-base-uncased"
ENV HF_MODEL_NAME=$HF_MODEL_NAME

COPY src/ ./src/

CMD ["python", "src/inference.py"]
