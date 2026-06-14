FROM python:3.11-slim

ARG HF_MODEL_NAME=VaibhavG25AIT2122/mlops-emotion-classifier
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

WORKDIR /app

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY src/ ./src/

CMD ["python", "src/inference.py"]
