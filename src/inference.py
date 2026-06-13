import os
from transformers import pipeline, AutoTokenizer


def main():
    # Pick model name from env
    model_name = os.getenv("HF_MODEL_NAME")
    text = os.getenv("INPUT_TEXT", "Doing MLOps assignment is a happy feeling!")
    hf_token = os.getenv("HF_TOKEN")

    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        classifier = pipeline("text-classification", model=model_name, tokenizer=tokenizer, token=hf_token)
        result = classifier(text)
        print(f"Input: {text}")
        print(f"Prediction: {result}")
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    main()
