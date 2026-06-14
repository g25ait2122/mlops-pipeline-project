import os
from transformers import pipeline


def main():
    model_name = os.getenv("HF_MODEL_NAME", "VaibhavG25AIT2122/mlops-emotion-classifier")
    input_text = os.getenv("INPUT_TEXT", "I am very happy today")

    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
    )

    prediction = classifier(input_text)

    print("Input text:", input_text)
    print("Model:", model_name)
    print("Prediction:", prediction)


if __name__ == "__main__":
    main()
