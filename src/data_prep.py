import json
from datasets import load_dataset


def main():
    # Loading dataset
    dataset = load_dataset("dair-ai/emotion", split="train[:5000]")

    # Cleanup, droping nulls and lowercase text
    dataset = dataset.filter(lambda x: x['text'] is not None)
    dataset = dataset.map(lambda x: {'text': x['text'].lower()})

    # Extract labels and create mapping
    labels = dataset.features["label"].names
    id2label = {id: label for id, label in enumerate(labels)}

    # Save mapping
    with open("id2label.json", "w") as f:
        json.dump(id2label, f, indent=4)

    print("Data prep complete. id2label.json saved.")
    print("Class distribution:", dataset.to_pandas()['label'].value_counts().to_dict())


if __name__ == "__main__":
    main()
