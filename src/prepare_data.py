from collections import Counter
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import os
import re

DATASET_NAME = "dair-ai/emotion"
LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
id2label = {i: label for i, label in enumerate(LABEL_NAMES)}
label2id = {label: i for i, label in id2label.items()}


def http_get_json(url, timeout=60):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def clean_text_value(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_label(label_value):
    if isinstance(label_value, int):
        return label_value
    if isinstance(label_value, str):
        label_value = label_value.strip()
        if label_value.isdigit():
            return int(label_value)
        if label_value in label2id:
            return label2id[label_value]
    raise ValueError(f"Unknown label value: {label_value!r}")


def fetch_split(split_name, max_rows=500):
    rows = []
    offset = 0
    while len(rows) < max_rows:
        params = {
            "dataset": DATASET_NAME,
            "split": split_name,
            "offset": offset,
            "length": min(100, max_rows - len(rows)),
        }
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(params)
        payload = http_get_json(url)

        if "error" in payload:
            params["config"] = "split"
            url = "https://datasets-server.huggingface.co/rows?" + urlencode(params)
            payload = http_get_json(url)

        batch = []
        for item in payload.get("rows", []):
            row = item.get("row", item)
            batch.append({
                "text": clean_text_value(row.get("text", "")),
                "label": normalize_label(row.get("label")),
            })

        if not batch:
            break

        rows.extend(batch)
        offset += len(batch)

    return rows


def inspect_rows(rows, split_name):
    missing_text = sum(1 for row in rows if clean_text_value(row.get("text")) == "")
    texts = [clean_text_value(row.get("text")) for row in rows]
    duplicate_count = len(texts) - len(set(texts))
    label_counts = Counter(int(row["label"]) for row in rows)

    print("=" * 80)
    print(f"Split: {split_name}")
    print("Rows:", len(rows))
    print("Columns: text, label")
    print("Missing/empty text rows:", missing_text)
    print("Duplicate text rows:", duplicate_count)
    print("Class distribution:")
    for label_id, count in sorted(label_counts.items()):
        print(f"  {label_id} - {id2label[label_id]}: {count}")


def clean_and_deduplicate(rows):
    cleaned = []
    seen = set()
    for row in rows:
        text = clean_text_value(row.get("text"))
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append({"text": text, "label": int(row["label"])})
    return cleaned


def main():
    os.makedirs("config", exist_ok=True)
    os.makedirs("prepared_data", exist_ok=True)

    train_rows = fetch_split("train", 2000)
    validation_rows = fetch_split("validation", 500)

    inspect_rows(train_rows, "train-raw")
    inspect_rows(validation_rows, "validation-raw")

    train_rows = clean_and_deduplicate(train_rows)
    validation_rows = clean_and_deduplicate(validation_rows)

    inspect_rows(train_rows, "train-cleaned")
    inspect_rows(validation_rows, "validation-cleaned")

    with open("config/id2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)

    with open("config/label2id.json", "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in label2id.items()}, f, indent=2)

    with open("prepared_data/train.jsonl", "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")

    with open("prepared_data/validation.jsonl", "w", encoding="utf-8") as f:
        for row in validation_rows:
            f.write(json.dumps(row) + "\n")

    print("Data preparation completed.")


if __name__ == "__main__":
    main()
