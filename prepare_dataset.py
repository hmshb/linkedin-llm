"""Step 2: Format raw JSONL data into Hugging Face datasets for training.

Converts raw {input, output} pairs into Llama 3.1 chat format and splits
into train/eval sets.

Usage:
    python prepare_dataset.py
"""

import json
import random

from datasets import Dataset
from tqdm import tqdm

import config
from linkedin_knowledge import FINE_TUNING_SYSTEM_PROMPT


def load_raw_data(path) -> list[dict]:
    """Load raw JSONL data."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_chat(example: dict) -> str:
    """Format a single example into Phi-2 instruct format."""
    return (
        f"System: {FINE_TUNING_SYSTEM_PROMPT}\n\n"
        f"Instruct: {example['input']}\n\n"
        f"Output: {example['output']}"
    )


def main():
    print(f"Loading raw data from {config.RAW_DATA_PATH}")
    raw_data = load_raw_data(config.RAW_DATA_PATH)
    print(f"Loaded {len(raw_data)} examples")

    # Format all examples
    formatted = []
    for example in tqdm(raw_data, desc="Formatting"):
        formatted.append({"text": format_chat(example)})

    # Shuffle and split
    random.seed(42)
    random.shuffle(formatted)

    split_idx = int(len(formatted) * (1 - config.EVAL_RATIO))
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Save as Hugging Face datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    config.TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset.save_to_disk(str(config.TRAIN_DATA_DIR))
    eval_dataset.save_to_disk(str(config.EVAL_DATA_DIR))

    print(f"Saved train dataset to {config.TRAIN_DATA_DIR}")
    print(f"Saved eval dataset to {config.EVAL_DATA_DIR}")

    # Print a sample
    print("\n--- Sample formatted example ---")
    print(train_data[0]["text"][:500])
    print("...")


if __name__ == "__main__":
    main()
