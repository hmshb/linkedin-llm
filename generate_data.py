"""Step 1: Generate synthetic LinkedIn post training data using Claude CLI.

Uses the `claude` CLI (from your Claude Code subscription) instead of the
Anthropic API, so no separate API key is needed.

Usage:
    python generate_data.py
    python generate_data.py --resume          # Resume from checkpoint
    python generate_data.py --batch-size 5    # Run 5 concurrent CLI calls
    python generate_data.py --timeout 600     # Set CLI timeout to 600s
"""

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import config
from linkedin_knowledge import (
    DATA_GENERATION_SYSTEM_PROMPT,
    META_ANSWERS,
    META_TOPICS,
)
from linkedin_topics import ALL_TOPICS

# Lock for thread-safe file writes
_file_lock = threading.Lock()


def load_existing_data(path: Path) -> list[dict]:
    """Load existing JSONL data for resume support."""
    if not path.exists():
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_entry(path: Path, entry: dict):
    """Append a single entry to the JSONL file (thread-safe)."""
    with _file_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_post(topic: str, timeout: int) -> str | None:
    """Generate a LinkedIn post using the claude CLI."""
    for attempt in range(config.MAX_RETRIES):
        try:
            # Clear CLAUDECODE env var so CLI doesn't refuse to run
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)

            result = subprocess.run(
                [
                    "claude.exe",
                    "-p", topic,
                    "--system-prompt", DATA_GENERATION_SYSTEM_PROMPT,
                    "--model", config.CLAUDE_MODEL,
                    "--output-format", "text",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                error = result.stderr.strip() or "Empty response"
                raise RuntimeError(error)
        except subprocess.TimeoutExpired:
            if attempt < config.MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"\n  Timeout on attempt {attempt + 1}/{config.MAX_RETRIES}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  Timed out after {config.MAX_RETRIES} attempts: {topic[:60]}...")
                return None
        except Exception as e:
            if attempt < config.MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"\n  Retry {attempt + 1}/{config.MAX_RETRIES} after error: {e}")
                time.sleep(wait)
            else:
                print(f"\n  Failed after {config.MAX_RETRIES} attempts: {e}")
                return None


def generate_meta_examples() -> list[dict]:
    """Generate meta examples about LinkedIn best practices (no CLI call needed)."""
    examples = []
    for topic in META_TOPICS:
        if topic in META_ANSWERS:
            examples.append({"input": topic, "output": META_ANSWERS[topic]})
    return examples


def process_batch(batch: list[str], output_path: Path, pbar: tqdm,
                  stats: dict, timeout: int):
    """Process a batch of topics concurrently using a thread pool."""
    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
        future_to_topic = {
            executor.submit(generate_post, topic, timeout): topic
            for topic in batch
        }

        for future in as_completed(future_to_topic):
            topic = future_to_topic[future]
            try:
                post = future.result()
            except Exception as e:
                print(f"\n  Unexpected error for topic: {e}")
                post = None

            if post:
                entry = {"input": topic, "output": post}
                save_entry(output_path, entry)
                stats["generated"] += 1
            else:
                stats["failed"] += 1

            pbar.update(1)
            pbar.set_postfix(ok=stats["generated"], fail=stats["failed"])


def main():
    parser = argparse.ArgumentParser(description="Generate LinkedIn training data")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Number of concurrent CLI calls per batch (default: 3)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout in seconds per CLI call (default: 600)")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RAW_DATA_PATH

    # Load existing data for resume
    existing = load_existing_data(output_path) if args.resume else []
    existing_inputs = {entry["input"] for entry in existing}

    if not args.resume and output_path.exists():
        output_path.unlink()

    # 1. Add pre-written meta examples
    meta_examples = generate_meta_examples()
    for ex in meta_examples:
        if ex["input"] not in existing_inputs:
            save_entry(output_path, ex)
            existing_inputs.add(ex["input"])

    # Collect meta topics that need CLI generation
    meta_to_generate = []
    for topic in META_TOPICS:
        if topic not in META_ANSWERS and topic not in existing_inputs:
            meta_to_generate.append(topic)

    # 2. Collect regular topics that need generation
    topics_to_generate = []
    for item in ALL_TOPICS:
        if item["topic"] not in existing_inputs:
            topics_to_generate.append(item["topic"])

    all_to_generate = meta_to_generate + topics_to_generate
    total = len(all_to_generate)

    if total == 0:
        print("All examples already generated. Nothing to do.")
        final_count = len(load_existing_data(output_path))
        print(f"Total examples: {final_count}")
        return

    print(f"Pre-written meta examples: {len(meta_examples)}")
    print(f"Claude CLI calls needed: {total}")
    print(f"Batch size: {args.batch_size} concurrent | Timeout: {args.timeout}s per call")
    if existing:
        print(f"Resuming from {len(existing)} existing examples")

    stats = {"generated": 0, "failed": 0}

    # Split into batches
    batches = [
        all_to_generate[i:i + args.batch_size]
        for i in range(0, total, args.batch_size)
    ]

    with tqdm(total=total, desc="Generating posts") as pbar:
        for batch_idx, batch in enumerate(batches):
            process_batch(batch, output_path, pbar, stats, args.timeout)

            # Small delay between batches to avoid rate limiting
            if batch_idx < len(batches) - 1:
                time.sleep(2)

    final_count = len(load_existing_data(output_path))
    print(f"\nDone! Generated: {stats['generated']}, Failed: {stats['failed']}")
    print(f"Total examples in {output_path}: {final_count}")


if __name__ == "__main__":
    main()
