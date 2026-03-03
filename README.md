# Fine-Tune Llama 3.1 8B as a LinkedIn Content Creator

Fine-tune Meta's Llama 3.1 8B Instruct model using QLoRA to generate high-performing LinkedIn posts from content ideas. Uses the `claude` CLI (from your Claude Code subscription) to generate 500 synthetic training examples with LinkedIn algorithm best practices baked in. No separate API key needed.

## Project Structure

```
├── config.py                  # Central config (model, hyperparams, paths)
├── linkedin_knowledge.py      # LinkedIn policies, algorithm rules, best practices
├── linkedin_topics.py         # 500 content ideas across 50 categories
├── generate_data.py           # Step 1: Generate training data via claude CLI
├── prepare_dataset.py         # Step 2: Format into chat template + train/eval split
├── train.py                   # Step 3: QLoRA fine-tuning
├── inference.py               # Step 4: Generate posts from fine-tuned model
├── data/                      # Generated datasets
└── output/                    # Fine-tuned adapter weights
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN for accessing gated Llama model
```

### 3. Accept Llama 3.1 license

Visit [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on Hugging Face and accept the license agreement. Then log in:

```bash
huggingface-cli login
```

## Usage

### Step 1: Generate training data

```bash
python generate_data.py
```

Generates ~500 LinkedIn posts via the `claude` CLI (uses your Claude Code subscription). Supports resuming with `--resume` if interrupted. Output: `data/raw_posts.jsonl`

### Step 2: Prepare dataset

```bash
python prepare_dataset.py
```

Formats data into Llama 3.1 chat template, splits 90/10 train/eval. Output: `data/train/` and `data/eval/`

### Step 3: Fine-tune

```bash
python train.py
```

QLoRA fine-tuning with 4-bit quantization. Requires a GPU with ~16GB VRAM (e.g., RTX 4090, A100). Output: `output/linkedin-llama-adapter/`

### Step 4: Generate posts

```bash
# Single post
python inference.py "Share 3 tips for networking at tech conferences"

# Interactive mode
python inference.py --interactive
```

## Configuration

All hyperparameters and settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | Base model |
| `LORA_R` | 16 | LoRA rank |
| `LORA_ALPHA` | 32 | LoRA alpha |
| `NUM_EPOCHS` | 3 | Training epochs |
| `BATCH_SIZE` | 4 | Per-device batch size |
| `LEARNING_RATE` | 2e-4 | Learning rate |
| `TEMPERATURE` | 0.7 | Inference temperature |

## Requirements

- Python 3.10+
- CUDA-capable GPU with 16GB+ VRAM
- Claude Code CLI (for data generation — uses your subscription)
- Hugging Face token (for Llama model access)
