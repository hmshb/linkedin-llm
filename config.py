"""Central configuration for the LinkedIn content creator fine-tuning project."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw_posts.jsonl"
TRAIN_DATA_DIR = DATA_DIR / "train"
EVAL_DATA_DIR = DATA_DIR / "eval"
OUTPUT_DIR = PROJECT_ROOT / "output" / "linkedin-phi2-adapter"

# Base model
MODEL_NAME = "microsoft/phi-2"

# Data generation (uses claude CLI from Claude Code subscription)
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3

# Dataset split
EVAL_RATIO = 0.1

# QLoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "dense"]

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.05
MAX_SEQ_LENGTH = 1024
BF16 = True
LOGGING_STEPS = 10
SAVE_STEPS = 50
EVAL_STEPS = 50

# Generation / inference settings
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.1
