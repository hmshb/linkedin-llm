"""Step 4: Generate LinkedIn posts using the fine-tuned model.

Usage:
    python inference.py "Share 3 tips for networking at tech conferences"
    python inference.py --interactive
"""

import argparse

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config
from linkedin_knowledge import FINE_TUNING_SYSTEM_PROMPT

load_dotenv()


def load_model():
    """Load base model with merged LoRA adapter."""
    print(f"Loading base model: {config.MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading adapter from: {config.OUTPUT_DIR}")
    model = PeftModel.from_pretrained(model, str(config.OUTPUT_DIR))

    tokenizer = AutoTokenizer.from_pretrained(str(config.OUTPUT_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_post(model, tokenizer, content_idea: str) -> str:
    """Generate a LinkedIn post from a content idea."""
    prompt = (
        f"System: {FINE_TUNING_SYSTEM_PROMPT}\n\n"
        f"Instruct: {content_idea}\n\n"
        f"Output: "
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            repetition_penalty=config.REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (skip the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate LinkedIn posts")
    parser.add_argument("topic", nargs="?", help="Content idea for the post")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive mode"
    )
    args = parser.parse_args()

    model, tokenizer = load_model()
    print("Model loaded!\n")

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            topic = input("Content idea: ").strip()
            if topic.lower() in ("quit", "exit", "q"):
                break
            if not topic:
                continue
            print("\n--- Generated LinkedIn Post ---\n")
            post = generate_post(model, tokenizer, topic)
            print(post)
            print("\n" + "=" * 50 + "\n")
    elif args.topic:
        print(f"Content idea: {args.topic}\n")
        print("--- Generated LinkedIn Post ---\n")
        post = generate_post(model, tokenizer, args.topic)
        print(post)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
