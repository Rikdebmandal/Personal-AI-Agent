"""QLoRA fine-tune a small instruct model locally via Unsloth.

Sized for 4GB VRAM: defaults to a 1.5B base model, rank-16 LoRA, batch size 1
with gradient accumulation. Trains on an Alpaca-style JSONL dataset (the
format finetune/build_dataset.py produces: {"instruction", "input", "output"}
per line) and saves a LoRA adapter you can hand to finetune/deploy_adapter.py.

Usage:
    python finetune/local_qlora.py finetune/data/dataset.jsonl
    python finetune/local_qlora.py finetune/data/dataset.jsonl --max-steps 10   # smoke test
"""
from __future__ import annotations

import argparse

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DEFAULT_MAX_SEQ_LENGTH = 1024

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input \
that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def make_formatting_func(eos_token: str):
    """Called per-example (unbatched) by Unsloth's SFTTrainer, but must return a
    single-element list wrapping the formatted string, not the bare string.
    Appends EOS so the model actually learns where a response ends."""

    def formatting_func(example: dict) -> list[str]:
        return [
            PROMPT_TEMPLATE.format(
                instruction=example["instruction"],
                input=example.get("input", ""),
                output=example["output"],
            )
            + eos_token
        ]

    return formatting_func


def train(
    dataset_path: str,
    base_model: str = DEFAULT_BASE_MODEL,
    output_dir: str = "finetune/output/lora_adapter",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    epochs: float = 1.0,
    max_steps: int | None = None,
) -> None:
    print(f"loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"loaded {len(dataset)} training examples from {dataset_path}")

    sft_config = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps else -1,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="finetune/output/checkpoints",
        max_seq_length=max_seq_length,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        formatting_func=make_formatting_func(tokenizer.eos_token),
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path", help="Path to an Alpaca-style JSONL dataset")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="HF base model to fine-tune")
    parser.add_argument("--output", default="finetune/output/lora_adapter", help="Where to save the LoRA adapter")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None, help="Cap total training steps (useful for a smoke test)")
    args = parser.parse_args()
    train(args.dataset_path, args.base_model, args.output, args.max_seq_length, args.epochs, args.max_steps)
