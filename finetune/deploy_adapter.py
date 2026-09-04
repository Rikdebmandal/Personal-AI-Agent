"""Deploy a trained LoRA adapter (from local_qlora.py or the Colab notebook)
as a local Ollama model: convert it to GGUF, write a Modelfile, and register
it with `ollama create`.

Usage:
    python finetune/deploy_adapter.py finetune/output/lora_adapter \\
        --base-model-id Qwen/Qwen2.5-1.5B-Instruct \\
        --ollama-base qwen2.5:1.5b \\
        --name my-finetuned-model

`--base-model-id` must be the *unquantized* Hugging Face repo matching the
base you trained against (e.g. the plain Qwen/Qwen2.5-1.5B-Instruct, not the
unsloth ...-bnb-4bit variant) - the converter can't read bitsandbytes-
quantized weights directly, and the LoRA math doesn't depend on how the base
was quantized anyway.

`--ollama-base` must be an Ollama model tag that wraps the *same* base
architecture/weights (e.g. already pulled via `ollama pull qwen2.5:1.5b`) -
that's what your fine-tuned adapter gets layered on top of at inference time.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONVERT_SCRIPT = REPO_ROOT / "finetune" / "vendor" / "llama.cpp" / "convert_lora_to_gguf.py"

MODELFILE_TEMPLATE = """FROM {ollama_base}
ADAPTER {adapter_gguf}
"""


def convert_to_gguf(adapter_dir: Path, base_model_id: str, outtype: str) -> Path:
    outfile = adapter_dir / "adapter.gguf"
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(adapter_dir),
        "--outfile", str(outfile),
        "--outtype", outtype,
        "--base-model-id", base_model_id,
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return outfile


def write_modelfile(adapter_gguf: Path, ollama_base: str) -> Path:
    modelfile_path = adapter_gguf.parent / "Modelfile"
    modelfile_path.write_text(
        MODELFILE_TEMPLATE.format(ollama_base=ollama_base, adapter_gguf=adapter_gguf.resolve()),
        encoding="utf-8",
    )
    return modelfile_path


def register_with_ollama(modelfile_path: Path, model_name: str) -> None:
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def deploy(adapter_dir: str, base_model_id: str, ollama_base: str, name: str, outtype: str = "f16") -> None:
    adapter_path = Path(adapter_dir)
    gguf_path = convert_to_gguf(adapter_path, base_model_id, outtype)
    modelfile_path = write_modelfile(gguf_path, ollama_base)
    print(f"wrote {modelfile_path}")
    register_with_ollama(modelfile_path, name)
    print(f"done - chat with it via: ollama run {name}")
    print(f"or set ollama.model: \"{name}\" in config.yaml to use it in the app")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("adapter_dir", help="Directory containing adapter_model.safetensors + adapter_config.json")
    parser.add_argument("--base-model-id", required=True, help="Unquantized HF repo id matching the trained base")
    parser.add_argument("--ollama-base", required=True, help="Ollama tag of the same base model to layer the adapter onto")
    parser.add_argument("--name", required=True, help="Name to register the fine-tuned model under in Ollama")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0", "auto"])
    args = parser.parse_args()
    deploy(args.adapter_dir, args.base_model_id, args.ollama_base, args.name, args.outtype)
