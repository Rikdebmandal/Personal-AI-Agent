"""Turn a folder of your own documents into a JSONL instruction-tuning dataset.

Walks a directory, extracts text from .txt/.md/.pdf/.docx files, chunks it,
and uses the already-running local Ollama model to generate grounded
question/answer pairs from each chunk (so the model can learn your content
without you having to hand-write training examples).

Usage:
    python finetune/build_dataset.py "C:\\path\\to\\your\\notes" --out finetune/data/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx
import yaml

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
PAIRS_PER_CHUNK = 3
SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf", ".docx")

GEN_PROMPT = """You are building a training dataset. Given the passage below, write {n} \
question-and-answer pairs that a person could reasonably ask about its content, with answers \
based ONLY on the passage. Do not invent facts that aren't in the passage.

Return ONLY a JSON array, nothing else, like:
[{{"instruction": "...", "output": "..."}}, ...]

Passage:
\"\"\"
{chunk}
\"\"\"
"""


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"skip {path} (pip install pypdf for PDF support)", file=sys.stderr)
            return ""
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        try:
            import docx
        except ImportError:
            print(f"skip {path} (pip install python-docx for .docx support)", file=sys.stderr)
            return ""
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return ""


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if len(chunk) > 200:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def generate_pairs(client: httpx.Client, host: str, model: str, chunk: str, n: int = PAIRS_PER_CHUNK) -> list[dict]:
    resp = client.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": GEN_PROMPT.format(n=n, chunk=chunk)}],
            "stream": False,
        },
        timeout=600,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return []
    try:
        pairs = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [
        {"instruction": p["instruction"], "input": "", "output": p["output"]}
        for p in pairs
        if isinstance(p, dict) and p.get("instruction") and p.get("output")
    ]


def build_dataset(source_dir: str, out_path: str, model: str | None = None) -> None:
    config = load_config()
    host = config["ollama"]["host"]
    model = model or config["ollama"]["model"]

    src = Path(source_dir)
    files = [p for p in src.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES]
    print(f"found {len(files)} documents under {src}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with httpx.Client() as client, out.open("w", encoding="utf-8") as f:
        for path in files:
            text = extract_text(path)
            if not text.strip():
                continue
            chunks = chunk_text(text)
            print(f"{path.name}: {len(chunks)} chunks")
            for chunk in chunks:
                pairs = generate_pairs(client, host, model, chunk)
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
                    written += 1
                    f.flush()

    print(f"wrote {written} training pairs to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", help="Folder of documents to build a dataset from")
    parser.add_argument("--out", default="finetune/data/dataset.jsonl", help="Output JSONL path")
    parser.add_argument("--model", default=None, help="Ollama model to use for generation (defaults to config.yaml)")
    args = parser.parse_args()
    build_dataset(args.source_dir, args.out, args.model)
