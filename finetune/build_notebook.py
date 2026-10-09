"""Build finetune/router_lora.ipynb from router_train.py + router_metrics.py (single source of truth).

uv run python finetune/build_notebook.py
"""

import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def main() -> None:
    system = yaml.safe_load((ROOT / "prompts" / "router_v1.yaml").read_text(encoding="utf-8"))["system"]
    train_src = (HERE / "router_train.py").read_text(encoding="utf-8")
    marker = "ROUTER_SYSTEM = None  # set by the notebook build; locally read from prompts/router_v1.yaml"
    assert marker in train_src
    train_src = train_src.replace(marker, f"ROUTER_SYSTEM = {system!r}  # inlined from prompts/router_v1.yaml")
    metrics_src = (HERE / "router_metrics.py").read_text(encoding="utf-8")

    cells = [
        md("""
# OrbitDesk router: QLoRA fine-tune of Qwen2.5-0.5B-Instruct (free Colab T4)

Goal: replace the prompt-based router (Groq gpt-oss-20b) with a small local model that outputs
**strict JSON** `{"intent": ..., "urgency": ...}`, then compare on the same 60-example held-out test set.

**How to run**
1. Runtime > Change runtime type > **T4 GPU**.
2. Optional: add a Colab secret `HF_TOKEN` (write access) and set `HF_PUSH_REPO` below to push the adapter.
3. Provide the data: set `RAW_BASE` to your GitHub repo's raw URL, or leave it empty and upload
   `finetune/data/train.jsonl` and `finetune/data/test.jsonl` when prompted.
4. Run all. Download `finetuned_predictions.json` at the end and put it in `finetune/results/`
   in the repo, then run `python finetune/compare_routers.py`.

The code cells below are generated from `finetune/router_train.py` and `finetune/router_metrics.py`
by `finetune/build_notebook.py`, so this notebook runs the same code that was smoke-tested locally.
"""),
        code("""
# Pinned to the versions the local smoke test used (trl 1.15 changed the SFT loss path).
!pip install -q "trl==1.14.2" "peft>=0.12" "accelerate>=1.0" "datasets>=2.20" bitsandbytes pyyaml
"""),
        code("""
import os
from pathlib import Path

RAW_BASE = ""          # e.g. "https://raw.githubusercontent.com/<you>/orbitdesk/main"
HF_PUSH_REPO = ""      # e.g. "<your-hf-username>/orbitdesk-router-lora" (needs HF_TOKEN secret)

Path("data").mkdir(exist_ok=True)
if RAW_BASE:
    for name in ("train.jsonl", "test.jsonl"):
        os.system(f"wget -q -O data/{name} {RAW_BASE}/finetune/data/{name}")
else:
    from google.colab import files
    print("Upload train.jsonl and test.jsonl")
    for name, content in files.upload().items():
        Path("data", name).write_bytes(content)
print({p.name: sum(1 for _ in open(p)) for p in Path("data").glob("*.jsonl")})

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    print("No HF_TOKEN secret: the adapter will not be pushed to the Hub.")
"""),
        code("%%writefile router_metrics.py\n" + metrics_src),
        code("%%writefile router_train.py\n" + train_src),
        code("""
import json, torch
import router_train

assert torch.cuda.is_available(), "Select a T4 GPU runtime"
push = HF_PUSH_REPO if os.environ.get("HF_TOKEN") and HF_PUSH_REPO else ""
if push:
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
res = router_train.run(Path("data"), "outputs/router-lora", use_4bit=True, smoke=False, push_to=push)
Path("finetuned_predictions.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "predictions"}, indent=2))
"""),
        md("""
### Read the result honestly
Compare `metrics` above with the Groq rows from `compare_routers.py`. A fine-tuned 0.5B model often
matches a large model on a narrow 4-way classification and wins on latency and cost per request,
but it can lose on messages unlike the synthetic training data. Report whatever happened.
"""),
        code("""
from google.colab import files
files.download("finetuned_predictions.json")
"""),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = HERE / "router_lora.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
