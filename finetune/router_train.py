"""QLoRA fine-tuning of a small instruct model as the OrbitDesk router (strict JSON {intent, urgency}).

This file IS the notebook: finetune/build_notebook.py embeds it into router_lora.ipynb,
so the code that runs on Colab is the code that was smoke-tested locally.

    Colab (T4):  run the notebook (QLoRA 4-bit, full data, ~3 epochs)
    Local smoke: uv run --group finetune python finetune/router_train.py --smoke
                 (CPU, no 4-bit, tiny data, 2 steps: checks the code path, not quality)
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ROUTER_SYSTEM = None  # set by the notebook build; locally read from prompts/router_v1.yaml
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def router_system() -> str:
    if ROUTER_SYSTEM:
        return ROUTER_SYSTEM
    import yaml

    return yaml.safe_load((HERE.parent / "prompts" / "router_v1.yaml").read_text(encoding="utf-8"))["system"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def to_prompt(message: str) -> list[dict]:
    """Same system text + user template as the production prompt router (zero-shot form)."""
    return [{"role": "system", "content": router_system()}, {"role": "user", "content": f"Message: {message}"}]


def to_example(row: dict) -> dict:
    target = json.dumps({"intent": row["intent"], "urgency": row["urgency"]})
    return {"prompt": to_prompt(row["message"]), "completion": [{"role": "assistant", "content": target}]}


def load_model(model_name: str, use_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {}
    if use_4bit:  # QLoRA: frozen base in 4-bit NF4, LoRA adapters trained in bf16/fp16
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16
        )
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tok


def train(
    train_rows,
    eval_rows,
    out_dir: str,
    use_4bit: bool,
    epochs: float = 3,
    max_steps: int = -1,
    model_name: str = BASE_MODEL,
):
    import math

    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    model, tok = load_model(model_name, use_4bit)
    steps_per_epoch = math.ceil(len(train_rows) / 16)  # batch 8 x grad accumulation 2
    warmup = 0 if max_steps > 0 else max(1, int(0.05 * steps_per_epoch * epochs))  # ~5% warmup
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        # smoke runs on CPU: tiny batches (a 151k-token vocabulary makes fp32 logits huge)
        per_device_train_batch_size=2 if max_steps > 0 else 8,
        gradient_accumulation_steps=1 if max_steps > 0 else 2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup,  # transformers 5 removed warmup_ratio
        logging_steps=1 if max_steps > 0 else 10,
        eval_strategy="epoch" if max_steps < 0 else "no",
        save_strategy="no",
        max_length=512,
        completion_only_loss=True,  # loss only on the JSON answer, not the system prompt
        fp16=use_4bit,  # T4: fp16 mixed precision
        bf16=False,  # TRL defaults to bf16, which the T4 (and CPU smoke runs) can't use
        use_cpu=not use_4bit,
        report_to="none",
        seed=13,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list([to_example(r) for r in train_rows]),
        eval_dataset=Dataset.from_list([to_example(r) for r in eval_rows]) if max_steps < 0 else None,
        processing_class=tok,
        peft_config=lora,
    )
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return trainer.model, tok, train_s


def predict(model, tok, messages: list[str]) -> tuple[list[str], list[float]]:
    """Greedy decoding, one message at a time (that's how the router runs in production)."""
    import torch

    model.eval()
    outs, lat = [], []
    for m in messages:
        # transformers 5 returns a BatchEncoding (input_ids + attention_mask) here, not a bare tensor
        enc = tok.apply_chat_template(
            to_prompt(m), add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        ids = enc["input_ids"]
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        lat.append((time.perf_counter() - t0) * 1000)
        outs.append(tok.decode(gen[0][ids.shape[1] :], skip_special_tokens=True))
    return outs, lat


def run(data_dir: Path, out_dir: str, use_4bit: bool, smoke: bool = False, push_to: str = "") -> dict:
    from router_metrics import score

    train_rows, test_rows = load_jsonl(data_dir / "train.jsonl"), load_jsonl(data_dir / "test.jsonl")
    if smoke:
        train_rows, test_rows = train_rows[:16], test_rows[:4]
    model, tok, train_s = train(train_rows, test_rows, out_dir, use_4bit, max_steps=2 if smoke else -1)
    outputs, lat = predict(model, tok, [r["message"] for r in test_rows])
    metrics = score(test_rows, outputs)
    import torch

    result = {
        "base_model": BASE_MODEL,
        "method": "QLoRA 4-bit NF4, LoRA r=16 alpha=32 on all linear layers" if use_4bit else "LoRA (no 4-bit)",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "smoke_test": smoke,
        "n_train": len(train_rows),
        "train_seconds": round(train_s, 1),
        "metrics": metrics,
        "latency_ms_p50": round(statistics.median(lat), 1),
        "latency_ms_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1),
        "predictions": [
            {
                "message": r["message"],
                "gold": {"intent": r["intent"], "urgency": r["urgency"]},
                "output": o,
                "latency_ms": round(ms, 1),
            }
            for r, o, ms in zip(test_rows, outputs, lat, strict=True)
        ],
    }
    if push_to:
        model.push_to_hub(push_to, private=False)
        tok.push_to_hub(push_to)
        result["hub_repo"] = push_to
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "router-lora"))
    args = ap.parse_args()
    data = Path(args.data)
    if args.smoke and not (data / "train.jsonl").exists():
        data = HERE / "data_smoke"  # tiny hand-written set so the code path runs before any LLM data exists
    import torch

    res = run(
        data,
        args.out,
        use_4bit=torch.cuda.is_available(),
        smoke=args.smoke,
        push_to=os.environ.get("HF_PUSH_REPO", "") if os.environ.get("HF_TOKEN") else "",
    )
    print(json.dumps({k: v for k, v in res.items() if k != "predictions"}, indent=2))
    name = "finetuned_smoke.json" if args.smoke else "finetuned_predictions.json"
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / name).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
