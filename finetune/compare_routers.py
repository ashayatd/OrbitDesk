"""Compare routers on the SAME held-out test set: Groq zero-shot vs Groq few-shot vs fine-tuned LoRA.

    uv run --group eval python finetune/compare_routers.py

- Groq rows need GROQ_API_KEY (model: GROQ_SMALL_MODEL, the production router model).
- The fine-tuned row comes from finetune/results/finetuned_predictions.json, written by the
  Colab notebook (T4 latency), or, if ROUTER_ADAPTER is set and peft is installed, from
  local CPU inference (CPU latency). Missing rows are reported as "not run", never invented.
Metrics: intent accuracy, intent macro-F1, urgency accuracy, joint accuracy, strict-JSON rate,
latency p50/p95, estimated cost per 1,000 requests. Logged to MLflow ("orbitdesk-router").
Writes reports/router_comparison.json.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))
sys.path.insert(0, str(ROOT / "evals"))

from router_metrics import score  # noqa: E402

from app.agent.router import router_messages  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm.pricing import token_cost  # noqa: E402

DATA = ROOT / "finetune" / "data"
RESULTS = ROOT / "finetune" / "results"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 1)


def run_groq(test: list[dict], few_shot: bool) -> dict:
    from app.llm.client import get_llm

    s = get_settings()
    s.llm_fallbacks_enabled = False
    llm = get_llm("small", max_tokens=40)
    outputs, lat, tin, tout = [], [], 0, 0
    for row in test:
        msgs = router_messages(row["message"], few_shot)
        t0 = time.perf_counter()
        resp = llm.invoke(msgs)
        lat.append((time.perf_counter() - t0) * 1000)
        outputs.append(str(resp.content))
        um = resp.usage_metadata or {}
        tin, tout = tin + um.get("input_tokens", 0), tout + um.get("output_tokens", 0)
    cost = token_cost(s.groq_small_model, tin, tout)
    return {
        "router": f"groq_{'few' if few_shot else 'zero'}_shot",
        "model": s.groq_small_model,
        "metrics": score(test, outputs),
        "latency_ms_p50": pct(lat, 50),
        "latency_ms_p95": pct(lat, 95),
        "latency_note": "network round trip to Groq included",
        "tokens_per_request": round((tin + tout) / len(test), 1),
        "est_cost_per_1k_requests_usd": round(cost / len(test) * 1000, 4) if cost is not None else None,
        "outputs": outputs,
    }


def finetuned_row(test: list[dict]) -> dict | None:
    p = RESULTS / "finetuned_predictions.json"
    if p.exists():
        res = json.loads(p.read_text(encoding="utf-8"))
        by_msg = {x["message"]: x for x in res["predictions"]}
        if all(r["message"] in by_msg for r in test):  # must be the same test set
            outputs = [by_msg[r["message"]]["output"] for r in test]
            lat = [by_msg[r["message"]]["latency_ms"] for r in test]
            return {
                "router": "finetuned_lora",
                "model": f"{res['base_model']} + LoRA ({res['method']})",
                "metrics": score(test, outputs),
                "latency_ms_p50": pct(lat, 50),
                "latency_ms_p95": pct(lat, 95),
                "latency_note": f"local generate() on {res['device']}, no network",
                "tokens_per_request": None,
                "est_cost_per_1k_requests_usd": 0.0,
                "cost_note": "self-hosted: no per-token fee, but you pay for the machine it runs on",
                "train_seconds": res.get("train_seconds"),
                "outputs": outputs,
            }
        print("finetuned_predictions.json was made on a different test set; ignoring it")
    s = get_settings()
    if s.router_adapter:
        from app.agent.router import FinetunedRouter

        router = FinetunedRouter(s.router_adapter)
        outputs, lat = [], []
        for r in test:
            t0 = time.perf_counter()
            d = router.route(r["message"])
            lat.append((time.perf_counter() - t0) * 1000)
            outputs.append(json.dumps({"intent": d.intent, "urgency": d.urgency}) if d.parsed else "")
        return {
            "router": "finetuned_lora",
            "model": s.router_adapter,
            "metrics": score(test, outputs),
            "latency_ms_p50": pct(lat, 50),
            "latency_ms_p95": pct(lat, 95),
            "latency_note": "local CPU",
            "est_cost_per_1k_requests_usd": 0.0,
            "outputs": outputs,
        }
    return None


def main() -> None:
    test_path = DATA / "test.jsonl"
    if not test_path.exists():
        sys.exit("finetune/data/test.jsonl not found: run finetune/generate_router_data.py first.")
    test = load_jsonl(test_path)
    rows, not_run = [], []
    if get_settings().groq_api_key:
        rows += [run_groq(test, few_shot=False), run_groq(test, few_shot=True)]
    else:
        not_run += ["groq_zero_shot", "groq_few_shot (no GROQ_API_KEY)"]
    ft = finetuned_row(test)
    if ft:
        rows.append(ft)
    else:
        not_run.append("finetuned_lora (no finetuned_predictions.json from Colab and no ROUTER_ADAPTER)")

    from mlflow_utils import init_mlflow

    mlflow = init_mlflow("orbitdesk-router")
    for r in rows:
        with mlflow.start_run(run_name=r["router"]):
            mlflow.log_params({"router": r["router"], "model": r["model"][:250], "n_test": len(test)})
            metrics = {**r["metrics"], "latency_ms_p50": r["latency_ms_p50"], "latency_ms_p95": r["latency_ms_p95"]}
            if r.get("est_cost_per_1k_requests_usd") is not None:
                metrics["est_cost_per_1k_requests_usd"] = r["est_cost_per_1k_requests_usd"]
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})

    best = max(rows, key=lambda r: (r["metrics"]["intent_macro_f1"], r["metrics"]["joint_accuracy"])) if rows else None
    report = {
        "n_test": len(test),
        "test_set_hand_reviewed": (DATA / "test_review.csv").exists() and "see finetune/data/test_review.csv",
        "rows": [{k: v for k, v in r.items() if k != "outputs"} for r in rows],
        "not_run": not_run,
        "best_by_intent_macro_f1": best["router"] if best else None,
        "per_example": [
            {
                "message": t["message"],
                "gold": f"{t['intent']}/{t['urgency']}",
                **{r["router"]: r["outputs"][i] for r in rows},
            }
            for i, t in enumerate(test)
        ],
    }
    out = ROOT / "reports" / "router_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in rows:
        m = r["metrics"]
        print(
            f"{r['router']:16s} intent_acc={m['intent_accuracy']:.3f} macroF1={m['intent_macro_f1']:.3f} "
            f"urgency_acc={m['urgency_accuracy']:.3f} json={m['strict_json_rate']:.2f} "
            f"p50={r['latency_ms_p50']}ms cost/1k={r.get('est_cost_per_1k_requests_usd')}"
        )
    print("not run:", not_run)


if __name__ == "__main__":
    main()
