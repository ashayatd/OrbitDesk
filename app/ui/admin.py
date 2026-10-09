"""Admin tab: usage, failures, latency, top unanswered questions, cache hit rate, feedback."""

import gradio as gr

from app.observability.request_log import admin_stats


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def render(days: int):
    s = admin_stats(int(days))
    fb = s["feedback"]
    summary = (
        f"**Requests** {s['requests']} | **failure rate** {_pct(s['failure_rate'])} | "
        f"**unanswered** {_pct(s['unanswered_rate'])} | **avg latency** {s['avg_latency_ms'] or 0} ms | "
        f"**cache hit rate** {_pct(s['cache_hit_rate'])} | **est. cost** ${s['est_cost_usd'] or 0} | "
        f"**feedback** 👍 {fb.get('up', 0)} / 👎 {fb.get('down', 0)}"
    )
    per_day = [
        [r["day"], r["requests"], r["errors"], r["unanswered"], r["avg_latency_ms"]] for r in s["questions_per_day"]
    ]
    unanswered = [[r["question"], r["times"]] for r in s["top_unanswered"]]
    endpoints = [[r["endpoint"], r["n"], r["avg_latency_ms"]] for r in s["by_endpoint"]]
    down = [[r["ts"], r["question"], r["comment"]] for r in s["recent_thumbs_down"]]
    return summary, per_day, unanswered, endpoints, down


def build_admin_tab() -> None:
    gr.Markdown("Usage from `request_log` and `feedback` (SQLite). Questions are stored PII-masked.")
    with gr.Row():
        days = gr.Slider(1, 90, value=14, step=1, label="Window (days)")
        refresh = gr.Button("Refresh", variant="primary")
    summary = gr.Markdown()
    per_day = gr.Dataframe(headers=["day", "requests", "errors", "unanswered", "avg latency ms"], label="Per day")
    with gr.Row():
        unanswered = gr.Dataframe(headers=["question", "times"], label="Top unanswered questions")
        endpoints = gr.Dataframe(headers=["endpoint", "requests", "avg latency ms"], label="By endpoint")
    down = gr.Dataframe(headers=["time", "question", "comment"], label="Recent 👎 feedback")
    outputs = [summary, per_day, unanswered, endpoints, down]
    refresh.click(render, days, outputs)
    days.release(render, days, outputs)
