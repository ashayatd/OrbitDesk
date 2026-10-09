"""Gradio UI mounted at /ui inside the FastAPI app (one process, one port).

The UI calls the same in-process services as the HTTP API, so there is one code
path to test. Tabs: Ask (streamed RAG), Agent (tools + confirm), Voice, Screenshot, Admin.
"""

import json
import uuid
from pathlib import Path

import gradio as gr

from app.agent.actions import ConfirmError, confirm_action
from app.db.seed import EMPLOYEES
from app.errors import InputRejectedError, LLMUnavailableError, RateLimitedError
from app.llm.usage import Usage
from app.multimodal.stt import transcribe
from app.multimodal.vision import IMAGE_TYPES, extract_error
from app.observability.request_log import logged, save_feedback
from app.schemas.ask import AskRequest, AskResponse
from app.services import get_agent_runner, get_rag_pipeline

ROLES = ["employee", "hr", "it_admin"]
EMPLOYEE_CHOICES = [(f"{name} ({eid}, {role})", eid) for eid, name, _, role, _, _ in EMPLOYEES]
ROLE_OF = {eid: role for eid, _, _, role, _, _ in EMPLOYEES}

INTRO = """### OrbitDesk: helpdesk copilot for the fictional Orbit Systems
Answers come only from 12 fictional HR policies and IT runbooks, with verified citations.
Switch **role** to see access control: an *employee* can never retrieve the HR-only salary bands."""


def _details(res: AskResponse) -> str:
    cites = "\n".join(f"- `{c.doc_id}` / {c.section}" for c in res.citations) or "- (none)"
    t = res.timings_ms
    stages = ", ".join(f"{k} {v:.0f} ms" for k, v in t.items() if k != "total")
    cost = f"${res.tokens.cost_usd:.6f}" if res.tokens.cost_usd is not None else "unknown"
    lines = [
        f"**Confidence** {res.confidence:.2f} | **cache** {res.cache} | **prompt** {res.prompt_version}",
        f"**Total** {t.get('total', 0):.0f} ms ({stages})",
        f"**Tokens** {res.tokens.total_tokens} | est. cost {cost}",
    ]
    if res.rewritten_question:
        lines.append(f"**Rewritten as:** {res.rewritten_question}")
    if res.guardrails:
        lines.append(f"**Guardrails:** `{json.dumps(res.guardrails)[:300]}`")
    if res.follow_up_suggestions:
        lines.append("**You could also ask:** " + " / ".join(res.follow_up_suggestions))
    return "\n\n".join(lines) + f"\n\n**Sources**\n{cites}"


def _error_text(e: Exception) -> str:
    if isinstance(e, LLMUnavailableError):
        return f"⚠️ {e.detail}"
    if isinstance(e, RateLimitedError | InputRejectedError):
        return f"⚠️ {e}"
    return f"⚠️ Unexpected error: {type(e).__name__}"


# ---------------- Ask ----------------
def ask_stream(question: str, history: list, role: str, session_id: str):
    """Yields (chat, textbox, details, last request id). gr.skip() leaves an output unchanged."""
    if not question.strip():
        yield history, "", "", gr.skip()
        return
    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": ""}]
    try:
        events = get_rag_pipeline().run(AskRequest(question=question, session_id=session_id, role=role))
        for name, data in logged(events, "/ui/ask", session_id, role, question):
            if name == "token":
                history[-1]["content"] += data
                yield history, "", "", gr.skip()
            elif name == "replace":
                history[-1]["content"] = data["text"]
            elif name == "result":
                history[-1]["content"] = data.answer
                yield history, "", _details(data), data.request_id
    except Exception as e:
        history[-1]["content"] = _error_text(e)
        yield history, "", "", gr.skip()


def send_feedback(rating: str, comment: str, request_id: str | None, session_id: str):
    if not request_id:
        return "Ask a question first.", ""
    save_feedback(request_id, session_id, rating, comment or None)
    return f"Thanks, feedback ({'👍' if rating == 'up' else '👎'}) saved.", ""


# ---------------- Agent ----------------
def agent_send(message: str, history: list, employee_id: str, session_id: str):
    if not message.strip():
        return history, "", None, gr.update(visible=False), ""
    history = history + [{"role": "user", "content": message}]
    try:
        out = get_agent_runner().run(message, session_id, ROLE_OF[employee_id], employee_id)
    except Exception as e:
        history.append({"role": "assistant", "content": _error_text(e)})
        return history, "", None, gr.update(visible=False), ""
    history.append({"role": "assistant", "content": out["reply"]})
    pending = out["pending_action"]
    route = out["route"]
    info = (
        f"**Route** {route.get('intent', '-')} / urgency {route.get('urgency', '-')} ({route.get('source', '-')}) | "
        f"**steps** {out['steps']} | **tools** {[t['tool'] for t in out['tool_calls']]} | "
        f"**tokens** {out['tokens']['total_tokens']} | **{out['timings_ms']['total']:.0f} ms**"
    )
    if pending:
        info += f"\n\n**Pending action:** {pending['summary']}. Press **Confirm** to create it."
    return history, "", pending, gr.update(visible=bool(pending)), info


def agent_confirm(pending: dict | None, history: list, employee_id: str):
    if not pending:
        return history, None, gr.update(visible=False)
    try:
        res = confirm_action(pending["confirmation_id"], employee_id)
        t = res["ticket"]
        msg = f"✅ Ticket **{t['ref']}** {res['status']} ({t['priority']}, {t['category']}): {t['title']}"
    except ConfirmError as e:
        msg = f"⚠️ {e.detail}"
    return history + [{"role": "assistant", "content": msg}], None, gr.update(visible=False)


# ---------------- Voice / Screenshot ----------------
def voice_ask(audio_path: str | None, role: str, session_id: str):
    if not audio_path:
        return "", "Record or upload audio first.", ""
    try:
        p = Path(audio_path)
        tr = transcribe(p.name, p.read_bytes())
        res = _collect(AskRequest(question=tr.text, session_id=session_id, role=role, stream=False))
        return tr.text, res.answer, f"**STT** {tr.latency_ms:.0f} ms ({tr.model})\n\n" + _details(res)
    except Exception as e:
        return "", _error_text(e), ""


def image_ask(image_path: str | None, role: str, session_id: str):
    if not image_path:
        return "", "Upload a screenshot first.", ""
    p = Path(image_path)
    mime = IMAGE_TYPES.get(p.suffix.lower(), "image/png")
    try:
        ex = extract_error(p.read_bytes(), mime, Usage())
        query = ex.search_query or f"How do I fix {ex.error_code} {ex.error_message}"
        res = _collect(AskRequest(question=query, session_id=session_id, role=role, stream=False, department="IT"))
        extracted = f"{ex.error_code} {ex.error_message} ({ex.application}) via {ex.provider}:{ex.model}"
        return extracted, res.answer, _details(res)
    except Exception as e:
        return "", _error_text(e), ""


def _collect(req: AskRequest) -> AskResponse:
    return [
        d for n, d in logged(get_rag_pipeline().run(req), "/ui/multimodal", req.session_id, req.role) if n == "result"
    ][0]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="OrbitDesk") as ui:
        gr.Markdown(INTRO)
        session = gr.State(lambda: uuid.uuid4().hex[:12])
        with gr.Tab("Ask"):
            role = gr.Dropdown(ROLES, value="employee", label="Role")
            chat = gr.Chatbot(height=380, label="OrbitDesk")
            q = gr.Textbox(placeholder="e.g. How many leave days can I carry over?", label="Question")
            details = gr.Markdown()
            gr.Examples(
                [
                    "How many unused leave days can I carry over?",
                    "My VPN says VPN-809 at the hotel",
                    "What is the salary range for a senior engineer?",
                    "Can I bring my dog to the office?",
                ],
                q,
            )
            last_rid = gr.State(None)
            with gr.Row():
                up, down = gr.Button("👍 Helpful"), gr.Button("👎 Not helpful")
                comment = gr.Textbox(placeholder="Optional comment", show_label=False, scale=3)
            fb_status = gr.Markdown()
            q.submit(ask_stream, [q, chat, role, session], [chat, q, details, last_rid])
            up.click(lambda c, r, s: send_feedback("up", c, r, s), [comment, last_rid, session], [fb_status, comment])
            down.click(
                lambda c, r, s: send_feedback("down", c, r, s), [comment, last_rid, session], [fb_status, comment]
            )
        with gr.Tab("Agent"):
            emp = gr.Dropdown(EMPLOYEE_CHOICES, value="E1001", label="Signed in as")
            achat = gr.Chatbot(height=380, label="OrbitDesk agent")
            msg = gr.Textbox(
                placeholder="e.g. How many leave days do I have left? / My laptop was stolen, raise a ticket"
            )
            pending = gr.State(None)
            confirm_btn = gr.Button("Confirm ticket", variant="primary", visible=False)
            ainfo = gr.Markdown()
            msg.submit(agent_send, [msg, achat, emp, session], [achat, msg, pending, confirm_btn, ainfo])
            confirm_btn.click(agent_confirm, [pending, achat, emp], [achat, pending, confirm_btn])
        with gr.Tab("Voice"):
            vrole = gr.Dropdown(ROLES, value="employee", label="Role")
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Ask by voice")
            vbtn = gr.Button("Transcribe and answer", variant="primary")
            vt, va, vd = gr.Textbox(label="Transcript"), gr.Markdown(), gr.Markdown()
            vbtn.click(voice_ask, [audio, vrole, session], [vt, va, vd])
        with gr.Tab("Screenshot"):
            irole = gr.Dropdown(ROLES, value="employee", label="Role")
            img = gr.Image(type="filepath", label="Screenshot of an error")
            ibtn = gr.Button("Read error and search IT runbooks", variant="primary")
            ie, ia, idt = gr.Textbox(label="Extracted error"), gr.Markdown(), gr.Markdown()
            ibtn.click(image_ask, [img, irole, session], [ie, ia, idt])
        with gr.Tab("Admin"):
            from app.ui.admin import build_admin_tab

            build_admin_tab()
    return ui
