"""Message router: {intent, urgency} -> which path handles the message.

ROUTER_MODE=prompt     Groq small model, zero- or few-shot prompt (prompts/router_v1.yaml)
ROUTER_MODE=finetuned  local Qwen2.5 + LoRA adapter from Phase 7 (ROUTER_ADAPTER = HF id or path)

If the fine-tuned model can't load (no adapter, peft not installed), we log it
and fall back to the prompt router rather than failing requests.
If the output can't be parsed, we route to `policy_question`: the read-only RAG
path is the safest default (it can't create tickets and refuses when unsure).
"""

import json
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings
from app.llm.client import get_llm
from app.llm.usage import Usage
from app.log import get_logger
from app.observability.tracing import traced
from app.rag.prompts import load_prompt

log = get_logger(__name__)
INTENTS = ("policy_question", "action", "small_talk", "out_of_scope")
URGENCIES = ("low", "medium", "high")


@dataclass
class RouteDecision:
    intent: str
    urgency: str
    source: str  # prompt_zero_shot | prompt_few_shot | finetuned | fallback_default
    latency_ms: float = 0.0
    parsed: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def parse_route(text: str) -> tuple[str, str] | None:
    """Extract a valid {intent, urgency} from model output, or None."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    intent, urgency = data.get("intent"), data.get("urgency", "low")
    if intent not in INTENTS or urgency not in URGENCIES:
        return None
    return intent, urgency


def router_messages(message: str, few_shot: bool) -> list[BaseMessage]:
    p = load_prompt("router", "v1")
    msgs: list[BaseMessage] = [SystemMessage(p["system"])]
    if few_shot:
        for ex in p["few_shot"]:
            msgs.append(HumanMessage(p["user_template"].format(message=ex["message"])))
            msgs.append(AIMessage(json.dumps(ex["label"])))
    msgs.append(HumanMessage(p["user_template"].format(message=message)))
    return msgs


class Router(Protocol):
    def route(self, message: str, usage: Usage | None = None) -> RouteDecision: ...


class PromptRouter:
    def __init__(self, few_shot: bool = True, model_tier: str = "small") -> None:
        self.few_shot = few_shot
        self.model_tier = model_tier

    def route(self, message: str, usage: Usage | None = None) -> RouteDecision:
        t0 = time.perf_counter()
        msgs = router_messages(message, self.few_shot)
        resp = get_llm(self.model_tier, max_tokens=30).invoke(msgs, config=traced(usage))
        if usage is not None:
            usage.add_message("router", resp, prompt_chars=sum(len(str(m.content)) for m in msgs))
        ms = round((time.perf_counter() - t0) * 1000, 1)
        parsed = parse_route(str(resp.content))
        source = "prompt_few_shot" if self.few_shot else "prompt_zero_shot"
        if parsed is None:
            log.warning("router_unparsed", output=str(resp.content)[:200])
            return RouteDecision("policy_question", "low", "fallback_default", ms, parsed=False)
        return RouteDecision(*parsed, source, ms)


class FinetunedRouter:
    """Qwen2.5-0.5B-Instruct + LoRA adapter, greedy decoding, CPU."""

    def __init__(self, adapter: str, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        import torch  # noqa: F401  (fail fast if missing)
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(base_model)
        self.model = PeftModel.from_pretrained(base, adapter).eval()

    def route(self, message: str, usage: Usage | None = None) -> RouteDecision:
        import torch

        t0 = time.perf_counter()
        p = load_prompt("router", "v1")
        chat = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user_template"].format(message=message)},
        ]
        enc = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        ids = enc["input_ids"]
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=24, do_sample=False)
        text = self.tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        parsed = parse_route(text)
        if parsed is None:
            return RouteDecision("policy_question", "low", "fallback_default", ms, parsed=False)
        return RouteDecision(*parsed, "finetuned", ms)


@lru_cache
def get_router() -> Router:
    s = get_settings()
    if s.router_mode == "finetuned":
        try:
            if not s.router_adapter:
                raise ValueError("ROUTER_ADAPTER is empty")
            return FinetunedRouter(s.router_adapter)
        except Exception as e:  # missing adapter / peft not installed / download failed
            log.warning("finetuned_router_unavailable_falling_back_to_prompt", error=str(e))
    return PromptRouter(few_shot=True)
