"""Accumulate token usage and estimated cost across all LLM calls in one request."""

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage

from app.llm.pricing import token_cost


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    estimated: bool = False  # True if a provider didn't report usage and we approximated it
    calls: list[dict] = field(default_factory=list)

    def add(self, stage: str, model: str, input_tokens: int, output_tokens: int, estimated: bool = False) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        cost = token_cost(model, input_tokens, output_tokens)
        if cost is None:
            self.cost_known = False
        else:
            self.cost_usd += cost
        self.estimated |= estimated
        self.calls.append({"stage": stage, "model": model, "in": input_tokens, "out": output_tokens})

    def add_message(self, stage: str, msg: BaseMessage, prompt_chars: int = 0) -> None:
        """Read usage_metadata from a LangChain AI message; approximate (chars/4) if missing."""
        model = (getattr(msg, "response_metadata", None) or {}).get("model_name", "unknown")
        um = getattr(msg, "usage_metadata", None)
        if um:
            self.add(stage, model, um.get("input_tokens", 0), um.get("output_tokens", 0))
        else:
            self.add(stage, model, prompt_chars // 4, len(str(msg.content)) // 4, estimated=True)

    def merge(self, other: "Usage") -> None:
        for c in other.calls:
            self.add(c["stage"], c["model"], c["in"], c["out"])
        self.estimated |= other.estimated
        self.cost_known &= other.cost_known

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": round(self.cost_usd, 6) if self.cost_known else None,
            "estimated": self.estimated,
            "calls": self.calls,
        }
