"""Load versioned prompt files (prompts/<name>_<version>.yaml) and build chat messages."""

from functools import lru_cache
from html import escape

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings
from app.retrieval.vectorstore.base import ScoredChunk


@lru_cache
def load_prompt(name: str, version: str) -> dict:
    path = get_settings().prompts_dir / f"{name}_{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"prompt {path.name} not found (check PROMPT_VERSION)")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def format_context(chunks: list[ScoredChunk]) -> str:
    """Wrap each chunk in a <chunk> tag carrying the exact labels the model must cite.

    Chunk text is escaped so a document containing "</context>" can't break out of the delimiters.
    """
    parts = []
    for sc in chunks:
        c = sc.chunk
        parts.append(
            f'<chunk doc_id="{c.doc_id}" section="{escape(c.section, quote=True)}">\n'
            f"{escape(c.text, quote=False)}\n</chunk>"
        )
    return "\n".join(parts)


def build_answer_messages(question: str, chunks: list[ScoredChunk], version: str) -> list[BaseMessage]:
    p = load_prompt("answer", version)
    msgs: list[BaseMessage] = [SystemMessage(p["system"])]
    for ex in p.get("few_shot") or []:
        msgs.append(HumanMessage(p["user_template"].format(context=ex["context"].strip(), question=ex["question"])))
        msgs.append(AIMessage(ex["answer"].strip()))
    msgs.append(HumanMessage(p["user_template"].format(context=format_context(chunks), question=question)))
    return msgs
