"""Split the streamed model output into the answer (streamed to the user) and the META JSON.

The model writes:  <answer text with citations>\n---META---\n{"follow_up_suggestions": [...]}

Tokens arrive in arbitrary pieces, so the delimiter can be split across tokens
("---ME" + "TA---"). The splitter holds back just enough characters to never
stream any part of the delimiter to the user.
"""

import json

DELIMITER = "---META---"


class MetaSplitter:
    def __init__(self) -> None:
        self._buf = ""
        self._answer_parts: list[str] = []
        self._meta = ""
        self._in_meta = False

    def feed(self, text: str) -> str:
        """Add model output; return the part that is safe to stream as answer text."""
        if self._in_meta:
            self._meta += text
            return ""
        self._buf += text
        idx = self._buf.find(DELIMITER)
        if idx >= 0:
            emit, self._meta = self._buf[:idx], self._buf[idx + len(DELIMITER) :]
            self._in_meta, self._buf = True, ""
        else:
            # keep a tail that could be the start of the delimiter
            keep = next((k for k in range(len(DELIMITER) - 1, 0, -1) if self._buf.endswith(DELIMITER[:k])), 0)
            emit, self._buf = self._buf[: len(self._buf) - keep], self._buf[len(self._buf) - keep :]
        self._answer_parts.append(emit)
        return emit

    def finish(self) -> tuple[str, dict]:
        """Return (full answer text, parsed META). Missing/broken META -> {}."""
        if self._buf:
            self._answer_parts.append(self._buf)
            self._buf = ""
        answer = "".join(self._answer_parts).strip()
        return answer, parse_meta(self._meta)


def parse_meta(raw: str) -> dict:
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        return {}
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
