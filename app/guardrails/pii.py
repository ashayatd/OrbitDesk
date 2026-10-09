"""Mask PII in user input before it reaches an LLM, a cache key, a log line or memory.

Masked: email addresses, phone numbers, ID-like numbers (SSN pattern, card-like
digit runs, long national-ID-style numbers).
Deliberately NOT masked, because the helpdesk needs them: ticket ids (TCK-1002),
employee ids (E1001), error codes (VPN-809, 0x0000007E), amounts and dates.
"""

import re
from dataclasses import dataclass, field

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Payment-card-like: 13-19 digits in groups (checked before phones, which are shorter)
CARD = re.compile(r"(?<![\w-])\d{4}(?:[ -]?\d{3,4}){2,4}(?:[ -]?\d{1,3})?(?![\w-])")
# SSN-style 123-45-6789
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Phones: optional +country, then 9-14 digits possibly separated by spaces, dots, dashes or (area code)
PHONE = re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]?\d{2,4}){2,4}(?![\w-])")
# Long digit runs (8+ digits, optional single spaces/dashes): card numbers, passport/national ids
LONG_ID = re.compile(r"(?<![\w-])\d(?:[ -]?\d){7,}(?![\w-])")


@dataclass
class MaskResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def masked(self) -> bool:
        return bool(self.counts)


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{2}[/.-]\d{2}[/.-]\d{4}$")


def _digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


def mask_pii(text: str) -> MaskResult:
    counts: dict[str, int] = {}

    def sub(pattern: re.Pattern, label: str, s: str, min_digits: int = 0, max_digits: int = 99) -> str:
        def repl(m: re.Match) -> str:
            n = _digits(m.group(0))
            if n < min_digits or n > max_digits or _DATE.match(m.group(0)):
                return m.group(0)
            counts[label] = counts.get(label, 0) + 1
            return f"[{label}]"

        return pattern.sub(repl, s)

    out = sub(EMAIL, "EMAIL", text)
    out = sub(CARD, "ID_NUMBER", out, min_digits=13, max_digits=19)
    out = sub(SSN, "ID_NUMBER", out)
    out = sub(PHONE, "PHONE", out, min_digits=9)  # dates/years/amounts have fewer digits
    out = sub(LONG_ID, "ID_NUMBER", out, min_digits=8)
    return MaskResult(out, counts)
