"""Generate ~500 labelled router examples with the LLM, then split train / held-out test.

    uv run python finetune/generate_router_data.py          # needs GROQ_API_KEY; resumable

Labels come from construction: each request asks the model for N messages in ONE
(intent, urgency) cell, with varied personas and writing styles, so the label is
known without a second labelling pass. Exact and near-duplicate messages are removed.

Outputs:
  finetune/data/raw.jsonl          every generated example (checkpoint; reruns resume)
  finetune/data/train.jsonl        ~440 examples
  finetune/data/test.jsonl         60 held-out examples, stratified by cell
  finetune/data/test_review.csv    the test set as a spreadsheet: REVIEW THESE BY HAND
                                   (fix wrong labels in test.jsonl before comparing routers)

Caveat: the labels are written by gpt-oss-120b, the same family as the Groq prompt
baselines. That can favour the prompt-only router; hand review of the test set reduces it.
"""

import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.llm.client import get_llm  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
SEED = 13
TEST_SIZE = 60

# (intent, urgency, target count, description given to the generator)
CELLS = [
    ("policy_question", "low", 80, "questions about what an HR/IT/finance policy says or how to do something routine"),
    ("policy_question", "medium", 60, "how-to questions where something is degraded or a deadline is near"),
    ("policy_question", "high", 40, "how-to questions where the employee is fully blocked or there's a security scare"),
    ("action", "low", 70, "requests to check MY leave balance, check a ticket status, or open a routine IT ticket"),
    ("action", "medium", 50, "requests to open a ticket for a degraded issue with a workaround, or soon-due problem"),
    ("action", "high", 40, "urgent requests to open a ticket: stolen laptop, phishing click, outage, fully blocked"),
    ("small_talk", "low", 70, "greetings, thanks, chit-chat, questions about the assistant itself"),
    ("out_of_scope", "low", 60, "unrelated requests: general knowledge, coding help, recipes, jokes, personal advice"),
    (
        "out_of_scope",
        "low",
        30,
        "harmful or forbidden asks: hack a colleague's account, get others' salaries, bypass security",
    ),
]
STYLES = [
    "short and casual, lowercase, maybe a typo",
    "polite and formal, full sentences",
    "stressed, with exclamation marks",
    "non-native English speaker, simple words",
    "very brief, 3-6 words",
    "detailed, mentions an error code or ticket id like TCK-1234",
]
PERSONAS = ["engineer", "sales rep", "new joiner", "manager", "HR specialist", "finance analyst", "IT admin", "intern"]

PROMPT = """Write {n} different messages that employees of a company called Orbit Systems might send
to OrbitDesk, the internal HR/IT helpdesk chatbot.
All messages must belong to this category: {desc}.
Writer persona: {persona}. Writing style: {style}.
Company context: leave policy, travel & expenses, code of conduct, work from home, VPN (OrbitConnect, errors like
VPN-809), passwords and MFA (OrbitAuth), laptops, onboarding, incident priorities P1-P4, software requests, tickets
(TCK-xxxx), leave balances.
Vary topics, length and wording. Do not number them.
Return ONLY a JSON array of strings."""


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def load_raw() -> list[dict]:
    p = DATA / "raw.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def generate() -> list[dict]:
    rows = load_raw()
    seen = {norm(r["message"]) for r in rows}
    llm = get_llm("main", temperature=1.0, max_tokens=1500)
    rng = random.Random(SEED)
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "raw.jsonl").open("a", encoding="utf-8") as out:
        for intent, urgency, target, desc in CELLS:

            def have(intent=intent, urgency=urgency, desc=desc) -> int:
                return sum(r["intent"] == intent and r["urgency"] == urgency and r["desc"] == desc for r in rows)

            attempts = 0
            while have() < target and attempts < 20:
                attempts += 1
                n = min(10, target - have())
                prompt = PROMPT.format(n=n, desc=desc, persona=rng.choice(PERSONAS), style=rng.choice(STYLES))
                resp = llm.invoke([SystemMessage("You write realistic synthetic test data."), HumanMessage(prompt)])
                m = re.search(r"\[.*\]", str(resp.content), re.S)
                try:
                    msgs = json.loads(m.group(0)) if m else []
                except json.JSONDecodeError:
                    continue
                for msg in msgs:
                    if not isinstance(msg, str) or not 3 <= len(msg) <= 400 or norm(msg) in seen:
                        continue
                    seen.add(norm(msg))
                    row = {"message": msg.strip(), "intent": intent, "urgency": urgency, "desc": desc}
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                print(f"{intent}/{urgency}: {have()}/{target}")
    return rows


def split(rows: list[dict]) -> None:
    rng = random.Random(SEED)
    by_cell: dict[tuple, list] = {}
    for r in rows:
        by_cell.setdefault((r["intent"], r["urgency"]), []).append(r)
    test, train = [], []
    for _cell, items in sorted(by_cell.items()):
        rng.shuffle(items)
        k = max(1, round(TEST_SIZE * len(items) / len(rows)))
        test += items[:k]
        train += items[k:]
    while len(test) > TEST_SIZE:  # rounding can overshoot by one or two
        train.append(test.pop())
    clean = lambda r: {"message": r["message"], "intent": r["intent"], "urgency": r["urgency"]}  # noqa: E731
    for name, data in (("train", train), ("test", test)):
        rng.shuffle(data)
        (DATA / f"{name}.jsonl").write_text("\n".join(json.dumps(clean(r)) for r in data) + "\n", encoding="utf-8")
    with (DATA / "test_review.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["message", "intent", "urgency", "label_ok (y/n)", "corrected_intent", "corrected_urgency"])
        for r in test:
            w.writerow([r["message"], r["intent"], r["urgency"], "", "", ""])
    print(f"train={len(train)} test={len(test)} -> {DATA}")


def main() -> None:
    if not get_settings().groq_api_key:
        sys.exit("GROQ_API_KEY is not set: data generation needs the LLM.")
    split(generate())


if __name__ == "__main__":
    main()
