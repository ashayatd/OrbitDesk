"""Data types shared by ingestion, retrieval and evals."""

from dataclasses import dataclass, field

ROLES = ("employee", "hr", "it_admin")


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    file: str
    title: str
    department: str
    allowed_roles: tuple[str, ...]
    version: str
    updated_at: str

    def __post_init__(self) -> None:
        bad = set(self.allowed_roles) - set(ROLES)
        if bad or not self.allowed_roles:
            raise ValueError(f"invalid allowed_roles for {self.doc_id}: {self.allowed_roles}")


@dataclass
class Section:
    """A run of text under one heading.

    heading_path example: ["Leave Policy", "2. Annual Leave", "2.3 Carry-over"]
    """

    heading_path: list[str]
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str  # what gets embedded / BM25-indexed / shown to the LLM (includes heading line)
    metadata: dict = field(default_factory=dict)

    @property
    def section(self) -> str:
        """Most specific heading, used in citations: [doc_id, section]."""
        return self.metadata.get("section", "")


def role_flags(allowed_roles: tuple[str, ...] | list[str]) -> dict[str, bool]:
    """Chroma metadata filters need scalars, so each role becomes a boolean field."""
    return {f"role_{r}": r in allowed_roles for r in ROLES}
