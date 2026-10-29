"""Validated data contracts shared by retrieval, API calls, and the UI."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Category = Literal[
    "Network",
    "Application",
    "Database",
    "Authentication",
    "Infrastructure",
    "Other",
]
Severity = Literal["Low", "Medium", "High", "Critical"]


class EvidenceRecord(BaseModel):
    """An app-created reference to an exact sanitized source excerpt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    kind: Literal["incident_description", "uploaded_log", "knowledge_base"]
    source_name: str
    excerpt: str
    line_start: int | None = None
    line_end: int | None = None
    passage_id: str | None = None
    truncated: bool = False


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class DiagnosticStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class IncidentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    category: Category
    severity: Severity
    observable_facts: list[Finding]
    potential_causes: list[Finding]
    diagnostic_steps: list[DiagnosticStep]
    suggested_resolution: str = Field(min_length=1)
    evidence_limitations: list[str]


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    source_path: str
    chunk_id: str
    text: str
    bm25_score: float
    semantic_similarity: float | None = None
    fused_score: float

    @property
    def source_name(self) -> str:
        return self.source_path

    @property
    def passage_id(self) -> str:
        return self.chunk_id

    @property
    def score(self) -> float:
        return self.fused_score


# Backwards-compatible name for existing code and external portfolio examples.
KnowledgePassage = RetrievalResult
