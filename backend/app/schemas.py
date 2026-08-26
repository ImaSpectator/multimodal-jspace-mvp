from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field

Modality = Literal["text", "audio", "image", "backend", "derived"]
ConceptStatus = Literal["supported", "disputed", "unresolved"]


class Evidence(BaseModel):
    source: Modality
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Concept(BaseModel):
    id: str
    name: str
    value: str
    sources: list[Modality]
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    task_relevance: float = Field(ge=0.0, le=1.0, default=0.5)
    conflict_importance: float = Field(ge=0.0, le=1.0, default=0.0)
    recency: float = Field(ge=0.0, le=1.0, default=1.0)
    redundancy: float = Field(ge=0.0, le=1.0, default=0.0)
    status: ConceptStatus = "supported"
    pinned: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def score(self) -> float:
        raw = (
            0.32 * self.task_relevance
            + 0.23 * self.confidence
            + 0.22 * self.conflict_importance
            + 0.18 * self.recency
            - 0.05 * self.redundancy
        )
        if self.pinned:
            raw += 1.0
        return round(raw, 4)


class Conflict(BaseModel):
    id: str
    concept_ids: list[str]
    description: str
    severity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)


class CustomerTurn(BaseModel):
    text: str
    audio_tone: Optional[Literal["calm", "neutral", "frustrated", "angry", "uncertain"]] = None


class BackendEvent(BaseModel):
    # Open string on purpose: the automated simulator spans many customer-service domains.
    event_type: str
    value: str
    metadata: dict = Field(default_factory=dict)


class ImageObservation(BaseModel):
    description: str
    concept_name: Optional[str] = None
    concept_value: Optional[str] = None
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    relevance: float = Field(default=0.70, ge=0.0, le=1.0)
    conflict_importance: float = Field(default=0.0, ge=0.0, le=1.0)


class SessionConfig(BaseModel):
    capacity_k: int = Field(default=5, ge=1, le=50)
    preserve_conflicts: bool = True


class SessionState(BaseModel):
    session_id: str
    config: SessionConfig
    concepts: list[Concept] = Field(default_factory=list)
    active_concepts: list[Concept] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    transcript: list[dict] = Field(default_factory=list)
    backend_history: list[dict] = Field(default_factory=list)
    recommended_action: Optional[str] = None
    recommended_action_code: Optional[str] = None
    last_response: Optional[str] = None


class ScenarioControls(BaseModel):
    domain: str = "random"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    include_conflict: Optional[bool] = None
    seed: Optional[int] = None


class ScenarioStep(BaseModel):
    label: str
    customer_turn: Optional[CustomerTurn] = None
    backend_events: list[BackendEvent] = Field(default_factory=list)
    image_observations: list[ImageObservation] = Field(default_factory=list)


class GeneratedScenario(BaseModel):
    scenario_id: str
    domain: str
    title: str
    difficulty: str
    customer_profile: dict
    hidden_ground_truth: dict
    expected_action_code: str
    expected_conflict: bool
    critical_concepts: list[str]
    steps: list[ScenarioStep]
    seed: int


class EvaluationResult(BaseModel):
    action_correct: bool
    conflict_expected: bool
    conflict_detected: bool
    conflict_correct: bool
    critical_evidence_retention: float
    final_active_concepts: list[str]
    expected_action_code: str
    actual_action_code: Optional[str]
    score: float
    notes: list[str] = Field(default_factory=list)


class ScenarioRunResult(BaseModel):
    scenario: GeneratedScenario
    final_state: SessionState
    step_states: list[SessionState]
    evaluation: EvaluationResult
