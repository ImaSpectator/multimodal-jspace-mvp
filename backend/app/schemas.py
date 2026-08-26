from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

Modality = Literal["text", "audio", "image", "video", "backend", "derived"]
ConceptStatus = Literal["supported", "disputed", "unresolved"]
Emotion = Literal[
    "calm", "neutral", "curious", "hopeful", "appreciative", "satisfied", "relieved",
    "uncertain", "confused", "anxious", "disappointed", "frustrated", "angry", "impatient",
    "skeptical", "distressed", "embarrassed",
]


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
            0.34 * self.task_relevance
            + 0.22 * self.confidence
            + 0.24 * self.conflict_importance
            + 0.17 * self.recency
            - 0.03 * self.redundancy
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
    emotion: Emotion = "neutral"
    emotion_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    nonverbal_cue: Optional[str] = None
    affect_source: Literal["text", "audio", "video"] = "audio"


class BackendEvent(BaseModel):
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
    capacity_k: int = Field(default=5, ge=2, le=20)
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
    current_emotion: Optional[Emotion] = None
    current_emotion_intensity: float = 0.0


class ScenarioControls(BaseModel):
    domain: str = "random"
    seed: Optional[int] = None


class ScenarioStep(BaseModel):
    label: str
    customer_turn: CustomerTurn
    backend_events: list[BackendEvent] = Field(default_factory=list)
    image_observations: list[ImageObservation] = Field(default_factory=list)


class GeneratedScenario(BaseModel):
    scenario_id: str
    domain: str
    title: str
    problem_summary: str = ""
    customer_profile: dict
    hidden_ground_truth: dict
    expected_conflict: bool
    critical_concepts: list[str]
    steps: list[ScenarioStep]
    seed: int
    generated_by_ai: bool = False


class ConversationProgress(BaseModel):
    scenario: GeneratedScenario
    state: SessionState
    next_step_index: int = 0
    finished: bool = False
