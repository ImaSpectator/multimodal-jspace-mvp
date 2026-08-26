from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .engine import decay_recency, extract_from_backend, extract_from_image, extract_from_turn, merge_concepts, refresh_state
from .scenario_generator import generate_scenario, list_domains
from .schemas import (
    BackendEvent, CustomerTurn, GeneratedScenario, ImageObservation, ScenarioControls, ScenarioRunResult,
    SessionConfig, SessionState,
)
from .simulator import run_generated_scenario
from .store import store

app = FastAPI(title="Multimodal JSpace MVP", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionRequest(BaseModel):
    capacity_k: int = 5
    preserve_conflicts: bool = True


class PinRequest(BaseModel):
    concept_id: str
    pinned: bool = True


class AutoRunRequest(BaseModel):
    controls: ScenarioControls = ScenarioControls()
    capacity_k: int = 5
    preserve_conflicts: bool = True


@app.get("/health")
def health():
    return {"ok": True, "service": "multimodal-jspace-mvp", "version": "0.2.0"}


@app.get("/scenarios/domains")
def scenario_domains():
    return {"domains": list_domains()}


@app.post("/scenarios/generate", response_model=GeneratedScenario)
def create_generated_scenario(controls: ScenarioControls):
    return generate_scenario(controls)


@app.post("/scenarios/autorun", response_model=ScenarioRunResult)
def autorun_scenario(req: AutoRunRequest):
    scenario = generate_scenario(req.controls)
    return run_generated_scenario(
        scenario,
        capacity_k=req.capacity_k,
        preserve_conflicts=req.preserve_conflicts,
    )


@app.post("/scenarios/run", response_model=ScenarioRunResult)
def run_scenario(scenario: GeneratedScenario, capacity_k: int = 5, preserve_conflicts: bool = True):
    return run_generated_scenario(scenario, capacity_k=capacity_k, preserve_conflicts=preserve_conflicts)


@app.post("/sessions", response_model=SessionState)
def create_session(req: CreateSessionRequest):
    return refresh_state(store.create(req.capacity_k, req.preserve_conflicts))


@app.get("/sessions/{session_id}", response_model=SessionState)
def get_session(session_id: str):
    try:
        return store.get(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")


@app.post("/sessions/{session_id}/reset", response_model=SessionState)
def reset_session(session_id: str):
    try:
        return refresh_state(store.reset(session_id))
    except KeyError:
        raise HTTPException(404, "session not found")


@app.put("/sessions/{session_id}/config", response_model=SessionState)
def update_config(session_id: str, config: SessionConfig):
    try:
        state = store.get(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    state.config = config
    return refresh_state(state)


@app.post("/sessions/{session_id}/turn", response_model=SessionState)
def add_turn(session_id: str, turn: CustomerTurn):
    try:
        state = store.get(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    decay_recency(state.concepts)
    state.transcript.append({"role": "customer", "text": turn.text, "audio_tone": turn.audio_tone})
    merge_concepts(state.concepts, extract_from_turn(turn))
    state = refresh_state(state)
    state.transcript.append({"role": "agent", "text": state.last_response})
    return state


@app.post("/sessions/{session_id}/backend-event", response_model=SessionState)
def add_backend_event(session_id: str, event: BackendEvent):
    try:
        state = store.get(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    decay_recency(state.concepts)
    state.backend_history.append(event.model_dump())
    merge_concepts(state.concepts, extract_from_backend(event))
    return refresh_state(state)


@app.post("/sessions/{session_id}/image-observation", response_model=SessionState)
def add_image_observation(session_id: str, obs: ImageObservation):
    try:
        state = store.get(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    decay_recency(state.concepts)
    merge_concepts(state.concepts, extract_from_image(obs))
    return refresh_state(state)


@app.post("/sessions/{session_id}/pin", response_model=SessionState)
def pin_concept(session_id: str, req: PinRequest):
    try:
        state = store.get(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    for concept in state.concepts:
        if concept.id == req.concept_id:
            concept.pinned = req.pinned
            return refresh_state(state)
    raise HTTPException(404, "concept not found")
