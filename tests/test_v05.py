from pathlib import Path
import json
import sys
import types

import pytest

from backend.app.ai_provider import (
    DEFAULT_MODEL,
    analyze_media_for_jspace,
    build_support_prompt,
    enhance_scenario_with_gemini,
    generate_support_reply,
)
from backend.app.engine import infer_text_emotion
from backend.app.scenario_generator import EMOTIONS, generate_manual_context, generate_scenario, list_domains
from backend.app.schemas import ScenarioControls
from backend.app.simulator import manual_customer_turn, new_manual_state, run_full_scenario


DOMAINS = list_domains()


def _install_fake_google(monkeypatch, response_text: str):
    class FakeResponse:
        text = response_text

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            assert model
            assert contents
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key=None):
            assert api_key == "test-key"
            self.models = FakeModels()

    class FakePart:
        @staticmethod
        def from_bytes(data, mime_type):
            assert data
            assert mime_type
            return {"data": data, "mime_type": mime_type}

    class FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types = types.ModuleType("google.genai.types")
    fake_types.Part = FakePart
    fake_types.ThinkingConfig = FakeThinkingConfig
    fake_types.GenerateContentConfig = FakeGenerateContentConfig

    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = FakeClient
    fake_genai_module.types = fake_types

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai_module
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    return fake_genai_module, fake_types


def test_default_model_is_gemini_37_flash():
    assert DEFAULT_MODEL == "gemini-3.7-flash"


def test_domain_count_expanded():
    assert len(DOMAINS) >= 18


@pytest.mark.parametrize("domain", DOMAINS)
def test_each_domain_generates_realistic_multiturn_scenario(domain):
    scenario = generate_scenario(ScenarioControls(domain=domain, seed=12345))
    assert scenario.domain == domain
    assert scenario.problem_summary
    assert 7 <= len(scenario.steps) <= 12
    assert all(step.customer_turn.text.strip() for step in scenario.steps)
    assert all(0.0 <= step.customer_turn.emotion_intensity <= 1.0 for step in scenario.steps)


@pytest.mark.parametrize("domain", DOMAINS)
def test_each_domain_runs_to_completion(domain):
    scenario = generate_scenario(ScenarioControls(domain=domain, seed=98765))
    state, snapshots = run_full_scenario(scenario, capacity_k=5)
    assert len(snapshots) == len(scenario.steps)
    assert len(state.transcript) == len(scenario.steps) * 2
    assert len(state.active_concepts) <= 5
    assert state.last_response
    assert state.recommended_action


def test_profile_contains_relationship_context():
    scenario = generate_scenario(ScenarioControls(domain="travel", seed=999))
    profile = scenario.customer_profile
    expected = {
        "name", "tenure", "relationship", "loyalty_tier", "previous_contacts_90d",
        "value_segment", "communication_style", "tech_comfort", "patience", "trust", "preferred_channel",
    }
    assert expected.issubset(profile.keys())
    assert 0 <= profile["patience"] <= 100
    assert 0 <= profile["trust"] <= 100


def test_conflict_is_random_across_seeds():
    values = {
        generate_scenario(ScenarioControls(domain="delivery", seed=seed)).expected_conflict
        for seed in range(1, 80)
    }
    assert values == {True, False}


def test_no_difficulty_field_in_controls():
    fields = ScenarioControls.model_fields
    assert "difficulty" not in fields
    assert "include_conflict" not in fields


def test_emotion_variety_across_scenarios():
    observed = set()
    intensities = set()
    for seed in range(1, 160):
        scn = generate_scenario(ScenarioControls(domain="random", seed=seed))
        for step in scn.steps:
            observed.add(step.customer_turn.emotion)
            intensities.add(step.customer_turn.emotion_intensity)
    assert len(observed) >= 12
    assert len(intensities) >= 30
    assert observed.issubset(set(EMOTIONS))


def test_conflict_cases_surface_conflict_when_fully_run():
    found = 0
    for seed in range(1, 80):
        scenario = generate_scenario(ScenarioControls(domain="payment", seed=seed))
        if not scenario.expected_conflict:
            continue
        found += 1
        state, snapshots = run_full_scenario(scenario, capacity_k=3)
        conflict_snaps = [snapshot for snapshot in snapshots if snapshot.conflicts]
        assert conflict_snaps
        names = {c.name for c in conflict_snaps[0].active_concepts}
        assert "authoritative_status" in names
        assert "customer_visible_status" in names or "customer_belief_status" in names
        assert state.session_ended
        assert not state.conflicts
        if found >= 8:
            break
    assert found >= 8


def test_manual_context_and_conversation_work_without_api_key():
    profile, events = generate_manual_context("internet", seed=42)
    state = new_manual_state(capacity_k=5, backend_events=events)
    manual_customer_turn(
        state,
        "I'm frustrated because I already restarted the router and it still keeps dropping.",
        profile=profile,
        domain="internet",
        responder=None,
    )
    assert len(state.transcript) == 2
    assert state.current_emotion in {"frustrated", "impatient", "angry", "anxious"}
    assert state.current_emotion_intensity > 0.5


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This is ridiculous, I've called three times!", "angry"),
        ("I'm really worried this won't be fixed before tomorrow", "anxious"),
        ("I don't understand why the app says something different", "confused"),
        ("Thanks, I really appreciate the help", "appreciative"),
        ("Are you sure that's actually correct??", "skeptical"),
    ],
)
def test_text_emotion_inference(text, expected):
    emotion, intensity = infer_text_emotion(text)
    assert emotion == expected
    assert 0.0 < intensity <= 1.0


def test_gemini_prompt_contains_jspace_and_satisfaction_goal():
    scenario = generate_scenario(ScenarioControls(domain="travel", seed=12))
    state, _ = run_full_scenario(scenario, capacity_k=5)
    prompt = build_support_prompt(state, scenario.customer_profile, scenario.domain, "voice call")
    assert "Active JSpace state" in prompt
    assert scenario.customer_profile["relationship"] in prompt
    assert "customer satisfaction" in prompt.lower()
    assert "Do not mention JSpace" in prompt
    assert "voice call" in prompt


def test_gemini_provider_without_key_uses_fallback():
    scenario = generate_scenario(ScenarioControls(domain="payment", seed=12))
    state, _ = run_full_scenario(scenario, capacity_k=5)
    reply, provider = generate_support_reply(
        state,
        scenario.customer_profile,
        scenario.domain,
        api_key=None,
        fallback="fallback reply",
    )
    assert reply == "fallback reply"
    assert provider == "Local simulation"


def test_gemini_provider_connected_path_with_mock(monkeypatch):
    _install_fake_google(monkeypatch, "I checked the current state. I'll take the next concrete step now.")
    scenario = generate_scenario(ScenarioControls(domain="delivery", seed=44))
    state, _ = run_full_scenario(scenario, capacity_k=5)
    reply, provider = generate_support_reply(
        state,
        scenario.customer_profile,
        scenario.domain,
        api_key="test-key",
        model="gemini-3.7-flash",
        media=[{"data": b"abc", "mime_type": "image/png", "name": "proof.png"}],
    )
    assert "next concrete step" in reply
    assert provider == "Gemini · gemini-3.7-flash"


def test_gemini_scenario_rewrite_with_mock(monkeypatch):
    scenario = generate_scenario(ScenarioControls(domain="subscription", seed=77))
    payload = {
        "title": "Renewal after cancellation",
        "problem_summary": "A long-time subscriber was charged after receiving a cancellation confirmation.",
        "turns": [f"Rewritten customer turn {i+1}?" for i in range(len(scenario.steps))],
    }
    _install_fake_google(monkeypatch, json.dumps(payload))
    updated, provider = enhance_scenario_with_gemini(
        scenario, api_key="test-key", model="gemini-3.7-flash", channel="text messages"
    )
    assert updated.generated_by_ai is True
    assert updated.problem_summary.startswith("A long-time")
    assert updated.steps[0].customer_turn.text.startswith("Rewritten")
    assert provider.startswith("Gemini")


def test_media_analysis_creates_multimodal_concepts(monkeypatch):
    payload = [
        {"source": "image", "summary": "Screenshot shows an error banner.", "confidence": 0.91},
        {"source": "audio", "summary": "Audio contains repeated disconnect tones.", "confidence": 0.84},
        {"source": "video", "summary": "Video shows the device reboot loop.", "confidence": 0.88},
    ]
    _install_fake_google(monkeypatch, json.dumps(payload))
    concepts = analyze_media_for_jspace(
        [
            {"data": b"img", "mime_type": "image/png", "name": "a.png"},
            {"data": b"aud", "mime_type": "audio/mp3", "name": "a.mp3"},
            {"data": b"vid", "mime_type": "video/mp4", "name": "a.mp4"},
        ],
        api_key="test-key",
        domain="device_support",
    )
    assert len(concepts) == 3
    assert {c.sources[0] for c in concepts} == {"image", "audio", "video"}


def test_manual_media_concepts_are_merged():
    profile, events = generate_manual_context("device_support", seed=42)
    state = new_manual_state(capacity_k=8, backend_events=events)
    from backend.app.schemas import Concept, Evidence
    media_concept = Concept(
        id="m1", name="video_evidence_1", value="Device is rebooting repeatedly", sources=["video"],
        evidence=[Evidence(source="video", detail="reboot loop")], confidence=.9, task_relevance=.9,
    )
    manual_customer_turn(
        state, "It keeps restarting like this.", profile=profile, domain="device_support",
        media_concepts=[media_concept], attachments=[{"name": "clip.mp4", "mime_type": "video/mp4"}],
    )
    assert any(c.name == "video_evidence_1" for c in state.concepts)
    assert state.transcript[0]["attachments"][0]["name"] == "clip.mp4"


def test_frontend_has_requested_experience_changes():
    source = Path("frontend/app.py").read_text(encoding="utf-8")
    required = [
        "Start Here", "Scenario Lab", "Manual Multimodal AI", "Text Messages", "Voice Call",
        "Video + Voice", "Multimodal Mix", "Start conversation", "Recommended next move",
        'expanded=False', "GEMINI_API_KEY", "GEMINI_MODEL", "Customer affect",
    ]
    for text in required:
        assert text in source
    forbidden = [
        "OPENAI_API_KEY", "OPENAI_MODEL", "OpenAI connected", "Run random batch of 8",
        "Difficulty", "Cross-modal / cross-source conflict", "Backend concept", "127.0.0.1:8000", "JSPACE_API_URL",
    ]
    for text in forbidden:
        assert text not in source


def test_requirements_use_google_genai_not_openai():
    req = Path("frontend/requirements.txt").read_text(encoding="utf-8").lower()
    assert "google-genai" in req
    assert "openai" not in req


@pytest.mark.parametrize("domain", DOMAINS)
def test_authoritative_domain_context_not_overwritten(domain):
    scenario = generate_scenario(ScenarioControls(domain=domain, seed=2026))
    state, _ = run_full_scenario(scenario, capacity_k=8)
    domain_concept = next(c for c in state.concepts if c.name == "customer_domain")
    assert domain_concept.value == domain

def test_media_affect_can_override_text_affect(monkeypatch):
    payload = [
        {"source": "audio", "summary": "Caller speaks with sharp, frustrated emphasis.", "confidence": 0.93,
         "emotion": "frustrated", "emotion_intensity": 0.88}
    ]
    _install_fake_google(monkeypatch, json.dumps(payload))
    concepts = analyze_media_for_jspace(
        [{"data": b"audio", "mime_type": "audio/wav", "name": "call.wav"}],
        api_key="test-key",
        domain="payment",
    )
    emotion = next(c for c in concepts if c.name == "customer_emotion")
    intensity = next(c for c in concepts if c.name == "emotion_intensity")
    assert emotion.value == "frustrated"
    assert emotion.sources == ["audio"]
    assert intensity.value == "0.88"
