from pathlib import Path

import pytest

from backend.app.ai_provider import build_support_prompt, generate_support_reply
from backend.app.engine import infer_text_emotion
from backend.app.scenario_generator import EMOTIONS, generate_manual_context, generate_scenario, list_domains
from backend.app.schemas import ScenarioControls
from backend.app.simulator import manual_customer_turn, new_manual_state, run_full_scenario


DOMAINS = list_domains()


def test_domain_count_expanded():
    assert len(DOMAINS) >= 18


@pytest.mark.parametrize("domain", DOMAINS)
def test_each_domain_generates_realistic_multiturn_scenario(domain):
    scenario = generate_scenario(ScenarioControls(domain=domain, seed=12345))
    assert scenario.domain == domain
    assert 5 <= len(scenario.steps) <= 7
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
        state, _ = run_full_scenario(scenario, capacity_k=3)
        assert state.conflicts
        names = {c.name for c in state.active_concepts}
        assert "authoritative_status" in names
        assert "customer_visible_status" in names or "customer_belief_status" in names
        if found >= 8:
            break
    assert found >= 8


def test_non_conflict_cases_do_not_invent_visual_conflict():
    found = 0
    for seed in range(1, 100):
        scenario = generate_scenario(ScenarioControls(domain="subscription", seed=seed))
        if scenario.expected_conflict:
            continue
        found += 1
        state, _ = run_full_scenario(scenario, capacity_k=5)
        assert not any("Customer-facing evidence" in c.description for c in state.conflicts)
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
    assert state.transcript[0]["role"] == "customer"
    assert state.transcript[1]["role"] == "agent"
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


def test_ai_prompt_contains_jspace_and_profile_without_hidden_ground_truth():
    scenario = generate_scenario(ScenarioControls(domain="travel", seed=12))
    state, _ = run_full_scenario(scenario, capacity_k=5)
    prompt = build_support_prompt(state, scenario.customer_profile, scenario.domain)
    assert "Active JSpace state" in prompt
    assert scenario.customer_profile["relationship"] in prompt
    assert "hidden_ground_truth" not in prompt
    assert "Do not mention JSpace" in prompt


def test_ai_provider_without_key_uses_fallback():
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


def test_frontend_removed_old_controls_and_backend_injection():
    source = Path("frontend/app.py").read_text(encoding="utf-8")
    forbidden = [
        "Run random batch of 8",
        "Difficulty",
        "Cross-modal / cross-source conflict",
        "Backend concept",
        "Backend unavailable",
        "127.0.0.1:8000",
        "JSPACE_API_URL",
    ]
    for text in forbidden:
        assert text not in source
    assert "Manual AI Conversation" in source
    assert "Continue conversation" in source
    assert "OPENAI_API_KEY" in source


def test_requirements_include_openai():
    req = Path("frontend/requirements.txt").read_text(encoding="utf-8")
    assert "openai" in req.lower()


def test_ai_provider_connected_path_with_mock(monkeypatch):
    import sys
    import types

    class FakeResponse:
        output_text = "I checked the current state and I'll take the next concrete step."

    class FakeResponses:
        def create(self, model, input):
            assert model
            assert "Active JSpace state" in input
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.responses = FakeResponses()

    fake_module = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    scenario = generate_scenario(ScenarioControls(domain="delivery", seed=44))
    state, _ = run_full_scenario(scenario, capacity_k=5)
    reply, provider = generate_support_reply(
        state,
        scenario.customer_profile,
        scenario.domain,
        api_key="test-key",
        model="test-model",
    )
    assert "next concrete step" in reply
    assert provider == "OpenAI · test-model"

@pytest.mark.parametrize("domain", DOMAINS)
def test_authoritative_domain_context_is_not_overwritten_by_ambiguous_text(domain):
    scenario = generate_scenario(ScenarioControls(domain=domain, seed=2026))
    state, _ = run_full_scenario(scenario, capacity_k=8)
    domain_concept = next(c for c in state.concepts if c.name == "customer_domain")
    assert domain_concept.value == domain
