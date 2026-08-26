from pathlib import Path
import sys
import types

from backend.app.ai_provider import generate_support_reply
from backend.app.scenario_generator import generate_scenario, list_domains
from backend.app.schemas import ScenarioControls
from backend.app.simulator import (
    apply_manual_customer_message,
    append_agent_reply,
    end_manual_session,
    new_manual_state,
    run_full_scenario,
)


def _install_flaky_google(monkeypatch):
    calls = {"n": 0}

    class FakeResponse:
        text = "I verified the current state and I'm taking the next concrete step now."

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("503 UNAVAILABLE ServerError")
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    class FakeThinkingConfig:
        def __init__(self, **kwargs): self.kwargs = kwargs

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs): self.kwargs = kwargs

    class FakePart:
        @staticmethod
        def from_bytes(data, mime_type):
            return {"data": data, "mime_type": mime_type}

    fake_types = types.ModuleType("google.genai.types")
    fake_types.ThinkingConfig = FakeThinkingConfig
    fake_types.GenerateContentConfig = FakeGenerateContentConfig
    fake_types.Part = FakePart
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = fake_types
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    return calls


def test_scenario_lengths_vary_and_close_normally():
    lengths = set()
    for seed in range(1, 100):
        scn = generate_scenario(ScenarioControls(domain="travel", seed=seed))
        lengths.add(len(scn.steps))
        assert scn.steps[-2].label == "Resolution confirmed"
        assert scn.steps[-1].label == "No other concerns"
        assert "no" in scn.steps[-1].customer_turn.text.lower() or "that's all" in scn.steps[-1].customer_turn.text.lower()
    assert len(lengths) >= 3
    assert min(lengths) >= 7
    assert max(lengths) <= 12


def test_full_scenario_never_ends_unresolved():
    for domain in list_domains():
        scn = generate_scenario(ScenarioControls(domain=domain, seed=20260826))
        state, snapshots = run_full_scenario(scn, capacity_k=4)
        assert state.session_ended
        assert state.session_phase == "ended"
        authoritative = next(c for c in state.concepts if c.name == "authoritative_status")
        assert authoritative.value == "resolved"
        assert not state.conflicts
        assert any(s.session_phase == "resolved" for s in snapshots)


def test_retry_recovers_transient_gemini_error(monkeypatch):
    calls = _install_flaky_google(monkeypatch)
    scn = generate_scenario(ScenarioControls(domain="delivery", seed=9))
    state, _ = run_full_scenario(scn, capacity_k=4)
    reply, provider = generate_support_reply(
        state, scn.customer_profile, scn.domain, api_key="test-key", max_attempts=3
    )
    assert calls["n"] == 2
    assert "next concrete step" in reply
    assert "retry recovered" in provider


def test_satisfaction_changes_with_response_quality():
    state = new_manual_state(capacity_k=4, profile={"trust": 50, "relationship": "neutral"})
    before = state.customer_satisfaction
    apply_manual_customer_message(state, "I'm frustrated and this is still not fixed.")
    append_agent_reply(state, "I can see why this is frustrating. I'll verify the current state and take the next concrete step.", "test")
    assert state.customer_satisfaction > before


def test_manual_end_session_is_explicit():
    state = new_manual_state(capacity_k=4)
    apply_manual_customer_message(state, "I think that's enough for now.")
    end_manual_session(state)
    assert state.session_ended
    assert state.session_phase == "ended"
    assert state.transcript[-1]["role"] == "agent"
    assert "good day" in state.transcript[-1]["text"].lower()


def test_frontend_contains_v06_interaction_requirements():
    source = Path("frontend/app.py").read_text(encoding="utf-8")
    required = [
        "Support Agent is typing", "Suggested customer prompt", "End session", "Satisfaction",
        "PUBLIC_APP_URL", "Email link", "Print this view", "expanded=False",
        "JSpace capacity K\", 3, 6, 4", "prepare_scenario_for_channel", "Multimodal Mix",
        "Customer messages", "on_change=_on_main_tab_change", "Start conversation",
    ]
    for term in required:
        assert term in source
    forbidden = [
        "Agent is composing a response", "Run random batch of 8", "Research MVP",
        "Gemini is connected through", 'expanded=True',
    ]
    for term in forbidden:
        assert term not in source


def test_native_toolbar_and_heading_links_are_hidden():
    source = Path("frontend/app.py").read_text(encoding="utf-8")
    assert '[data-testid="stToolbar"]' in source
    assert "a.anchor-link" in source
    assert "#MainMenu" in source
