from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.jspace import ai_provider
from backend.jspace.ai_provider import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_AUDIO_MODEL,
    DEFAULT_VIDEO_MODEL,
    _fallback_reply,
    _request_kwargs,
    analyze_media_for_jspace,
    build_support_prompt,
    enhance_scenario_with_deepseek,
    generate_support_reply,
    probe_deepseek,
    transcribe_audio_with_hyasr,
    stream_support_reply,
)
from backend.jspace.scenario_generator import generate_scenario, list_domains
from backend.jspace.schemas import ScenarioControls
from backend.jspace.simulator import (
    append_agent_reply,
    apply_scenario_customer_step,
    new_scenario_state,
    run_full_scenario,
)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if kwargs.get("stream"):
            chunks = []
            for piece in item:
                chunks.append(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))]))
            return iter(chunks)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item))])


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def _state(domain="travel", seed=7001):
    scenario = generate_scenario(ScenarioControls(domain=domain, seed=seed))
    state = new_scenario_state(scenario, capacity_k=4)
    apply_scenario_customer_step(scenario, state, 0)
    return scenario, state


def test_default_tokenhub_settings():
    assert DEFAULT_MODEL == "deepseek/deepseek-v4-flash-vision-exp"
    assert DEFAULT_AUDIO_MODEL == "hy-asr-3.0-preview"
    assert DEFAULT_VIDEO_MODEL == "youtu-vita"
    assert DEFAULT_BASE_URL == "https://tokenhub.tencentmaas.com/v1"


def test_request_disables_thinking_and_streams_when_requested():
    kwargs = _request_kwargs(model=DEFAULT_MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=100, stream=True)
    assert kwargs["stream"] is True
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kwargs["model"] == DEFAULT_MODEL


def test_support_prompt_emphasizes_latest_customer_and_no_repeat():
    scenario, state = _state()
    append_agent_reply(state, "I am checking that now.", "DeepSeek · test", step_label=scenario.steps[0].label)
    apply_scenario_customer_step(scenario, state, 1)
    prompt = build_support_prompt(state, scenario.customer_profile, scenario.domain)
    assert "Newest customer message" in prompt
    assert "MUST NOT be repeated" in prompt
    assert "Each turn must advance" in prompt


def test_no_key_uses_contextual_local_fallback():
    scenario, state = _state()
    reply, provider = generate_support_reply(state, scenario.customer_profile, scenario.domain, api_key=None)
    assert reply
    assert provider == "Local simulation"


def test_deepseek_connected_nonstream(monkeypatch):
    scenario, state = _state()
    client = FakeClient(["I found the ticketing blocker and I’m checking whether the reissue can be completed now."])
    monkeypatch.setattr(ai_provider, "_cached_client", lambda *args, **kwargs: client)
    reply, provider = generate_support_reply(
        state, scenario.customer_profile, scenario.domain,
        api_key="x", model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL,
    )
    assert "still unresolved" in reply
    assert "checking" not in reply.lower()
    assert provider.startswith("DeepSeek ·")
    call = client.chat.completions.calls[0]
    assert call["extra_body"]["thinking"]["type"] == "disabled"


def test_deepseek_streaming_provider(monkeypatch):
    scenario, state = _state()
    client = FakeClient([["I checked the current booking state. ", "The next step is to complete the ticket reissue."]])
    monkeypatch.setattr(ai_provider, "_cached_client", lambda *args, **kwargs: client)
    rows = list(stream_support_reply(
        state, scenario.customer_profile, scenario.domain,
        api_key="x", model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL,
    ))
    assert rows[-1][2] is True
    assert "still unresolved" in rows[-1][0]
    assert rows[-1][1].startswith(("DeepSeek ·", "Local fallback"))


def test_transient_failure_recovers_on_retry(monkeypatch):
    scenario, state = _state()
    client = FakeClient([RuntimeError("503 Service Unavailable"), "I recovered and checked the specific blocker."])
    monkeypatch.setattr(ai_provider, "_cached_client", lambda *args, **kwargs: client)
    reply, provider = generate_support_reply(
        state, scenario.customer_profile, scenario.domain,
        api_key="x", max_attempts=2,
    )
    assert "specific blocker" in reply
    assert "retry recovered" in provider
    assert len(client.chat.completions.calls) == 2


def test_repetitive_nonstream_reply_is_retried(monkeypatch):
    scenario, state = _state()
    previous = "The authoritative system still shows this as unresolved. I’ll keep the case open and check the specific blocker."
    append_agent_reply(state, previous, "Local fallback")
    apply_scenario_customer_step(scenario, state, 1)
    client = FakeClient([previous, "I checked the next status field and found a different blocker that needs verification."])
    monkeypatch.setattr(ai_provider, "_cached_client", lambda *args, **kwargs: client)
    reply, provider = generate_support_reply(state, scenario.customer_profile, scenario.domain, api_key="x", max_attempts=2)
    assert reply != previous
    assert "different blocker" in reply
    assert "retry recovered" in provider


def test_contextual_fallback_rotates_instead_of_repeating():
    scenario, state = _state("hotel_hospitality")
    first = _fallback_reply(state)
    append_agent_reply(state, first, "Local simulation")
    apply_scenario_customer_step(scenario, state, 1)
    second = _fallback_reply(state, first)
    assert second != first


def test_probe_deepseek(monkeypatch):
    client = FakeClient(["OK"])
    monkeypatch.setattr(ai_provider, "_cached_client", lambda *args, **kwargs: client)
    ok, detail = probe_deepseek(api_key="x")
    assert ok is True
    assert detail == "OK"


def test_probe_deepseek_without_key():
    ok, detail = probe_deepseek(api_key=None)
    assert ok is False
    assert "TOKENHUB_API_KEY" in detail


def test_scenario_rewrite_uses_deepseek(monkeypatch):
    scenario = generate_scenario(ScenarioControls(domain="delivery", seed=7005))
    body = json.dumps({
        "title": "Package scan mismatch",
        "problem_summary": "The customer sees a delivery state that conflicts with the carrier record.",
        "turns": [f"customer turn {i}" for i in range(len(scenario.steps))],
    })
    client = FakeClient([body])
    monkeypatch.setattr(ai_provider, "_cached_client", lambda *args, **kwargs: client)
    updated, provider = enhance_scenario_with_deepseek(scenario, api_key="x")
    assert updated.generated_by_ai is True
    assert updated.title == "Package scan mismatch"
    assert provider.startswith("DeepSeek ·")


def test_image_analysis_uses_image_url_data_uri(monkeypatch):
    client = FakeClient([json.dumps({"items": [{"summary": "The screenshot shows a pending status.", "confidence": 0.91}]})])
    monkeypatch.setattr(ai_provider, "_cached_client", lambda *args, **kwargs: client)
    concepts = analyze_media_for_jspace(
        [{"name": "x.png", "mime_type": "image/png", "data": b"pngbytes"}],
        api_key="x", domain="delivery",
    )
    assert any(c.name == "image_evidence_1" for c in concepts)
    content = client.chat.completions.calls[0]["messages"][0]["content"]
    assert any(part.get("type") == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,") for part in content)


def test_audio_uses_wand_asr_transcription(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({
                "status": "completed",
                "output": {"source": "en", "duration_ms": 2100, "text": "My payment still has not gone through."}
            }).encode("utf-8")

    seen = {}
    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(ai_provider.urllib.request, "urlopen", fake_urlopen)
    concepts = analyze_media_for_jspace(
        [{"name": "call.wav", "mime_type": "audio/wav", "data": b"audio"}],
        api_key="x", domain="payment",
    )
    transcript = next(c for c in concepts if c.name == "audio_transcript")
    assert "payment still" in transcript.value
    assert seen["url"].endswith("/wand/asrproxy/sync_transcribe")
    assert seen["body"]["model"] == DEFAULT_AUDIO_MODEL
    assert seen["body"]["voice_encode_format"] == "wav"
    assert seen["body"]["data"]


def test_video_uses_youtuvita_video_url_data_uri(monkeypatch):
    body = json.dumps({
        "summary": "The router is flashing red while the customer says the connection keeps dropping.",
        "visible_evidence": "A red status light is flashing.",
        "spoken_content": "The connection keeps dropping."
    })
    client = FakeClient([body])
    monkeypatch.setattr(ai_provider, "_cached_client", lambda *args, **kwargs: client)
    concepts = analyze_media_for_jspace(
        [{"name": "router.mp4", "mime_type": "video/mp4", "data": b"videobytes"}],
        api_key="x", domain="internet",
    )
    assert any(c.name == "video_summary" for c in concepts)
    call = client.chat.completions.calls[0]
    assert call["model"] == DEFAULT_VIDEO_MODEL
    content = call["messages"][0]["content"]
    video_part = next(x for x in content if x.get("type") == "video_url")
    assert video_part["video_url"]["url"].startswith("data:video/mp4;base64,")



def test_video_public_url_is_passed_directly(monkeypatch):
    body = json.dumps({"summary": "A checkout error is visible.", "visible_evidence": "Error banner", "spoken_content": ""})
    client = FakeClient([body])
    monkeypatch.setattr(ai_provider, "_cached_client", lambda *args, **kwargs: client)
    concepts = analyze_media_for_jspace(
        [{"name": "linked-video", "mime_type": "video/mp4", "url": "https://example.com/case.mp4"}],
        api_key="x", domain="payment",
    )
    assert any(c.name == "video_summary" for c in concepts)
    content = client.chat.completions.calls[0]["messages"][0]["content"]
    video_part = next(x for x in content if x.get("type") == "video_url")
    assert video_part["video_url"]["url"] == "https://example.com/case.mp4"

def test_scenario_step_is_idempotent_on_rerun():
    scenario = generate_scenario(ScenarioControls(domain="return_refund", seed=7010))
    state = new_scenario_state(scenario, capacity_k=4)
    apply_scenario_customer_step(scenario, state, 0)
    apply_scenario_customer_step(scenario, state, 0)
    customer_rows = [r for r in state.transcript if r.get("role") == "customer"]
    assert len(customer_rows) == 1


def test_agent_append_is_idempotent_on_same_completed_rerun():
    scenario, state = _state("subscription", seed=7011)
    label = scenario.steps[0].label
    append_agent_reply(state, "I checked that.", "DeepSeek · x", step_label=label)
    append_agent_reply(state, "I checked that.", "DeepSeek · x", step_label=label)
    agent_rows = [r for r in state.transcript if r.get("role") == "agent"]
    assert len(agent_rows) == 1


def test_frontend_is_tencent_multimodal_and_has_shortcut_guard():
    source = (Path(__file__).parents[1] / "frontend" / "app.py").read_text()
    assert "TOKENHUB_API_KEY" in source
    assert "TOKENHUB_AUDIO_MODEL" in source
    assert "TOKENHUB_VIDEO_MODEL" in source
    assert "Test DeepSeek connection" in source
    assert "enhance_scenario_with_deepseek" in source
    assert "__jspaceShortcutGuardInstalled" in source
    assert "GEMINI_API_KEY" not in source
    assert "Test Gemini" not in source


def test_top_controls_precede_hero_in_source():
    source = (Path(__file__).parents[1] / "frontend" / "app.py").read_text()
    controls = source.index('with st.container(key="utility_toolbar")')
    hero = source.index('<div class="j-hero">')
    assert controls < hero


def test_v082_toolbar_uses_compact_centered_material_buttons_and_dialogs():
    source = (Path(__file__).parents[1] / "frontend" / "app.py").read_text()
    assert '@st.dialog("Help"' in source
    assert 'def _copy_current_link()' in source
    assert 'navigator.clipboard.writeText' in source
    assert 'mailto:' not in source
    assert '@st.dialog("Settings"' in source
    for icon in (":material/help:", ":material/link:", ":material/refresh:", ":material/settings:"):
        assert icon in source
    assert 'with st.popover("❔"' not in source
    assert 'with st.popover("↗"' not in source
    assert 'with st.popover("⚙"' not in source
    assert 'display:flex!important; align-items:center!important; justify-content:center!important' in source
    assert '.st-key-top_help .stButton > button p' in source
    assert 'key="top_language"' in source


def test_v082_manual_composer_scroll_and_bilingual_controls():
    source = (Path(__file__).parents[1] / "frontend" / "app.py").read_text()
    assert 'with st.form("manual_chat_form"' in source
    assert 'st.form_submit_button' in source
    assert 'use_manual_suggestion_' in source
    assert 'on_click=_queue_manual_suggestion' in source
    assert 'manual_chat_prefill' in source
    assert 'Fill the chat box immediately; then press Enter or Send.' in source
    assert 'height:min(58vh,590px)' in source
    assert 'overflow-y:scroll' in source
    assert 'settings_auto_scroll' in source
    assert 'ui_language' in source
    assert 'Simplified Chinese' in source
    assert 'Scenario progress' not in source
    assert 'MANUAL_MODE_CONFIG' in source
    assert '"Image Upload"' in source and '"Audio Upload"' in source and '"Video Upload"' in source


def test_v082_chinese_prompt_and_fallback_are_language_aware():
    scenario, state = _state()
    prompt = build_support_prompt(state, scenario.customer_profile, scenario.domain, language="Simplified Chinese")
    assert "Reply entirely in Simplified Chinese" in prompt
    fallback = _fallback_reply(state, language="Simplified Chinese")
    assert any(ch in fallback for ch in "我会查核确认客户问题状态解决下一步")


def test_v082_readme_profiles_are_current_and_no_secret_setup_section():
    text = (Path(__file__).parents[1] / "README.md").read_text()
    assert "# JSpace Live — v1.4.3" in text
    assert "**Fast** | 12 seconds | Up to 2 | 4 recent messages | 12 seconds" in text
    assert "**Balanced** | 20 seconds | Up to 2 | 6 recent messages | 20 seconds" in text
    assert "**Concise** — up to 120 output tokens" in text
    assert "**Standard** — up to 180 output tokens" in text
    assert "Streamlit Secrets" not in text
    assert 'TOKENHUB_API_KEY = "YOUR_PRIVATE_TOKENHUB_KEY"' not in text
    assert "hy-asr-3.0-preview" in text
    assert "English / 中文" in text


def test_requirements_use_openai_not_google_genai():
    req = (Path(__file__).parents[1] / "frontend" / "requirements.txt").read_text()
    assert "openai" in req
    assert "google-genai" not in req


def test_secrets_example_is_tokenhub_only():
    text = (Path(__file__).parents[1] / ".streamlit" / "secrets.toml.example").read_text()
    assert "TOKENHUB_API_KEY" in text
    assert DEFAULT_MODEL in text
    assert DEFAULT_AUDIO_MODEL in text
    assert DEFAULT_VIDEO_MODEL in text
    assert "GEMINI" not in text


@pytest.mark.parametrize("domain", list_domains())
@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_every_domain_completes_with_capacity_and_natural_close(domain, k):
    scenario = generate_scenario(ScenarioControls(domain=domain, seed=7100 + k))
    state, snapshots = run_full_scenario(
        scenario,
        capacity_k=k,
        responder=lambda s, p, d: (_fallback_reply(s), "Local simulation"),
    )
    assert state.session_ended is True
    assert state.session_phase == "ended"
    assert all(len(s.active_concepts) <= k for s in snapshots)
    assert scenario.steps[-1].customer_turn.text
    assert any(r.get("role") == "agent" for r in state.transcript)


def test_v09_manual_modes_are_strict_and_suggestions_only_text_multimodal():
    source = (Path(__file__).parents[1] / "frontend" / "app.py").read_text()
    assert '"Image Upload": {"allow_text": False' in source
    assert '"Audio Upload": {"allow_text": False' in source
    assert '"Video Upload": {"allow_text": False' in source
    assert '"Text Messages": {"allow_text": True, "show_suggestion": True' in source
    assert '"Multimodal Mix": {"allow_text": True, "show_suggestion": True' in source
    assert 'Analyze & send evidence' in source


def test_v09_chinese_scenario_fallback_and_dynamic_prompts():
    source = (Path(__file__).parents[1] / "frontend" / "app.py").read_text()
    assert 'Guarantee Chinese customer-facing scenario text' in source
    assert '客户当前遇到一个需要客服核实' in source or '客户看到的信息与公司系统记录存在差异' in source
    assert 'last_agent = next(' in source
    assert 'def _unused_customer_move(state, candidates: list[str])' in source
    assert '请直接帮我把这个问题处理好' in source


def test_v09_status_concepts_use_separate_lane():
    source = (Path(__file__).parents[1] / "frontend" / "app.py").read_text()
    assert 'STATUS_CONCEPT_NAMES' in source
    assert 'Primary task concepts' in source
    assert 'Resolution/status context' in source
    assert 'primary_workspace_concepts' in source
