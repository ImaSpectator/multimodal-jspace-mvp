from pathlib import Path


def _source():
    return (Path(__file__).parents[1] / "frontend" / "app.py").read_text()


def test_v10_version_and_prompt_prefill_is_streamlit_safe():
    source = _source()
    assert 'APP_VERSION = "1.4.3-plain-transcript-bilingual-conflicts"' in source
    assert 'def _queue_manual_suggestion' in source
    assert 'st.session_state["manual_chat_prefill"] = suggestion' in source
    assert 'queued_prefill = st.session_state.pop("manual_chat_prefill", None)' in source
    assert 'on_click=_queue_manual_suggestion' in source
    # Regression: do not mutate the instantiated manual_chat_input inside the button body.
    assert 'if st.button(\n                                        L("Use prompt"' not in source


def test_v10_evidence_panel_is_open_and_mode_specific():
    source = _source()
    assert 'evidence_title = {' in source
    assert '"Image Upload": L("Image evidence", "图片证据")' in source
    assert '"Audio Upload": L("Audio evidence", "音频证据")' in source
    assert '"Video Upload": L("Video evidence", "视频证据")' in source
    assert '"Multimodal Mix": L("Multimodal evidence", "多模态证据")' in source
    assert 'with st.expander(evidence_title, expanded=True)' in source
    assert 'Add evidence for this turn (optional)' not in source


def test_v10_scenario_length_hidden_and_researcher_view_settings_gated():
    source = _source()
    assert 'Scenario progress ·' not in source
    assert '**Planned customer turns:**' not in source
    assert '"researcher_view": False' in source
    assert 'Enable Researcher View' in source
    assert 'if _setting("researcher_view"):' in source


def test_v10_share_is_clipboard_only():
    source = _source()
    assert 'def _copy_current_link()' in source
    assert 'navigator.clipboard.writeText' in source
    assert ':material/link:' in source
    assert 'mailto:' not in source
    assert 'Recipient email' not in source


def test_v10_toolbar_has_explicit_full_button_centering():
    source = _source()
    assert 'display:flex!important; align-items:center!important; justify-content:center!important;' in source
    assert 'background:transparent!important; box-shadow:none!important;' in source
    assert 'display:flex!important; align-items:center!important; justify-content:center!important;' in source
    assert 'width:100%!important; height:100%!important;' in source
