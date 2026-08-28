from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from backend.jspace.conversation_export import build_conversation_pdf, build_plain_transcript_pdf

ROOT = Path(__file__).parents[1]


def _kwargs():
    return dict(
        transcript=[
            {"role": "customer", "text": "The app still shows the old room."},
            {"role": "agent", "text": "I corrected the reservation record.", "provider": "DeepSeek"},
        ],
        profile={"patience": 70, "trust": 80},
        domain="Hotel Hospitality",
        channel="Text Messages",
        session_id="website-download-test",
        satisfaction=90,
        phase="ended",
        language="English",
        analysis="**Summary**\nThe issue was resolved.",
    )


def test_v144_website_download_uses_stable_plain_transcript_entrypoint():
    source = (ROOT / "frontend" / "app.py").read_text()
    assert "from backend.jspace.conversation_export import build_conversation_pdf" in source
    assert "pdf_bytes = build_conversation_pdf(" in source
    assert 'file_name=f"jspace_{mode}_{state.session_id}_transcript.pdf"' in source
    assert 'key=f"download_pdf_plain_v145_{state.session_id}"' in source
    assert "from backend.jspace.conversation_export import build_plain_transcript_pdf" not in source


def test_v144_both_export_names_render_the_same_plain_transcript():
    direct = build_plain_transcript_pdf(**_kwargs())
    legacy = build_conversation_pdf(**_kwargs())
    for data in (direct, legacy):
        assert data.startswith(b"%PDF")
        text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
        assert "Customer" in text
        assert "The app still shows the old room." in text
        assert "Support Agent" in text
        assert "I corrected the reservation record." in text
        assert "Conversation Analysis" in text


def test_v144_exporter_has_no_legacy_card_or_table_renderer():
    source = (ROOT / "backend" / "jspace" / "conversation_export.py").read_text()
    assert "Table(" not in source
    assert "TableStyle(" not in source
    assert "RoundRect" not in source
    assert "build_plain_transcript_pdf" in source
