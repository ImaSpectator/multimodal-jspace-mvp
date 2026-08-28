from pathlib import Path

from backend.jspace.conversation_export import build_conversation_pdf, build_plain_transcript_pdf

ROOT = Path(__file__).parents[1]


def test_v145_app_does_not_import_new_pdf_symbol_at_startup():
    source = (ROOT / "frontend" / "app.py").read_text()
    assert "from backend.jspace.conversation_export import build_conversation_pdf" in source
    assert "from backend.jspace.conversation_export import build_plain_transcript_pdf" not in source


def test_v145_stable_export_name_is_the_canonical_implementation():
    source = (ROOT / "backend" / "jspace" / "conversation_export.py").read_text()
    assert "def build_conversation_pdf(" in source
    assert "def build_plain_transcript_pdf(**kwargs)" in source
    assert "return build_conversation_pdf(**kwargs)" in source
    assert callable(build_conversation_pdf)
    assert callable(build_plain_transcript_pdf)
