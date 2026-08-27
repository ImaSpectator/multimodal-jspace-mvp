# JSpace Live — v0.8

## Updates

- Added Tencent multimodal routing while keeping one shared TokenHub API key.
- Text and screenshots/images use `deepseek/deepseek-v4-flash-vision-exp`.
- Standalone audio uses `hy-asr-3.0-preview` for transcription. Manual mode does not claim to detect raw vocal emotion; affect is inferred from wording/transcript only.
- Video uses `youtu-vita` for visual + audio-track understanding and converts its observations into JSpace evidence. The UI supports a public video URL (the officially documented YT-VITA path) and a best-effort local-upload data URL.
- DeepSeek remains the final customer-service responder and reasons over the selected Top-K JSpace concepts from all modalities.
- Media-only turns are supported: a customer can send an attachment without also typing text.
- Provider failures create explicit `analysis unavailable` evidence instead of fabricated image/audio/video observations.
- DeepSeek replies still stream with thinking disabled, bounded retries, repeat-response protection, provider labels, and rerun/double-click guards.
- Help, Share, Reset, and Settings remain compact controls in the top-right above the JSpace Live hero.
- Plain `R` and `C` Streamlit developer shortcuts remain blocked outside text inputs.

## Speed profiles

### Fast
- 12-second DeepSeek maximum per attempt
- Up to 2 bounded attempts
- 4 recent conversation messages
- Concise replies
- DeepSeek thinking disabled
- Streaming output enabled

### Balanced
- 20-second DeepSeek maximum per attempt
- Up to 2 bounded attempts
- 6 recent conversation messages
- More context and slightly longer replies
- DeepSeek thinking disabled
- Streaming output enabled

Audio/video analysis has its own bounded request window because uploaded media can take longer than text generation.

## How to run locally

From the project root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run frontend/app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

Use Streamlit Secrets in deployment; do not put the real key in source control:

```toml
TOKENHUB_API_KEY = "YOUR_PRIVATE_TOKENHUB_KEY"
TOKENHUB_MODEL = "deepseek/deepseek-v4-flash-vision-exp"
TOKENHUB_AUDIO_MODEL = "hy-asr-3.0-preview"
TOKENHUB_VIDEO_MODEL = "youtu-vita"
TOKENHUB_BASE_URL = "https://tokenhub.tencentmaas.com/v1"
```

The same `TOKENHUB_API_KEY` is used for all three routes. The key itself must be authorized for each selected model in TokenHub.
