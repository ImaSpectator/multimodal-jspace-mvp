# JSpace Live — v0.8.1

## Updates

- Tencent multimodal routing uses one shared TokenHub connection across the active model routes.
- Text and screenshots/images use `deepseek/deepseek-v4-flash-vision-exp`.
- Standalone audio uses `hy-asr-3.0-preview` for transcription. Manual mode does not claim to detect raw vocal emotion; affect can only be inferred from the transcript wording.
- Video uses `youtu-vita` for visual + audio-track understanding and converts its observations into JSpace evidence.
- DeepSeek remains the final customer-service responder and reasons over the selected Top-K JSpace concepts from all modalities.
- Media-only turns are supported, and unavailable media analysis is shown explicitly instead of fabricated.
- DeepSeek replies stream with thinking disabled, bounded retries, repeat-response protection, provider labels, and rerun/double-click guards.
- v0.8.1 redesigns Help, Share, Reset, and Settings as a small consistent top-right icon toolbar with modal dialogs instead of oversized popovers.
- Plain `R` and `C` Streamlit developer shortcuts remain blocked outside text inputs.

## Speed profiles

Speed profile and reply length are independent settings.

| Profile | DeepSeek attempt cap | Attempts | Conversation context | Scenario wording cap |
| --- | ---: | ---: | ---: | ---: |
| **Fast** | 12 seconds | Up to 2 | 4 recent messages | 12 seconds |
| **Balanced** | 20 seconds | Up to 2 | 6 recent messages | 20 seconds |

Both profiles keep DeepSeek thinking disabled and stream output as it arrives. The timeout is a maximum per attempt, not a target response time.

**Reply length** is controlled separately:

- **Concise** — up to 120 output tokens, targeting about 2 sentences.
- **Standard** — up to 180 output tokens, targeting about 3 sentences.

Audio and video analysis use their own bounded request windows because media processing can take longer than text generation.

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
