# JSpace Live — v0.8.2.2

## Updates

- Manual Multimodal AI has a redesigned composer directly beneath the conversation window.
- The chat composer is now a real form: **Enter** and **Send** submit the same turn.
- **Use prompt** immediately fills the customer-message box without requiring the user to focus the input first.
- The suggested prompt sits in a compact panel to the right of the composer, away from the Live JSpace workspace.
- Scenario Lab and Manual conversations use a fixed-height **scrollable transcript** and automatically follow the newest turn.
- Scenario Lab shows turn progress so it is clear that the case is continuing rather than frozen.
- Added a global **English / 中文** switch in the top-right toolbar. Navigation, controls, help/settings copy, scenario wording, suggested prompts, generated customer turns, and AI responses follow the selected language.
- Switching language restarts the active practice session so one transcript does not mix English and Chinese generated turns.
- The top-right Help, Share, Reset, and Settings icons use fixed square buttons with explicitly centered Material glyphs.
- Text and screenshots/images use `deepseek/deepseek-v4-flash-vision-exp`.
- Standalone audio uses `hy-asr-3.0-preview` for transcription. Manual mode does not claim to detect raw vocal emotion; affect can only be inferred from typed/transcribed wording.
- Video uses `youtu-vita` for visual + audio-track understanding and converts its observations into JSpace evidence.
- DeepSeek remains the final customer-service responder and reasons over the selected Top-K JSpace concepts from all modalities.
- DeepSeek replies stream with thinking disabled, bounded retries, repeat-response protection, provider labels, and rerun/double-click guards.
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
