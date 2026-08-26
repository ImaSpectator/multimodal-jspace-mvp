# JSpace Live — Multimodal Customer Service v0.6.2

A single-service Streamlit research experience for capacity-limited, conflict-aware multimodal customer-service reasoning.

## What changed in v0.6.2

- Gemini support replies now **stream progressively** instead of waiting for the entire answer before rendering.
- Gemini requests have an explicit bounded timeout; Fast mode uses an 8-second timeout per attempt and compact context.
- The Gemini client is reused across turns so the app does not create a fresh HTTP client every message.
- Gemini 3.7 Flash stays on `thinking_level="low"`; extra sampling parameters were removed from the live reply path.
- Scenario generation has its own bounded timeout and can be switched to instant curated wording in Settings.
- After generating a scenario, the app can automatically scroll to the case/conversation area.
- Help / Share / Reset / Settings are compact controls on the top-right.
- Settings now control AI speed profile, reply length, AI scenario wording, auto-scroll, Gemini connection testing, and printing.
- Main tabs use Streamlit dynamic tabs so hidden tabs do not execute expensive content. Switching tabs invalidates the old generation result.
- Scenario Lab now has an explicit **End session** button.
- Share accepts a recipient email and opens a pre-addressed email containing the public app URL.

## Important note about tab cancellation

Switching tabs triggers a Streamlit rerun, invalidates the old generation result, and prevents hidden-tab work from continuing in the UI. A network request already sent to Gemini may still finish remotely, but Fast mode caps each request so it cannot leave the UI waiting for minutes.

## Gemini setup

Streamlit Community Cloud entry point:

```text
frontend/app.py
```

In **Streamlit → App settings → Secrets**:

```toml
GEMINI_API_KEY = "your-google-ai-studio-api-key"
GEMINI_MODEL = "gemini-3.7-flash"
PUBLIC_APP_URL = "https://your-app.streamlit.app" # optional, used by Share
```

Never commit a real API key to GitHub.

## Run locally

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run frontend/app.py --server.port 8501
```

This version requires Streamlit 1.62+ because it uses dynamic/lazy tabs.

## Speed profiles

### Fast (default)
- 8-second Gemini timeout per attempt
- at most 2 bounded attempts
- last 4 messages in the prompt
- concise 1–2 sentence replies
- streaming output

### Balanced
- 14-second timeout per attempt
- last 6 messages
- slightly longer answers

If the free Gemini tier is overloaded, latency can still vary. The app now fails over quickly instead of waiting for several minutes.

## Testing

The packaged build passed:

- 94 unit/regression/integration tests
- 18 domains × 100 seeds × K=3/4/5/6 = 7,200 full scenario runs
- 0 state/capacity/resolution failures
- Python compilation across frontend, runtime, and tests

## Deployment integrity

The Streamlit frontend imports from the versioned runtime:

```text
backend/jspace_v062/
```

When upgrading, commit both `frontend/app.py` and the entire new `backend/jspace_v062/` directory. Legacy `backend/app/` and older versioned runtimes may remain for compatibility, but the live frontend does not use them.

## Render

Render is not used. This remains a single-service Streamlit app.
