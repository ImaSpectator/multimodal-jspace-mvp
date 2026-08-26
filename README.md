# Multimodal JSpace Customer-Service MVP — v0.5

A single-service Streamlit research MVP for capacity-limited, conflict-aware multimodal customer-service reasoning.

## What changed in v0.5

- Replaced OpenAI with **Gemini 3.7 Flash** (`gemini-3.7-flash`) through the official `google-genai` Python SDK.
- Gemini powers live customer-service replies when `GEMINI_API_KEY` is configured.
- Scenario wording is automatically remixed by Gemini when connected, while controlled ground truth remains unchanged.
- Manual mode accepts real **image, audio, and video** evidence and sends it to Gemini.
- Added channel choices: Text Messages, Voice Call, Video + Voice, and Multimodal Mix.
- Scenario generation now prepares a case first; the conversation starts only when the user presses **Start conversation**.
- New customer/agent turns animate into a phone-style conversation instead of appearing as a finished transcript.
- Recommended next move is surfaced near the top of JSpace.
- Evidence & provenance and Researcher view are expanded by default.
- Added a **Start Here** page explaining modes, JSpace nodes, capacity K, affect intensity, priority, provenance, and all domains.
- Improved responsive emotion display so longer emotional labels do not get cut off.

## Streamlit Community Cloud setup

The app is single-service. You do **not** need Render or FastAPI.

Set the Streamlit entry point to:

```text
frontend/app.py
```

In **Streamlit → App settings → Secrets**, add:

```toml
GEMINI_API_KEY = "your-google-ai-studio-api-key"
GEMINI_MODEL = "gemini-3.7-flash"
```

Do not put your API key in GitHub.

Gemini free-tier availability and rate limits are controlled by Google. The app gracefully falls back to its local simulator if the API key is absent, rate-limited, or a request fails.

## Run locally

From the repository root:

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

For local Gemini calls, either add a `.streamlit/secrets.toml` file (do not commit it) or set the environment variable `GEMINI_API_KEY`.

## Research architecture

```text
Customer text / voice / video / image
             +
Company/backend evidence
             ↓
Candidate concepts
             ↓
Conflict detection
             ↓
Capacity-limited Top-K JSpace
             ↓
Recommended next action
             ↓
Gemini 3.7 Flash support response
```

The JSpace is an external research workspace inspired by limited-capacity global-workspace ideas. It is not an extraction of Gemini's internal hidden state.
