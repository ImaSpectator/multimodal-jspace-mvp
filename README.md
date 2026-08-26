# Multimodal JSpace Customer-Service MVP — v0.4

A single-service Streamlit research MVP for a **capacity-limited, conflict-aware shared workspace** in customer service.

## What's new in v0.4

- 18 customer-service domains
- Multi-turn scenarios that unfold one customer turn at a time
- Randomized conflicts (no manual conflict switch)
- No difficulty control or batch-of-8 feature
- No raw backend-concept injection UI
- Rich customer profiles: tenure, relationship, loyalty, recent contacts, value segment, communication style, tech comfort, patience, trust
- 17 emotion labels with continuously varying emotion intensity
- Cleaner futuristic JSpace UI focused on active evidence, conflicts, customer affect, and next action
- Manual JSpace is now a real customer ↔ AI conversation when an OpenAI API key is configured
- Local deterministic fallback keeps the app usable without an API key
- No FastAPI/Render service required

## Run locally

From the project root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run frontend/app.py --server.port 8501
```

Open `http://localhost:8501`.

## Enable real OpenAI replies locally

Create a file at `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-5.6-luna"
```

The key is read only by the server-side Streamlit process. Never commit `secrets.toml` to GitHub.

The app uses the OpenAI **Responses API**. If the API is unavailable, the app falls back to the local simulation so the demo remains usable.

## Enable real OpenAI replies on Streamlit Community Cloud

In your deployed Streamlit app:

1. Open **App settings / Secrets**.
2. Add:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-5.6-luna"
```

3. Save/reboot the app.

Anyone with access to the public Streamlit app can then trigger API calls using your configured key, so monitor API usage/costs and add access controls before broad public distribution.

## Deploy

This version needs only one service:

```text
GitHub → Streamlit Community Cloud → shareable *.streamlit.app URL
```

Use `frontend/app.py` as the Streamlit entry point. `frontend/requirements.txt` contains all cloud dependencies.

## Main domains

Account access, banking fraud, delivery, device support, event ticketing, healthcare appointments, hotel/hospitality, insurance claims, internet, marketplace disputes, payments, returns/refunds, rideshare, software/SaaS, subscriptions, telecom/mobile, travel, and utilities.

## Research behavior

JSpace keeps only the top-K active concepts while preserving evidence involved in important conflicts. Customer emotion is modeled as a label **plus a variable intensity**, rather than as a fixed value. The UI intentionally de-emphasizes a single "accuracy" score and instead exposes:

- active concepts and provenance
- conflict state
- emotional trajectory
- customer relationship context
- recommended next action
- hidden ground truth only in an optional researcher view

## Tests

Run:

```powershell
python -m pytest -q
```
