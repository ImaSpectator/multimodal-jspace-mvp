# JSpace Live — Multimodal Customer Service v0.6

A single-service Streamlit research experience for capacity-limited, conflict-aware multimodal customer-service reasoning.

## Highlights in v0.6

- Scenario and Manual conversations now show the **customer message immediately**, then display **“Support Agent is typing…”** while the AI call is running.
- Gemini calls retry transient service errors before using the local recovery responder.
- Scenario conversations vary in length and do not close while the simulated system-of-record is unresolved.
- Every Scenario Lab case progresses through confirmed resolution, an “anything else?” check, a customer “no other concerns” turn, and a final goodbye.
- Manual mode stays open for unlimited turns until the user presses **End session**.
- Manual input sits directly below the live conversation and includes a context-aware **Suggested customer prompt**.
- Added dynamic **Satisfaction** alongside Patience and Trust.
- JSpace K is now intentionally compact: **3–6 concepts**, default 4.
- Text, Voice, Video + Voice, and Multimodal Mix now have different evidence behavior.
- Multimodal modes add visual/audio affect evidence that can support or contradict customer wording and backend state.
- Evidence & provenance and Researcher View are closed by default again.
- Removed the native Streamlit toolbar and replaced it with purpose-built Help, Share, Reset, Settings and Print controls.
- Removed visible provider/model labels from the customer conversation. Provider diagnostics remain in Researcher View.
- Primary actions use a cyan/blue/violet visual system rather than the previous orange accent.
- The product is now labeled **JSpace Live**, not “MVP” in the UI.

## Gemini setup

Streamlit Community Cloud entry point:

```text
frontend/app.py
```

In **Streamlit → App settings → Secrets**:

```toml
GEMINI_API_KEY = "your-google-ai-studio-api-key"
GEMINI_MODEL = "gemini-3.7-flash"

# Optional: fills the custom Share control automatically.
PUBLIC_APP_URL = "https://your-app.streamlit.app"
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

Open `http://localhost:8501`.

## Conversation lifecycle

```text
Customer signal
   ↓ appears immediately
JSpace update / evidence merge
   ↓
Support Agent is typing…
   ↓
Gemini reply (with transient retry)
   ↓
Satisfaction + JSpace update
   ↓
continue until resolved
   ↓
agent asks if there are other concerns
   ↓
customer says no
   ↓
polite goodbye / session ended
```

## Channel behavior

- **Text Messages** — text affect is primary; screenshots can still be attached in Manual mode.
- **Voice Call** — vocal affect is treated as evidence; visual scenario evidence is suppressed.
- **Video + Voice** — vocal/visible affect plus live visual context can enter JSpace.
- **Multimodal Mix** — text/voice, visual/media, and company evidence can all compete in the shared workspace.

## Testing

The v0.6 source includes regression and interaction tests covering:

- 18 domains
- variable conversation length
- explicit resolved/closing lifecycle
- transient Gemini retry recovery
- satisfaction updates
- manual End Session
- compact capacity limits
- conflict surfacing and later resolution
- UI controls and removal of obsolete features

The packaged build was stress-tested across 7,200 full scenario runs (18 domains × 100 seeds × K=3/4/5/6) with zero state/capacity/closure failures.

## v0.6.1 deployment integrity fix

The Streamlit frontend now imports its runtime from `backend/jspace_v061/` rather than the legacy `backend/app/` path. This prevents a partially updated GitHub repository from combining a new frontend with older backend modules. The old `backend/app/` files remain for compatibility/tests but are not used by the deployed Streamlit UI.

If upgrading from v0.6 or earlier, make sure the entire new `backend/jspace_v061/` folder is committed along with `frontend/app.py`.

### Render is no longer used

This is a single-service Streamlit application. If an old Render Web Service is still connected to the GitHub repository, Render may continue attempting deployments and sending failure notifications. Disable Auto-Deploy, suspend, or delete that old Render service. No Render URL or FastAPI service is required by this version.
