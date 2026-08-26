# JSpace Live — v0.6.3

Website URL: https://multimodal-jspace-mvp-xl8khikqwpvcxh44x3cjvq.streamlit.app/

## Updates

- Fixed Gemini connection failures caused by invalid sub-10-second manual deadlines.
- Gemini requests now use bounded deadlines and streaming output for faster perceived responses.
- Every support-agent message shows whether it came from **Gemini 3.7 Flash** or the **Backup responder**.
- Backup replies are context-aware and no longer repeat the previous support-agent message verbatim.
- Help, Share, Reset, and Settings use compact controls above the main JSpace Live header.
- Manual chat input sits directly below the conversation.
- Pressing **Enter** submits a customer message.
- **Use suggested prompt** reliably fills the chat input so it can be edited or sent.
- Suggested prompts appear below the chat input.
- Scenario and Manual sessions both support explicit session ending.
- Customer messages appear immediately before the support agent starts generating its reply.
- JSpace continues to track customer affect, patience, trust, satisfaction, conflicts, evidence, and active concepts.

## Speed profiles

### Fast — default

- Gemini 3.7 Flash with low thinking
- 12-second maximum request deadline per attempt
- Up to 2 bounded attempts for transient failures
- Last 4 conversation messages in context
- Concise 1–2 sentence agent replies
- Streaming output so text can appear before the full response finishes

### Balanced

- Gemini 3.7 Flash with low thinking
- 20-second maximum request deadline per attempt
- Last 6 conversation messages in context
- Slightly longer responses and more conversational context

The deadline is only a maximum cap. A successful Gemini response can begin streaming much sooner.

## Run locally

From the project root in PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run frontend/app.py --server.port 8501
```

Then open:

```text
http://localhost:8501
```
