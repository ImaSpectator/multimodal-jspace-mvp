# JSpace Live — v1.1

## Updates

- Reworked the top-right controls into a **borderless website-style toolbar**. Help, link, reset, and settings icons use transparent hit areas and explicit horizontal/vertical centering; Share now uses a link icon.
- Suggested customer prompts are now **customer moves, not question loops**. Depending on the latest agent reply and JSpace state, suggestions can acknowledge progress, authorize a fix, refuse repeated troubleshooting, ask one targeted question, or close naturally after resolution.
- Manual simulated cases can now **progress to resolution** after the customer authorizes a concrete remediation, preventing unresolved conflicts from remaining stuck forever.
- Chinese mode now localizes **JSpace concept names and common concept values** (status, domain, emotion, root cause, relationship state, etc.). Image/video evidence requests also ask the multimodal model to return evidence in Simplified Chinese while Chinese mode is active.
- **Patience starts at 100** for every new case. It stays unchanged when support is progressing and decays when the interaction stalls, repeats work, falls back, or leaves conflict unresolved for too long.
- **Trust in company** now changes gradually with concrete progress, successful resolution, prolonged conflict, repeated troubleshooting, and fallback responses. Satisfaction remains on its existing scoring logic.
- DeepSeek support responses use a modest temperature for more natural variation while keeping thinking disabled and bounded retries.
- **Settings now persist across tab switches and session resets.** Preferences are stored separately from temporary Streamlit dialog widget keys, so Manual ↔ Scenario navigation no longer resets them.
- Existing v1.0 behavior remains: mode-specific Manual inputs, Hy-ASR audio, YT-VITA video, DeepSeek text/images, bilingual scenario/customer/agent prompts, hidden scenario length, Settings-gated Researcher View, and clipboard-only sharing.
- The global **English / 中文** language switch remains available across Scenario Lab and Manual mode.
- Active model IDs: `deepseek/deepseek-v4-flash-vision-exp` for text/images, `hy-asr-3.0-preview` for standalone audio transcription, and `youtu-vita` for video understanding.

## Speed profiles

Speed and reply length are independent.

| Profile | DeepSeek attempt cap | Attempts | Conversation context | Scenario wording cap |
| --- | ---: | ---: | ---: | ---: |
| **Fast** | 12 seconds | Up to 2 | 4 recent messages | 12 seconds |
| **Balanced** | 20 seconds | Up to 2 | 6 recent messages | 20 seconds |

Both profiles keep DeepSeek thinking disabled and stream output as it arrives.

**Reply length:**

- **Concise** — up to 120 output tokens, targeting about 2 sentences.
- **Standard** — up to 180 output tokens, targeting about 3 sentences.

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
