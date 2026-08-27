# JSpace Live — v0.9

## Updates

- Manual Multimodal AI now has five distinct input modes: **Text Messages**, **Image Upload**, **Audio Upload**, **Video Upload**, and **Multimodal Mix**.
- **Text Messages** accepts text only. **Image Upload** accepts images only. **Audio Upload** accepts audio only. **Video Upload** accepts video only. **Multimodal Mix** accepts text, images, audio, and video.
- Suggested prompts appear only in **Text Messages** and **Multimodal Mix**. They now change from turn to turn using the latest AI reply, conflict state, session phase, and conversation progress.
- **Use prompt** fills the customer textbox immediately so you can edit it and press Enter/Send.
- The global **English / 中文** toggle keeps generated customer turns, DeepSeek agent replies, local fallback replies, and suggested prompts in the selected language; Chinese mode uses Simplified Chinese. A deterministic Chinese scenario fallback prevents English curated turns from leaking into Chinese sessions if scenario rewriting fails.
- The main JSpace display prioritizes task-specific concepts. `authoritative_status`, `customer_visible_status`, and `customer_belief_status` are moved into a separate collapsed **Resolution/status context** lane below the primary concepts so they do not dominate the workspace.
- The top-right Help, Share, Reset, and Settings buttons use icon-only controls with strictly centered internal wrappers.
- Scenario and Manual transcripts remain fixed-height, scrollable, and auto-follow the newest turn.
- Text/images use `deepseek/deepseek-v4-flash-vision-exp`; standalone audio uses `hy-asr-3.0-preview`; video uses `youtu-vita`.

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
