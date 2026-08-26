# Multimodal JSpace — Automated Customer Service MVP v0.3

This edition is intentionally **single-service**. The Streamlit app imports the JSpace engine, scenario generator, simulator, and evaluator directly. **FastAPI and Render are not required to run or deploy the MVP.**

## Local run (Windows)

From the project root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m streamlit run frontend/app.py
```

Then open `http://localhost:8501`.

On future runs, you only need to activate the environment and run Streamlit. No Uvicorn terminal is needed.

## Streamlit Community Cloud deployment

1. Upload/commit this repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository.
3. Use `frontend/app.py` as the Main file path.
4. Deploy.

There are **no backend URL secrets** and no Render service to configure.

## Project structure

- `frontend/app.py` — Streamlit UI and direct orchestration
- `backend/app/engine.py` — JSpace concept extraction, ranking, conflicts, actions
- `backend/app/scenario_generator.py` — multi-domain controlled scenario generation
- `backend/app/simulator.py` — automatic scenario execution
- `backend/app/evaluator.py` — research metrics and scoring
- `backend/app/schemas.py` — shared data models
- `tests/` — engine and scenario tests

The `backend/` name remains for code organization only; it is imported as a Python package by Streamlit. You do not deploy it separately.
