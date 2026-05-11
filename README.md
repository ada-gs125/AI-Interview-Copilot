# AI Interview Copilot

AI Interview Copilot is a local AI SaaS prototype for software and AI interview preparation. Upload a PDF resume, paste a target job description, choose a role type and output language, and the app generates:

- JD skill and topic analysis
- Resume-to-JD match scoring
- Strengths, gaps, positioning strategy, and project talking points
- Technical, project, system design, and behavioral questions
- Personalized answer scripts grounded only in the resume
- English, Chinese, or JD-language-matched outputs
- Demo mode for portfolio walkthroughs without OpenAI API spend
- Markdown and PDF interview prep report exports
- Saved preparation sessions in PostgreSQL
- Async session jobs with progress, step timing, and AI usage metadata

## Tech Stack

- Backend: Python 3.11, FastAPI, Uvicorn
- AI: OpenAI API with structured Pydantic outputs
- Database: PostgreSQL
- Frontend: Streamlit
- PDF parsing: pdfplumber with PyPDF2 fallback
- Deployment: Docker

## Project Structure

```text
app/
|-- main.py
|-- routes/
|-- services/
|-- database/
|-- utils/
`-- frontend/
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API key to `.env`:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_INPUT_COST_PER_1M_TOKENS=0
OPENAI_OUTPUT_COST_PER_1M_TOKENS=0
DATABASE_URL=postgresql://interview_copilot:interview_copilot@localhost:5432/interview_copilot
```

## Run Locally

Start PostgreSQL first:

```bash
make db
```

This starts a local Docker container named `ai-interview-copilot-postgres`.

Start backend and frontend together:

```bash
make dev
```

Or run them separately. Start the backend:

```bash
uvicorn app.main:app --reload
```

Start the Streamlit frontend in another terminal:

```bash
streamlit run app/frontend/streamlit_app.py
```

Open Streamlit at `http://localhost:8501`.

To run only the local Streamlit frontend against the deployed Railway backend, create
`.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`, or set:

```toml
API_BASE_URL = "https://backend-production-b0243.up.railway.app"
```

Then start the frontend:

```bash
make frontend
```

The backend creates the PostgreSQL `sessions` table on startup.

The frontend supports three output language modes:

- `Match job description language`
- `English`
- `Chinese`

Turn on `Demo mode` in the sidebar to generate sample analysis without OpenAI API calls. If `OPENAI_API_KEY` is not configured, the backend automatically uses demo responses.

The full session workflow batches answer generation into one OpenAI call instead of calling the model once per question.

## Branch Workflow

- `main` is connected to the Railway production backend. Pushing to `main` can trigger a production deployment.
- Use `dev` for feature work and test pushes before merging to `main`.
- CI runs on pushes and pull requests for both `main` and `dev`.

## API Endpoints

- `GET /health`
- `POST /analyze-jd`
- `POST /match-resume`
- `POST /generate-questions`
- `POST /generate-answer`
- `POST /sessions/from-upload`
- `POST /sessions/jobs`
- `GET /sessions/jobs/{job_id}`
- `GET /sessions`
- `GET /sessions/{session_id}`

FastAPI docs are available at `http://localhost:8000/docs`.

For production-style workflows, prefer the async job API:

1. `POST /sessions/jobs` with the same multipart form fields as `/sessions/from-upload`.
2. Poll `GET /sessions/jobs/{job_id}` until the status is `succeeded` or `failed`.
3. Read `steps`, `progress_percent`, `usage`, and `result` from the job response.

Set the optional token price environment variables above if you want `usage.estimated_cost_usd`
to be calculated for real OpenAI calls.

## Deploy Streamlit Frontend

The FastAPI backend is already deployed on Railway. Deploy the Streamlit frontend on
Streamlit Community Cloud with these settings:

- Repository: this GitHub repository
- Branch: `main`
- Main file path: `app/frontend/streamlit_app.py`
- Python version: `3.11`
- Secrets:

```toml
API_BASE_URL = "https://backend-production-b0243.up.railway.app"
```

After deployment, the Streamlit app will call the Railway backend directly. Keep
`.streamlit/secrets.toml` local only; it is ignored by git.

## Docker

```bash
docker compose up --build
```

This starts PostgreSQL and the FastAPI backend. Run Streamlit separately if you want the local frontend:

```bash
API_BASE_URL=http://localhost:8000 streamlit run app/frontend/streamlit_app.py
```

You can still build the backend image directly:

```bash
docker build -t ai-interview-copilot .
docker run --env-file .env -p 8000:8000 ai-interview-copilot
```

## Notes

The answer generator is intentionally conservative: it instructs the model to use only resume evidence and to call out adjacent experience instead of inventing details.
