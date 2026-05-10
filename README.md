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
- Saved preparation sessions in SQLite

## Tech Stack

- Backend: Python 3.11, FastAPI, Uvicorn
- AI: OpenAI API with structured Pydantic outputs
- Database: SQLite
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
```

## Run Locally

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

SQLite does not need a separate terminal because it runs as a local database file at `data/interview_copilot.db`.

The frontend supports three output language modes:

- `Match job description language`
- `English`
- `Chinese`

Turn on `Demo mode` in the sidebar to generate sample analysis without OpenAI API calls. If `OPENAI_API_KEY` is not configured, the backend automatically uses demo responses.

The full session workflow batches answer generation into one OpenAI call instead of calling the model once per question.

## API Endpoints

- `GET /health`
- `POST /analyze-jd`
- `POST /match-resume`
- `POST /generate-questions`
- `POST /generate-answer`
- `POST /sessions/from-upload`
- `GET /sessions`
- `GET /sessions/{session_id}`

FastAPI docs are available at `http://localhost:8000/docs`.

## Docker

```bash
docker build -t ai-interview-copilot .
docker run --env-file .env -p 8000:8000 ai-interview-copilot
```

## Notes

The answer generator is intentionally conservative: it instructs the model to use only resume evidence and to call out adjacent experience instead of inventing details.
