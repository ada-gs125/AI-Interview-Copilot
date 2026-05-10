.PHONY: dev backend frontend test install db db-down

install:
	.venv/bin/python -m pip install -r requirements.txt

db:
	docker run --name ai-interview-copilot-postgres -e POSTGRES_DB=interview_copilot -e POSTGRES_USER=interview_copilot -e POSTGRES_PASSWORD=interview_copilot -p 5432:5432 -v ai_interview_copilot_postgres_data:/var/lib/postgresql/data -d postgres:16-alpine || docker start ai-interview-copilot-postgres

db-down:
	docker stop ai-interview-copilot-postgres

dev:
	.venv/bin/python scripts/dev.py

backend:
	.venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

frontend:
	API_BASE_URL=http://localhost:8000 .venv/bin/python -m streamlit run app/frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false

test:
	PYTHONPYCACHEPREFIX=/private/tmp/ai-interview-copilot-pycache .venv/bin/python -m pytest -q
