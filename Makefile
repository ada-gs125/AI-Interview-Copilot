.PHONY: dev backend frontend test install

install:
	.venv/bin/python -m pip install -r requirements.txt

dev:
	.venv/bin/python scripts/dev.py

backend:
	.venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

frontend:
	API_BASE_URL=http://localhost:8000 .venv/bin/python -m streamlit run app/frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false

test:
	PYTHONPYCACHEPREFIX=/private/tmp/ai-interview-copilot-pycache .venv/bin/python -m pytest -q

