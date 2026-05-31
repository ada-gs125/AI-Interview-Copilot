from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def start_process(name: str, command: list[str], env: dict[str, str]) -> subprocess.Popen:
    print(f"[dev] starting {name}: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )


def stop_process(name: str, process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    print(f"[dev] stopping {name}")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AI Interview Copilot backend and frontend together.")
    parser.add_argument("--backend-port", default="8000")
    parser.add_argument("--frontend-port", default="8501")
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("API_BASE_URL", f"http://localhost:{args.backend_port}")

    backend = start_process(
        "backend",
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--reload",
            "--host",
            "127.0.0.1",
            "--port",
            args.backend_port,
        ],
        env,
    )
    frontend = start_process(
        "frontend",
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/frontend/streamlit_app.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            args.frontend_port,
            "--browser.gatherUsageStats",
            "false",
        ],
        env,
    )

    print(f"[dev] backend:  http://127.0.0.1:{args.backend_port}")
    print(f"[dev] frontend: http://127.0.0.1:{args.frontend_port}")
    print("[dev] press Ctrl+C to stop both services")

    processes = {"backend": backend, "frontend": frontend}
    try:
        while True:
            for name, process in processes.items():
                code = process.poll()
                if code is not None:
                    print(f"[dev] {name} exited with code {code}")
                    return code
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[dev] shutdown requested")
        return 0
    finally:
        for name, process in processes.items():
            stop_process(name, process)


if __name__ == "__main__":
    raise SystemExit(main())
