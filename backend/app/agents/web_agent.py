"""
WebAgent — implements the Paper2Web pipeline as an isolated subprocess.

Mirrors PosterAgent exactly: one-at-a-time serialisation via _lock /
_running_job_id, JSON-line progress from run_web_job.py subprocess,
and a download endpoint that serves the output ZIP.
"""
import json
import logging
import os
import subprocess
import sys
import threading
from typing import Any

from app.core.config import settings
from app.agents.generation_agent import GenerationAgent

logger = logging.getLogger(__name__)


class WebAgent(GenerationAgent):
    """
    Implements the Paper2Web pipeline by delegating to run_web_job.py
    as an isolated subprocess (CWD = app/agents/).

    One-at-a-time serialisation is enforced via _lock / _running_job_id.
    is_busy() lets the router return HTTP 409 instead of queueing.
    """

    job_type: str = "web"
    _jobs: dict[str, dict[str, Any]] = {}   # isolated from other agents

    # Class-level serialisation primitives
    _lock: threading.Lock = threading.Lock()
    _running_job_id: str | None = None

    _PAPER2WEB_DIR: str = os.path.abspath(settings.PAPER2WEB_DIR)
    _OUTPUT_DIR: str    = os.path.abspath(settings.PAPER2WEB_OUTPUT_DIR)
    _RUNNER_SCRIPT: str = os.path.join(os.path.abspath(settings.PAPER2WEB_DIR), "run_web_job.py")

    @classmethod
    def is_busy(cls) -> bool:
        """Return True if a web generation job is currently running."""
        return cls._running_job_id is not None

    # --- Internal thread target ---

    def _run_pipeline(
        self,
        job_id: str,
        pdf_path: str,
        paper_id: str,
        notebook_id: str,
    ) -> None:
        job_output_dir = os.path.join(self._OUTPUT_DIR, job_id)
        os.makedirs(job_output_dir, exist_ok=True)

        website_slug = paper_id.replace(" ", "_").replace("/", "_")

        cmd = [
            sys.executable,
            self._RUNNER_SCRIPT,
            "--pdf_path",     pdf_path,
            "--website_name", website_slug,
            "--output_dir",   job_output_dir,
            "--model_t",      settings.WEB_MODEL_T,
            "--model_g",      settings.WEB_MODEL_G,
            "--model_v",      settings.WEB_MODEL_V,
            "--model_c",      settings.WEB_MODEL_C,
        ]

        env = os.environ.copy()
        env["OPENROUTER_API_KEY"] = settings.OPENROUTER_API_KEY
        env["PYTHONUTF8"] = "1"

        proc: subprocess.Popen | None = None
        stderr_lines: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self._PAPER2WEB_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )

            def _drain_stderr() -> None:
                assert proc is not None
                for line in proc.stderr:
                    stripped = line.rstrip()
                    stderr_lines.append(stripped)
                    logger.warning("web-runner stderr: %s", stripped)

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            assert proc.stdout is not None
            for raw_line in proc.stdout:
                if self._is_cancelled(job_id):
                    proc.kill()
                    proc.wait()
                    stderr_thread.join(timeout=5)
                    logger.info("Web job %s cancelled; subprocess killed.", job_id)
                    return

                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                try:
                    msg = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.debug("web-runner (non-JSON stdout): %s", raw_line)
                    continue

                if not isinstance(msg, dict):
                    logger.debug("web-runner (non-dict JSON): %s", raw_line)
                    continue

                if "error" in msg:
                    err_msg = msg["error"]
                    tb = msg.get("traceback", "")
                    if tb:
                        logger.error("web-runner traceback:\n%s", tb)
                        err_msg = err_msg + "\n\n" + tb
                    self._update(job_id, status="error", error=err_msg, step="Error")
                    proc.kill()
                    return

                if "progress" in msg:
                    self._update(job_id, progress=msg["progress"], step=msg.get("step", ""))

                if msg.get("done"):
                    zip_path = msg.get("zip_path")
                    self._update(
                        job_id,
                        status="done",
                        progress=1.0,
                        step="Done",
                        output_path=zip_path,
                    )
                    return

            proc.wait()
            stderr_thread.join(timeout=5)

            if self._is_cancelled(job_id):
                return

            if proc.returncode != 0:
                tail = "\n".join(stderr_lines[-40:]) if stderr_lines else "(no stderr)"
                self._update(
                    job_id,
                    status="error",
                    error=f"Runner exited with code {proc.returncode}.\n{tail}",
                    step="Error",
                )

        except Exception as exc:
            logger.exception("Web pipeline raised an exception: %s", exc)
            self._update(job_id, status="error", error=str(exc), step="Error")
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
        finally:
            with self._lock:
                if self.__class__._running_job_id == job_id:
                    self.__class__._running_job_id = None

    # --- Public API ---

    def run(self, **kwargs) -> str:
        """
        Implements GenerationAgent.run() abstract method.
        Delegates to generate_web() with keyword arguments.
        """
        job_id = self.generate_web(
            notebook_id=kwargs.get("notebook_id"),
            paper_id=kwargs.get("paper_id"),
            paper_title=kwargs.get("paper_title"),
            pdf_path=kwargs.get("pdf_path"),
        )
        return job_id or ""  # Return job_id or empty string if busy

    def generate_web(
        self,
        notebook_id: str,
        paper_id: str,
        paper_title: str,
        pdf_path: str,
    ) -> str | None:
        """
        Start a web generation job.
        Returns the job_id, or None if another job is already running
        (caller should respond with HTTP 409).
        """
        with self._lock:
            if self.__class__._running_job_id is not None:
                return None  # busy
            job_id = self._new_job()
            self.__class__._running_job_id = job_id

        t = threading.Thread(
            target=self._run_pipeline,
            args=(job_id, pdf_path, paper_id, notebook_id),
            daemon=True,
            name=f"web-{job_id[:8]}",
        )
        t.start()
        logger.info("Web job %s started (paper=%s, notebook=%s).", job_id, paper_id, notebook_id)
        return job_id
