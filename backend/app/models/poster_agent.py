"""
PosterAgent — implements the Paper2Poster pipeline.

Refactored from: backend/app/services/paper2poster_service.py

Delegates to run_poster_job.py as an isolated subprocess
(CWD = app/poster_pipeline/).

One-at-a-time serialisation is enforced via _lock / _running_job_id.
is_busy() lets the router return HTTP 409 instead of queueing.

Stage methods (parse_raw, gen_outline_layout, gen_bullet_content, build_pptx)
document the pipeline's conceptual stages; the actual implementation lives
inside run_poster_job.py which is executed as a subprocess.
"""
import json
import logging
import os
import subprocess
import sys
import threading
from typing import Any

from app.config import settings
from app.models.generation_agent import GenerationAgent

logger = logging.getLogger(__name__)


class PosterAgent(GenerationAgent):
    """
    Implements the Paper2Poster pipeline by delegating to run_poster_job.py
    as an isolated subprocess (CWD = app/poster_pipeline/).

    One-at-a-time serialisation is enforced via _lock / _running_job_id.
    is_busy() lets the router return HTTP 409 instead of queueing.

    Stage methods (parse_raw, gen_outline_layout, gen_bullet_content, build_pptx)
    document the pipeline's conceptual stages; the actual implementation lives
    inside run_poster_job.py which is executed as a subprocess.
    """

    job_type: str = "poster"
    _jobs: dict[str, dict[str, Any]] = {}   # isolated from other agents

    # Class-level serialisation primitives
    _lock: threading.Lock = threading.Lock()
    _running_job_id: str | None = None

    _PAPER2POSTER_DIR: str = os.path.abspath(settings.PAPER2POSTER_DIR)
    _OUTPUT_DIR: str       = os.path.abspath(settings.PAPER2POSTER_OUTPUT_DIR)
    _RUNNER_SCRIPT: str    = os.path.join(os.path.abspath(settings.PAPER2POSTER_DIR), "run_poster_job.py")

    @classmethod
    def is_busy(cls) -> bool:
        """Return True if a poster generation job is currently running."""
        return cls._running_job_id is not None

    # --- Conceptual stage methods (subprocess handles the actual execution) ---

    def parse_raw(self, paper_id: str) -> dict:
        """
        Stage 1 — LLM reads full paper text, extracts up to 9 sections.
        Executed inside run_poster_job.py subprocess (parse_raw.py).
        This method documents the interface; call run() to trigger it.
        """
        raise NotImplementedError("parse_raw runs inside the poster subprocess")

    def gen_outline_layout(self, sections: dict) -> dict:
        """
        Stage 2 — LLM assigns sections to panels and figures to panels.
        Executed inside run_poster_job.py subprocess (gen_outline_layout.py).
        """
        raise NotImplementedError("gen_outline_layout runs inside the poster subprocess")

    def gen_bullet_content(self, layout: dict) -> dict:
        """
        Stage 3 — LLM writes bullet points per textbox with visual critic loop.
        Executed inside run_poster_job.py subprocess (gen_poster_content.py).
        """
        raise NotImplementedError("gen_bullet_content runs inside the poster subprocess")

    def build_pptx(self, content: dict) -> str:
        """
        Stage 4 — Generate python-pptx code, execute it, return .pptx path.
        Executed inside run_poster_job.py subprocess (build_poster.py).
        """
        raise NotImplementedError("build_pptx runs inside the poster subprocess")

    # --- Internal thread target ---

    def _run_pipeline(
        self,
        job_id: str,
        pdf_path: str,
        paper_id: str,
        notebook_id: str,
    ) -> None:
        from app.services import qdrant_service

        job_output_dir = os.path.join(self._OUTPUT_DIR, job_id)
        job_tmp_dir    = os.path.join(job_output_dir, "tmp")
        os.makedirs(job_output_dir, exist_ok=True)
        os.makedirs(job_tmp_dir,    exist_ok=True)

        poster_slug = paper_id.replace(" ", "_").replace("/", "_")

        preextracted_path: str | None = None
        try:
            page_texts = qdrant_service.get_all_page_texts(notebook_id, paper_id)
            if page_texts:
                preextracted_path = os.path.join(job_output_dir, "preextracted_pages.json")
                with open(preextracted_path, "w", encoding="utf-8") as _f:
                    json.dump(page_texts, _f, ensure_ascii=False, indent=2)
                logger.info(
                    "Poster job %s: wrote pre-extracted text for %d pages to %s",
                    job_id, len(page_texts), preextracted_path,
                )
            else:
                logger.warning(
                    "Poster job %s: no pre-extracted text found for paper %s; Docling will extract from scratch.",
                    job_id, paper_id,
                )
        except Exception as exc:
            logger.warning("Poster job %s: failed to retrieve pre-extracted text (%s); continuing.", job_id, exc)

        cmd = [
            sys.executable,
            self._RUNNER_SCRIPT,
            "--pdf_path",    pdf_path,
            "--poster_name", poster_slug,
            "--output_dir",  job_output_dir,
            "--model_t",     settings.POSTER_MODEL_T,
            "--model_v",     settings.POSTER_MODEL_V,
            "--tmp_dir",     job_tmp_dir,
        ]
        if preextracted_path is not None:
            cmd += ["--preextracted_text_path", preextracted_path]

        env = os.environ.copy()
        env["OPENROUTER_API_KEY"] = settings.OPENROUTER_API_KEY
        env["PYTHONUTF8"] = "1"

        proc: subprocess.Popen | None = None
        stderr_lines: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self._PAPER2POSTER_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )

            def _drain_stderr() -> None:
                assert proc is not None
                for line in proc.stderr:
                    stripped = line.rstrip()
                    stderr_lines.append(stripped)
                    logger.debug("poster-runner stderr: %s", stripped)

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            assert proc.stdout is not None
            for raw_line in proc.stdout:
                if self._is_cancelled(job_id):
                    proc.kill()
                    logger.info("Poster job %s cancelled; subprocess killed.", job_id)
                    return

                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                try:
                    msg: dict = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.debug("poster-runner (non-JSON stdout): %s", raw_line)
                    continue

                if "error" in msg:
                    self._update(job_id, status="error", error=msg["error"], step="Error")
                    proc.kill()
                    return

                if "progress" in msg:
                    self._update(job_id, progress=msg["progress"], step=msg.get("step", ""))

                if msg.get("done"):
                    pptx_path = msg.get("pptx_path")
                    self._update(
                        job_id,
                        status="done",
                        progress=1.0,
                        step="Done",
                        output_path=pptx_path,
                    )
                    return

            proc.wait()
            stderr_thread.join(timeout=5)

            if self._is_cancelled(job_id):
                return

            if proc.returncode != 0:
                tail = "\n".join(stderr_lines[-15:]) if stderr_lines else "(no stderr)"
                self._update(
                    job_id,
                    status="error",
                    error=f"Runner exited with code {proc.returncode}.\n{tail}",
                    step="Error",
                )

        except Exception as exc:
            logger.exception("Poster pipeline raised an exception: %s", exc)
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

    # --- Public run ---

    def run(
        self,
        notebook_id: str,
        paper_id: str,
        paper_title: str,
        pdf_path: str,
    ) -> str | None:
        """
        Start a poster generation job.
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
            name=f"poster-{job_id[:8]}",
        )
        t.start()
        logger.info("Poster job %s started (paper=%s, notebook=%s).", job_id, paper_id, notebook_id)
        return job_id
