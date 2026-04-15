"""
GenerationAgent — abstract base class for all generation agents.

Subclasses (CodeAgent, PosterAgent, WebAgent) each live in their own module:
  backend/app/models/code_agent.py
  backend/app/models/poster_agent.py
  backend/app/models/web_agent.py

GenerationAgent defines the common job-lifecycle interface and an in-memory
class-level job store that is isolated per subclass (each subclass declares its
own _jobs: dict = {} class attribute, shadowing this one).

Heavy work runs in daemon threading.Thread so it never blocks the FastAPI
async event loop.
"""
import abc
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GenerationAgent — abstract base
# ---------------------------------------------------------------------------

class GenerationAgent(abc.ABC):
    """
    Shared interface for all three generation agents.
    Manages a class-level in-memory job store and exposes run / cancel / get_job.
    """

    job_type: str = ""  # overridden by each subclass

    # Class-level job store — each subclass shadows this with its own dict.
    _jobs: dict[str, dict[str, Any]] = {}

    # --- Job lifecycle helpers ---

    def _new_job(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": "running",
            "progress": 0.0,
            "step": "Starting\u2026",
            "output_path": None,
            "error": None,
            "cancelled": False,
        }
        return job_id

    def _update(self, job_id: str, **kwargs: Any) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)

    def _is_cancelled(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        return job is not None and job.get("cancelled", False)

    # --- Public API ---

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return the job dict or None if the job_id is unknown."""
        return self._jobs.get(job_id)

    def get_progress(self, job_id: str) -> float:
        """Return current progress (0.0–1.0), or -1.0 if job not found."""
        job = self._jobs.get(job_id)
        return job["progress"] if job else -1.0

    def cancel(self, job_id: str) -> bool:
        """
        Request cancellation of a running job.
        Returns True if the job existed and was running.
        The thread checks _is_cancelled() between LLM calls and exits cleanly.
        """
        job = self._jobs.get(job_id)
        if job and job["status"] == "running":
            job["cancelled"] = True
            job["status"] = "cancelled"
            job["step"] = "Cancelled"
            return True
        return False

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> str:
        """Start the pipeline in a background thread and return the job_id immediately."""
