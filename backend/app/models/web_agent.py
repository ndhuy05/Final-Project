"""
WebAgent — scaffold for the Paper2Web pipeline.

Implements the Paper2Web pipeline: paper → interactive web application bundle.

This agent is a scaffold — the pipeline stages are not yet implemented.
Mirrors CodeAgent's structure (threading, in-memory job store).
"""
import logging
import threading
from typing import Any

from openai import OpenAI

from app.config import settings
from app.models.generation_agent import GenerationAgent

logger = logging.getLogger(__name__)


class WebAgent(GenerationAgent):
    """
    Implements the Paper2Web pipeline: paper → interactive web application bundle.

    This agent is a scaffold — the pipeline stages are not yet implemented.
    Mirrors CodeAgent's structure (threading, in-memory job store).
    """

    job_type: str = "web"
    _jobs: dict[str, dict[str, Any]] = {}   # isolated from other agents

    def __init__(self) -> None:
        self._sync_client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )
        return self._sync_client

    def plan(self, paper: dict) -> dict:
        """
        Stage 1 — LLM reads paper, plans web app structure
        (pages, components, data, interactivity).
        """
        raise NotImplementedError("WebAgent.plan not yet implemented")

    def generate_app(self, plan: dict) -> dict[str, str]:
        """
        Stage 2 — LLM generates HTML / CSS / JS files from the plan.
        Returns a dict of {filename: content}.
        """
        raise NotImplementedError("WebAgent.generate_app not yet implemented")

    def package_bundle(self, files: dict[str, str]) -> str:
        """
        Stage 3 — Compress generated files into a ZIP bundle and return its path.
        """
        raise NotImplementedError("WebAgent.package_bundle not yet implemented")

    def _run_pipeline(self, job_id: str, **kwargs: Any) -> None:
        """Placeholder pipeline — sets job to error until implemented."""
        self._update(
            job_id,
            status="error",
            error="WebAgent pipeline is not yet implemented.",
            step="Not implemented",
        )

    def run(self, notebook_id: str, paper_id: str, **kwargs: Any) -> str:
        """
        Start a web generation job.
        Currently returns a job that immediately errors (pipeline not implemented).
        """
        job_id = self._new_job()
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(job_id,),
            kwargs={"notebook_id": notebook_id, "paper_id": paper_id, **kwargs},
            daemon=True,
            name=f"web-{job_id[:8]}",
        )
        thread.start()
        return job_id
