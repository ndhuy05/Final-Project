"""
CodeAgent — implements the 3-stage Paper2Code pipeline.

Refactored from: backend/app/services/paper2code_service.py

Stages:
    1. plan()          — overall plan, architecture, task list, config.yaml
    2. analyze()       — per-file logic analysis
    3. generate_code() — file-by-file code generation
    4. package_zip()   — ZIP the repo directory and return the path

Runs in a daemon thread via run(); never blocks the event loop.
Uses a synchronous OpenAI client (blocking calls are safe inside threads).
"""
import json
import logging
import os
import re
import shutil
import threading
from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.agents.generation_agent import GenerationAgent

logger = logging.getLogger(__name__)


class CodeAgent(GenerationAgent):
    """
    Implements the 3-stage Paper2Code pipeline:
        1. plan()         — overall plan, architecture, task list, config.yaml
        2. analyze()      — per-file logic analysis
        3. generate_code()— file-by-file code generation
        4. package_zip()  — ZIP the repo directory and return the path

    Runs in a daemon thread via run(); never blocks the event loop.
    Uses a synchronous OpenAI client (blocking calls are safe inside threads).
    """

    job_type: str = "code"
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

    def _call(self, messages: list[dict], job_id: str = "") -> str:
        if job_id and self._is_cancelled(job_id):
            raise InterruptedError("Job cancelled")
        client = self._get_client()
        completion = client.chat.completions.create(
            model=settings.PAPER2CODE_CODE_MODEL,
            messages=messages,
        )
        return completion.choices[0].message.content or ""

    # --- Static parsing helpers ---

    @staticmethod
    def _content_to_json(data: str) -> dict:
        clean = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()
        clean = re.sub(r'(".*?"),\s*#.*', r'\1,', clean)
        clean = re.sub(r',\s*\]', ']', clean)
        clean = re.sub(r'\n\s*', '', clean)
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass
        pattern = r'"Logic Analysis":\s*(\[[\s\S]*?\])\s*,\s*"Task list":\s*(\[[\s\S]*?\])'
        match = re.search(pattern, data)
        if match:
            return {
                "Logic Analysis": json.loads(match.group(1)),
                "Task list": json.loads(match.group(2)),
            }
        return {}

    @staticmethod
    def _extract_code(content: str) -> str:
        pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
        code = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
        return code[0] if code else content

    @staticmethod
    def _extract_yaml(content: str) -> str:
        match = re.search(r"```yaml\n(.*?)\n```", content, re.DOTALL)
        if match:
            return match.group(1)
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        match = re.search(r"```yaml\n(.*?)\n```", content, re.DOTALL)
        return match.group(1) if match else ""

    # --- Paper text retrieval ---

    @staticmethod
    def _fetch_paper_text(notebook_id: str, paper_id: str, page_count: int) -> str:
        from app.services import qdrant_service
        pages = {}
        for page_num in range(1, page_count + 1):
            text = qdrant_service.get_page_text(notebook_id, paper_id, page_num)
            if text:
                pages[page_num] = text
        if not pages:
            raise ValueError("No page text found in Qdrant for this paper.")
        lines = [f"=== Page {pg} ===\n{pages[pg]}" for pg in sorted(pages.keys())]
        return "\n\n".join(lines)

    # --- Public stage methods (DESIGN.md API) ---

    def plan(self, paper_content: str, output_dir: str, job_id: str, step_counter: list) -> list[str]:
        """
        Stage 1 — 4 sequential LLM calls:
        overall plan → architecture → task/logic design → config.yaml.
        Returns [plan_text, arch_text, logic_text, config_text].
        """
        os.makedirs(output_dir, exist_ok=True)

        plan_msg = [
            {"role": "system", "content": (
                "You are an expert researcher and strategic planner with a deep understanding of "
                "experimental design and reproducibility in scientific research. "
                "You will receive a research paper as plain text. "
                "Your task is to create a detailed and efficient plan to reproduce the experiments "
                "and methodologies described in the paper. "
                "This plan should align precisely with the paper's methodology, experimental setup, "
                "and evaluation metrics."
            )},
            {"role": "user", "content": (
                f"## Paper\n{paper_content}\n\n"
                "## Task\n"
                "1. We want to reproduce the method described in the attached paper.\n"
                "2. The authors did not release any official code, so we have to plan our own implementation.\n"
                "3. Before writing any Python code, please outline a comprehensive plan that covers:\n"
                "   - Key details from the paper's **Methodology**.\n"
                "   - Important aspects of **Experiments**, including dataset requirements, "
                "experimental settings, hyperparameters, or evaluation metrics.\n"
                "4. The plan should be as **detailed and informative** as possible to help us write the final code later.\n\n"
                "## Requirements\n"
                "- You don't need to provide the actual code yet; focus on a **thorough, clear strategy**.\n"
                "- If something is unclear from the paper, mention it explicitly.\n\n"
                "## Instruction\n"
                "The response should give us a strong roadmap, making it easier to write the code later."
            )},
        ]

        file_list_msg = {"role": "user", "content": """Your goal is to create a concise, usable, and complete software system design for reproducing the paper's method.

Based on the plan for reproducing the paper's main method, please design a concise, usable, and complete software system.
Keep the architecture simple and make effective use of open-source libraries.

-----

## Format Example
[CONTENT]
{
    "Implementation approach": "We will ...",
    "File list": ["main.py", "model.py", "trainer.py", "evaluation.py"],
    "Data structures and interfaces": "\\nclassDiagram\\n    class Main {\\n        +run_experiment()\\n    }\\n",
    "Program call flow": "\\nsequenceDiagram\\n    participant M as Main\\n",
    "Anything UNCLEAR": "Need clarification on dataset format."
}
[/CONTENT]

## Nodes
- Implementation approach: <str> Summarize the chosen solution strategy.
- File list: List[str] Only relative paths. ALWAYS include main.py or app.py.
- Data structures and interfaces: Optional[str] Use mermaid classDiagram syntax. Very detailed.
- Program call flow: Optional[str] Use sequenceDiagram syntax. Complete and detailed.
- Anything UNCLEAR: <str> Mention ambiguities.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT], nothing else.

## Action
Generate the output following the format example."""}

        task_list_msg = {"role": "user", "content": """Break down tasks according to the design, generate a task list, and analyze task dependencies.

-----

## Format Example
[CONTENT]
{
    "Required packages": ["numpy==1.21.0", "torch==1.9.0"],
    "Required Other language third-party packages": ["No third-party dependencies required"],
    "Logic Analysis": [
        ["model.py", "Defines the model architecture..."],
        ["trainer.py", "Handles training loop..."],
        ["main.py", "Entry point..."]
    ],
    "Task list": ["model.py", "trainer.py", "main.py"],
    "Full API spec": "",
    "Shared Knowledge": "Both trainer.py and evaluation.py import the Model class.",
    "Anything UNCLEAR": "Clarification needed on hardware config."
}
[/CONTENT]

## Nodes
- Required packages: Optional[List[str]] requirements.txt format.
- Required Other language third-party packages: List[str]
- Logic Analysis: List[List[str]] Each item is [filename, description].
- Task list: List[str] Filenames in dependency order. Must include all files from File list.
- Full API spec: <str> OpenAPI 3.0 spec if frontend/backend communication needed, else blank.
- Shared Knowledge: <str> Shared utilities or config vars.
- Anything UNCLEAR: <str>

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT], nothing else.

## Action
Generate the output following the format example."""}

        config_msg = {"role": "user", "content": (
            "Based on the paper and plan above, extract the training details "
            "(learning rate, batch size, epochs, etc.) and generate a config.yaml.\n"
            "DO NOT FABRICATE DETAILS — only use what the paper provides.\n\n"
            "You must write `config.yaml`.\n\n"
            "ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Output format:\n\n"
            "## Code: config.yaml\n"
            "```yaml\n"
            "## config.yaml\n"
            "training:\n"
            "  learning_rate: ...\n"
            "...\n"
            "```\n\n"
            "## Code: config.yaml"
        )}

        trajectories = list(plan_msg)
        responses = []
        stages = [
            ("Planning: Overall plan", None),
            ("Planning: Architecture design", file_list_msg),
            ("Planning: Logic design", task_list_msg),
            ("Planning: Config generation", config_msg),
        ]
        for label, extra_msg in stages:
            self._update(job_id, step=label, progress=step_counter[0] / step_counter[1])
            if extra_msg:
                trajectories.append(extra_msg)
            reply = self._call(trajectories, job_id)
            trajectories.append({"role": "assistant", "content": reply})
            responses.append(reply)
            step_counter[0] += 1

        with open(os.path.join(output_dir, "planning_trajectories.json"), "w") as f:
            json.dump(trajectories, f)

        config_yaml = self._extract_yaml(responses[3])
        config_out = os.path.join(output_dir, "planning_config.yaml")
        with open(config_out, "w") as f:
            f.write(config_yaml if config_yaml else "# config not extracted\n")

        return responses

    def analyze(
        self,
        paper_content: str,
        context_lst: list[str],
        config_yaml: str,
        task_list: list[str],
        logic_analysis_dict: dict[str, str],
        output_dir: str,
        job_id: str,
        step_counter: list,
    ) -> dict[str, str]:
        """Stage 2 — per-file logic analysis. Returns filename → analysis text."""
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert researcher, strategic analyzer and software engineer. "
                "You will receive a research paper, an overview of the plan, a design, a task, "
                "and a configuration file. "
                "Your task is to conduct a comprehensive logic analysis to accurately reproduce "
                "the experiments described in the paper. "
                "Follow the design exactly. Reference config.yaml values only."
            ),
        }

        analyses: dict[str, str] = {}
        for filename in task_list:
            if filename == "config.yaml":
                step_counter[0] += 1
                continue

            self._update(job_id, step=f"Analyzing: {filename}", progress=step_counter[0] / step_counter[1])

            desc = logic_analysis_dict.get(filename, "")
            draft = (
                f"Write the logic analysis in '{filename}', which is intended for '{desc}'."
                if desc else f"Write the logic analysis in '{filename}'."
            )

            messages = [
                system_msg,
                {"role": "user", "content": (
                    f"## Paper\n{paper_content}\n\n"
                    f"-----\n\n## Overview of the plan\n{context_lst[0]}\n\n"
                    f"-----\n\n## Design\n{context_lst[1]}\n\n"
                    f"-----\n\n## Task\n{context_lst[2]}\n\n"
                    f"-----\n\n## Configuration file\n```yaml\n{config_yaml}\n```\n\n"
                    f"-----\n\n## Instruction\n"
                    f"Conduct a Logic Analysis to assist in writing the code. "
                    f"You DON'T need to provide the actual code yet.\n\n"
                    f"{draft}\n\n"
                    f"-----\n\n## Logic Analysis: {filename}"
                )},
            ]

            reply = self._call(messages, job_id)
            analyses[filename] = reply

            save_name = filename.replace("/", "_")
            with open(os.path.join(output_dir, f"{save_name}_simple_analysis_response.json"), "w") as f:
                json.dump([{"choices": [{"message": {"content": reply}}]}], f)

            step_counter[0] += 1

        return analyses

    def generate_code(
        self,
        paper_content: str,
        context_lst: list[str],
        config_yaml: str,
        task_list: list[str],
        analyses: dict[str, str],
        output_dir: str,
        output_repo_dir: str,
        job_id: str,
        step_counter: list,
    ) -> None:
        """Stage 3 — generate code files one by one in dependency order."""
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert researcher and software engineer. "
                "You will receive a research paper, an overview of the plan, a design, a task, "
                "and a configuration file. "
                "Write elegant, modular, and maintainable code adhering to Google-style guidelines. "
                "The code must align with the paper's methodology, experimental setup, and evaluation metrics. "
                "Write code with triple quotes."
            ),
        }

        os.makedirs(output_repo_dir, exist_ok=True)
        done_files: list[str] = ["config.yaml"]
        done_code: dict[str, str] = {}

        for filename in task_list:
            if filename == "config.yaml":
                step_counter[0] += 1
                continue

            self._update(job_id, step=f"Coding: {filename}", progress=step_counter[0] / step_counter[1])

            code_context = ""
            for done in done_files:
                if done.endswith(".yaml"):
                    continue
                code_context += f"\n```python\n{done_code.get(done, '')}\n```\n\n"

            logic_analysis = analyses.get(filename, "")

            messages = [
                system_msg,
                {"role": "user", "content": (
                    f"# Context\n## Paper\n{paper_content}\n\n"
                    f"-----\n\n## Overview of the plan\n{context_lst[0]}\n\n"
                    f"-----\n\n## Design\n{context_lst[1]}\n\n"
                    f"-----\n\n## Task\n{context_lst[2]}\n\n"
                    f"-----\n\n## Configuration file\n```yaml\n{config_yaml}\n```\n\n"
                    f"-----\n\n## Code Files\n{code_context}\n"
                    f"-----\n\n# Format example\n## Code: {filename}\n```python\n## {filename}\n...\n```\n\n"
                    f"-----\n\n# Instruction\n"
                    f"Based on the paper, plan, design, task and config.yaml, write the code.\n\n"
                    f"We have {done_files}.\nNext, write only '{filename}'.\n"
                    "1. Only One file: do your best to implement THIS ONLY ONE FILE.\n"
                    "2. COMPLETE CODE: implement complete, reliable, reusable code snippets.\n"
                    "3. Set default value: ALWAYS SET A DEFAULT VALUE, USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.\n"
                    "4. Follow design: FOLLOW 'Data structures and interfaces'. DON'T CHANGE ANY DESIGN.\n"
                    "5. CHECK THAT YOU DON'T MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.\n"
                    "6. Before using an external variable/module, import it first.\n"
                    "7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.\n"
                    "8. REFER TO CONFIGURATION: use configuration from config.yaml. DO NOT FABRICATE values.\n\n"
                    f"{logic_analysis}\n\n"
                    f"## Code: {filename}"
                )},
            ]

            reply = self._call(messages, job_id)
            code = self._extract_code(reply)

            done_files.append(filename)
            done_code[filename] = code

            file_path = os.path.join(output_repo_dir, filename)
            file_dir = os.path.dirname(file_path)
            if file_dir:
                os.makedirs(file_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            step_counter[0] += 1

        config_src = os.path.join(output_dir, "planning_config.yaml")
        if os.path.exists(config_src):
            shutil.copy(config_src, os.path.join(output_repo_dir, "config.yaml"))

    def package_zip(self, output_repo_dir: str, zip_base: str) -> str:
        """Compress the repo directory into a ZIP and return the zip file path."""
        return shutil.make_archive(zip_base, "zip", output_repo_dir)

    # --- Internal thread target ---

    def _run_pipeline(
        self,
        job_id: str,
        notebook_id: str,
        paper_id: str,
        paper_title: str,
        page_count: int,
    ) -> None:
        try:
            output_dir = os.path.join(settings.PAPER2CODE_OUTPUT_DIR, paper_id)
            output_repo_dir = os.path.join(settings.PAPER2CODE_OUTPUT_DIR, f"{paper_id}_repo")
            zip_base = os.path.join(settings.PAPER2CODE_OUTPUT_DIR, f"{paper_id}_repo")

            os.makedirs(output_dir, exist_ok=True)

            self._update(job_id, step="Fetching paper content\u2026", progress=0.0)
            paper_content = self._fetch_paper_text(notebook_id, paper_id, page_count)

            # First-pass estimate: assume 5 files → total = 4 + 5 + 5 = 14
            step_counter = [0, 14]
            responses = self.plan(paper_content, output_dir, job_id, step_counter)

            context_lst = [responses[0], responses[1], responses[2]]

            config_path = os.path.join(output_dir, "planning_config.yaml")
            with open(config_path) as f:
                config_yaml = f.read()

            task_list_data = self._content_to_json(context_lst[2])
            todo_file_lst = (
                task_list_data.get("Task list")
                or task_list_data.get("task_list")
                or task_list_data.get("task list")
                or []
            )
            logic_analysis_raw = (
                task_list_data.get("Logic Analysis")
                or task_list_data.get("logic_analysis")
                or task_list_data.get("logic analysis")
                or []
            )
            logic_analysis_dict = {
                item[0]: item[1] for item in logic_analysis_raw if len(item) >= 2
            }

            n = len([f for f in todo_file_lst if f != "config.yaml"])
            step_counter[1] = 4 + n + n

            analyses = self.analyze(
                paper_content, context_lst, config_yaml,
                todo_file_lst, logic_analysis_dict,
                output_dir, job_id, step_counter,
            )

            self.generate_code(
                paper_content, context_lst, config_yaml,
                todo_file_lst, analyses,
                output_dir, output_repo_dir,
                job_id, step_counter,
            )

            self._update(job_id, step="Creating ZIP\u2026", progress=0.98)
            zip_path = self.package_zip(output_repo_dir, zip_base)

            self._update(job_id, status="done", progress=1.0, step="Done", output_path=zip_path)

        except InterruptedError:
            pass  # cancelled — status already set by cancel()
        except Exception as exc:
            self._update(job_id, status="error", error=str(exc), step=f"Error: {exc}")

    # --- Public run ---

    def run(
        self,
        notebook_id: str,
        paper_id: str,
        paper_title: str,
        page_count: int,
    ) -> str:
        """
        Launch the Paper2Code pipeline in a background thread.
        Returns the job_id immediately.
        """
        job_id = self._new_job()
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(job_id, notebook_id, paper_id, paper_title, page_count),
            daemon=True,
            name=f"code-{job_id[:8]}",
        )
        thread.start()
        return job_id
