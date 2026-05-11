# -*- coding: utf-8 -*-
"""
Markdown report generation.
"""

from pathlib import Path
from datetime import datetime
import json

import pandas as pd


def format_code_output(output) -> str:
    """
    Nicely format the code output for Markdown.
    Handles dicts, JSON strings, and plain strings.
    """

    if isinstance(output, dict):
        return json.dumps(output, indent=2, default=str)

    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            return json.dumps(parsed, indent=2, default=str)
        except json.JSONDecodeError:
            return output

    return str(output)


def write_markdown(
    benchmark_results: pd.DataFrame,
    output_path: str | Path,
    title: str = "Agentic Data Science Agent Benchmark Results",
) -> None:
    output_path = Path(output_path)

    lines = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated on: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append(
        "This benchmark evaluates an agentic data science workflow where the agent "
        "writes Python code, executes it against a pandas dataframe, retries failed "
        "code when necessary, and returns a natural-language answer together with "
        "the executed code and code output."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")

    summary_cols = [
        col for col in [
            "question_id",
            "question",
            "num_code_attempts",
            "tool_failed",
            "hit_retry_limit",
        ]
        if col in benchmark_results.columns
    ]

    if summary_cols:
        summary_df = benchmark_results[summary_cols].copy()
        lines.append(summary_df.to_markdown(index=False))
        lines.append("")

    lines.append("## Detailed Results")
    lines.append("")

    for _, row in benchmark_results.iterrows():
        question_id = row.get("question_id", "")
        question = row.get("question", "")
        answer = row.get("answer", "")

        code = row.get("code", "")
        code_output = row.get("code_output", None)

        num_attempts = row.get("num_code_attempts", "")
        tool_failed = row.get("tool_failed", "")
        hit_retry_limit = row.get("hit_retry_limit", "")

        lines.append(f"### {question_id}: {question}")
        lines.append("")

        lines.append("**Final answer**")
        lines.append("")
        lines.append(str(answer).strip())
        lines.append("")

        lines.append("**Run metadata**")
        lines.append("")
        lines.append(f"- Code attempts: `{num_attempts}`")
        lines.append(f"- Tool failed: `{tool_failed}`")
        lines.append(f"- Hit retry limit: `{hit_retry_limit}`")
        lines.append("")

        lines.append("**Executed code**")
        lines.append("")
        lines.append("```python")
        lines.append(str(code).strip())
        lines.append("```")
        lines.append("")

        lines.append("**Code output**")
        lines.append("")
        lines.append("```json")
        lines.append(format_code_output(code_output).strip())
        lines.append("```")
        lines.append("")

        all_attempts = row.get("all_code_attempts", [])

        if isinstance(all_attempts, list) and len(all_attempts) > 1:
            lines.append("<details>")
            lines.append("<summary>All code attempts</summary>")
            lines.append("")

            for attempt_idx, attempt_code in enumerate(all_attempts, start=1):
                lines.append(f"#### Attempt {attempt_idx}")
                lines.append("")
                lines.append("```python")
                lines.append(str(attempt_code).strip())
                lines.append("```")
                lines.append("")

            lines.append("</details>")
            lines.append("")

        all_tool_outputs = row.get("all_tool_outputs", [])

        if isinstance(all_tool_outputs, list) and len(all_tool_outputs) > 1:
            lines.append("<details>")
            lines.append("<summary>All tool outputs</summary>")
            lines.append("")

            for attempt_idx, tool_output in enumerate(all_tool_outputs, start=1):
                lines.append(f"#### Tool output {attempt_idx}")
                lines.append("")
                lines.append("```json")
                lines.append(format_code_output(tool_output).strip())
                lines.append("```")
                lines.append("")

            lines.append("</details>")
            lines.append("")

        lines.append("---")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")