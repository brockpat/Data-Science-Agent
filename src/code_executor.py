# -*- coding: utf-8 -*-
"""
Constrained Python execution tool for dataframe analysis.
"""

from io import StringIO
import contextlib
import json
import traceback

import pandas as pd
import numpy as np
from langchain.tools import tool


FORBIDDEN_PATTERNS = [
    "import ",
    "from ",
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "subprocess",
    "os.",
    "sys.",
]


SAFE_BUILTINS = {
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "print": print,
    "enumerate": enumerate,
    "zip": zip,
    "any": any,
    "all": all,
    "isinstance": isinstance,
}


def make_python_analysis_tool(
    df: pd.DataFrame,
    max_output_chars: int,
):
    """
    Build a LangChain tool that executes model-written Python
    against a protected deep copy of the dataframe.
    """

    @tool
    def run_python_analysis_code(code: str) -> str:
        """
        Run Python code to analyze the dataframe.

        Available objects:
        - df: pandas dataframe
        - pd: pandas
        - np: numpy

        The code should assign the final answer to a variable called result.
        """

        local_env = {
            "df": df.copy(deep=True),
            "pd": pd,
            "np": np,
        }

        global_env = {
            "__builtins__": SAFE_BUILTINS
        }

        if any(pattern in code for pattern in FORBIDDEN_PATTERNS):
            return json.dumps(
                {
                    "status": "error",
                    "error_type": "RejectedCode",
                    "error_message": (
                        "The code used a forbidden operation. "
                        "Do not import libraries or use file/system operations. "
                        "Use only the provided df, pd, and np objects."
                    ),
                },
                default=str,
            )

        try:
            stdout_buffer = StringIO()

            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, global_env, local_env)

            stdout = stdout_buffer.getvalue()

            if "result" not in local_env:
                return json.dumps(
                    {
                        "status": "error",
                        "error_type": "MissingResult",
                        "error_message": "Code ran, but no variable named result was assigned.",
                        "stdout": stdout,
                    },
                    default=str,
                )

            result = local_env["result"]

            if isinstance(result, pd.DataFrame):
                formatted_result = result.to_string(index=False)
            elif isinstance(result, pd.Series):
                formatted_result = result.to_string()
            else:
                formatted_result = str(result)

            if len(formatted_result) > max_output_chars:
                formatted_result = (
                    formatted_result[:max_output_chars]
                    + "\n\n[Output truncated because it exceeded the maximum display length.]"
                )

            return json.dumps(
                {
                    "status": "success",
                    "result": formatted_result,
                    "stdout": stdout,
                },
                default=str,
            )

        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                },
                default=str,
            )

    return run_python_analysis_code