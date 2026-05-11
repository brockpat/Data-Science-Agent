# -*- coding: utf-8 -*-
"""
LLM data science agent.
"""

from dataclasses import dataclass
import json

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage


@dataclass
class AgentState:
    dataset: pd.DataFrame


class LLMAgent:
    def __init__(
        self,
        state: AgentState,
        model: ChatOpenAI,
        system_instruction: str,
        tools: list,
        max_code_attempts: int,
    ):
        self.state = state
        self.system_instruction = system_instruction
        self.max_code_attempts = max_code_attempts

        self.tools = tools
        self.base_model = model
        self.model_with_tools = model.bind_tools(self.tools)

        self.tool_map = {
            tool.name: tool
            for tool in self.tools
        }

        self.generated_code_log = []
        self.tool_result_log = []

    def _tool_output_failed(self, output: str) -> bool:
        try:
            parsed = json.loads(output)
            return parsed.get("status") != "success"
        except json.JSONDecodeError:
            return True

    def _dataset_context(self) -> str:
        dataset_preview = self.state.dataset.head().to_string()
        dataset_columns = list(self.state.dataset.columns)
        dataset_shape = self.state.dataset.shape
        dataset_dtypes = self.state.dataset.dtypes.astype(str).to_string()

        return f"""
The dataframe `df` is already loaded in memory.

Dataframe shape:
{dataset_shape}

Dataframe head:
{dataset_preview}

Dataframe columns:
{dataset_columns}

Dataframe dtypes:
{dataset_dtypes}
"""

    def answer(self, question: str, question_id: str | None = None) -> dict:
        used_code = []
        tool_outputs = []

        messages = [
            SystemMessage(content=self.system_instruction),
            HumanMessage(
                content=f"""
{self._dataset_context()}

Question:
{question}
"""
            ),
        ]

        response = self.model_with_tools.invoke(messages)
        messages.append(response)

        code_attempts = 0
        final_code = ""
        final_tool_output = ""
        parsed_tool_result = {}
        last_tool_failed = False

        while response.tool_calls and code_attempts < self.max_code_attempts:
            if len(response.tool_calls) > 1:
                code_attempts += 1

                messages.append(
                    HumanMessage(
                        content=(
                            "You called more than one tool. "
                            "Call exactly one tool with one complete Python program. "
                            "Do not explain; only call the tool once."
                        )
                    )
                )

                response = self.model_with_tools.invoke(messages)
                messages.append(response)
                continue

            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name not in self.tool_map:
                tool_result = json.dumps(
                    {
                        "status": "error",
                        "error_type": "UnknownTool",
                        "error_message": f"Unknown tool requested: {tool_name}",
                    }
                )
            else:
                if tool_name == "run_python_analysis_code":
                    code_attempts += 1

                    code = tool_args.get("code", "")
                    used_code.append(code)
                    final_code = code

                    self.generated_code_log.append(
                        {
                            "question_id": question_id,
                            "question": question,
                            "attempt": code_attempts,
                            "code": code,
                        }
                    )

                    print("\n" + "=" * 100)
                    print(f"AGENT-GENERATED PYTHON CODE | {question_id} | ATTEMPT {code_attempts}")
                    print("=" * 100)
                    print(code)
                    print("=" * 100 + "\n")

                selected_tool = self.tool_map[tool_name]
                tool_result = selected_tool.invoke(tool_args)

            tool_result_str = str(tool_result)

            try:
                parsed_tool_result = json.loads(tool_result_str)
            except json.JSONDecodeError:
                parsed_tool_result = {
                    "status": "unknown",
                    "raw_output": tool_result_str,
                }

            tool_outputs.append(tool_result_str)
            final_tool_output = tool_result_str
            last_tool_failed = self._tool_output_failed(tool_result_str)

            self.tool_result_log.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "tool_name": tool_name,
                    "attempt": code_attempts,
                    "tool_result": tool_result_str,
                }
            )

            messages.append(
                ToolMessage(
                    content=tool_result_str,
                    tool_call_id=tool_call_id,
                )
            )

            if last_tool_failed and code_attempts < self.max_code_attempts:
                messages.append(
                    HumanMessage(
                        content=(
                            "The code failed. Debug the Python program. "
                            "Do not import libraries; df, pd, and np are already available. "
                            "Use the same instructions: assign the final answer to result. "
                            "Do not explain; only call the tool with corrected code."
                        )
                    )
                )

                response = self.model_with_tools.invoke(messages)
                messages.append(response)

            else:
                response = self.base_model.invoke(messages)
                messages.append(response)
                break

        return {
            "question_id": question_id,
            "question": question,
            "answer": response.content,
            "code": final_code,
            "code_output": parsed_tool_result,
            "raw_code_output": final_tool_output,
            "all_code_attempts": used_code,
            "all_tool_outputs": tool_outputs,
            "num_code_attempts": code_attempts,
            "tool_failed": last_tool_failed,
            "hit_retry_limit": last_tool_failed and code_attempts >= self.max_code_attempts,
        }