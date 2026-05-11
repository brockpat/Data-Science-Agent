# -*- coding: utf-8 -*-
"""
Entry point for the Data Science Agent benchmark.
"""

from pathlib import Path
import sys

# Make the project root importable, so `src.*` imports work
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from langchain_openai import ChatOpenAI

from src.config import (
    DATA_PATH,
    MARKDOWN_REPORT_PATH,
    API_KEY,
    OPENAI_MODEL,
    MAX_CODE_ATTEMPTS,
    MAX_OUTPUT_CHARS,
)

from src.data_loader import load_dataframe
from src.code_executor import make_python_analysis_tool
from src.agent import AgentState, LLMAgent
from src.prompts import SYSTEM_INSTRUCTION
from src.benchmark import run_questions
from src.reporting import write_markdown


def build_agent() -> LLMAgent:
    # Load the dataset from the configured path
    df = load_dataframe(DATA_PATH)

    # Initialize the custom tool that executes generated Python code safely
    run_python_analysis_code = make_python_analysis_tool(
        df=df,
        max_output_chars=MAX_OUTPUT_CHARS,
    )
    
    # Configure the base LLM with specified model and API key
    model = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=API_KEY,
    )
    
    # Store the dataframe in the agent's state
    state = AgentState(dataset=df)

    # Construct the LLM agent tying together state, model, system prompts, and tools
    llm = LLMAgent(
        state=state,
        model=model,
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[run_python_analysis_code],
        max_code_attempts=MAX_CODE_ATTEMPTS,
    )

    return llm


def main() -> None:
    llm = build_agent()
    
    # Execute the predefined benchmark questions using the agent
    results = run_questions(
        llm_agent=llm,
        sleep_seconds=2.0,
    )
    
    # Export the benchmark results to a formatted Markdown file
    write_markdown(
        benchmark_results=results,
        output_path=MARKDOWN_REPORT_PATH,
    )
    

    print(f"\nMarkdown benchmark report written to: {MARKDOWN_REPORT_PATH}")


if __name__ == "__main__":
    main()