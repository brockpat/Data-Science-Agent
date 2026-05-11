# Data Science Agent

An agentic data science workflow that answers natural-language questions about a pandas dataframe by writing, executing, debugging, and summarizing Python analysis code.

This project demonstrates a lightweight but intentional pattern for building a data analysis agent: the LLM is allowed to generate Python, but only inside a constrained execution environment with explicit guardrails, retry logic, structured logging, and benchmark reporting.

## Overview

The agent is designed to answer questions over a pre-loaded dataframe named `df`.

Instead of guessing from a small preview of the data, the agent can call a Python execution tool that runs analysis code against the full dataframe. The result is then returned as a natural-language answer, while the generated code, tool output, retry count, and failure status are stored separately for inspection.

The workflow is especially useful for benchmark-style data science questions such as:

- Which features have the strongest correlations?
- Which stock has the highest feature volatility?
- Which dates look unusual based on z-scores?
- Which feature best separates `stock_id` groups?
- Which feature changed most over time for each stock?

## Key Features

### Agent-written Python analysis

The agent writes Python code on demand to answer questions about the dataframe.

The execution environment provides only:

- `df`: a deep copy of the dataframe
- `pd`: pandas
- `np`: NumPy

The generated code must assign its final answer to a variable named `result`.

This makes tool outputs predictable, easy to parse, and simple to benchmark.

### Controlled execution guardrails

The project intentionally restricts what the model-generated code is allowed to do.

Before execution, the tool checks for forbidden patterns including:

```python
import 
from 
__import__
open(
exec(
eval(
subprocess
os.
sys.
```

This prevents the agent from importing new libraries, reading or writing files, executing nested code, or accessing system-level functionality.

The execution tool also runs with a restricted `__builtins__` dictionary. Only a small set of safe Python functions are exposed, such as:

```python
len, range, min, max, sum, abs, round, sorted,
list, dict, set, tuple, str, int, float, bool,
print, enumerate, zip, any, all, isinstance
```

This keeps the agent focused on dataframe analysis rather than arbitrary code execution.

### Dataset protection

The dataframe is passed into the local execution environment as:

```python
df.copy(deep=True)
```

This prevents model-generated code from mutating the original dataframe held by the agent state.

### One code program per attempt

The system prompt and agent loop both enforce a strict discipline:

- The model should call exactly one tool at a time.
- Each tool call should contain one complete Python program.
- The model should not call multiple tools in the same assistant response.
- The model should not explain while calling the tool.
- After a successful tool result, the model must stop using tools and produce a final answer.

If the model attempts more than one tool call in a single response, the agent rejects that behavior and asks it to retry with exactly one complete Python program.

### Automatic debugging with maximum retries

The agent supports controlled retries when generated code fails.

By default:

```python
max_code_attempts = 3
```

If the tool output indicates failure, the agent gives the model the error message and asks it to debug the code. The model can then submit a corrected complete Python program.

The retry loop stops when either:

- the code succeeds, or
- the maximum number of code attempts is reached.

The returned answer record includes:

```python
"num_code_attempts"
"tool_failed"
"hit_retry_limit"
"all_code_attempts"
"all_tool_outputs"
```

This makes failures visible rather than hidden.

### Structured tool outputs

The Python execution tool always returns JSON.

Successful outputs look like:

```json
{
  "status": "success",
  "result": "...",
  "stdout": ""
}
```

Failure outputs include the error type, message, and a short traceback:

```json
{
  "status": "error",
  "error_type": "NameError",
  "error_message": "...",
  "traceback": "..."
}
```

This structure makes it easy for the agent to detect failed code and retry intelligently.

### Output truncation

Large outputs are automatically truncated to avoid overwhelming the model context or the terminal.

The maximum formatted result size is:

```python
MAX_OUTPUT_CHARS = 12_000
```

If an output exceeds that limit, the tool returns the beginning of the result followed by a truncation notice.

### Full audit trail

For each question, the agent records:

- the question ID
- the user question
- every generated code attempt
- every tool output
- the final code
- the final natural-language answer
- whether the tool failed
- whether the retry limit was reached

This makes the agent easier to debug, evaluate, and compare across benchmark runs.

### Benchmark report generation

The project includes a benchmark runner with ten data science questions. Results are collected into a dataframe and exported to a Markdown report.

The generated report includes:

- benchmark overview
- summary table
- final answer for each question
- run metadata
- executed code
- structured code output
- expandable details for multiple code attempts

Example output file:

```text
outputs/results.md
```

## Project Structure

```text
Data-Science-Agent/
├── data/
│   └── dataframe.csv
├── outputs/
│   └── results.md
├── src/
│   ├── agent.py
│   ├── benchmark.py
│   ├── code_executor.py
│   ├── config.py
│   ├── data_loader.py
│   ├── prompts.py
│   └── reporting.py
├── main.py
└── README.md
```

## How It Works

### 1. Load the dataframe

The project expects a CSV file at:

```python
data/dataframe.csv
```

The dataframe is loaded with:

```python
df = pd.read_csv(data_path, parse_dates=["date"])
```

### 2. Create the analysis tool

The `run_python_analysis_code` tool executes model-generated Python against the dataframe.

The generated code can use `df`, `pd`, and `np`, but it cannot import libraries, access the filesystem, or run system commands.

The code must assign the final answer to:

```python
result
```

### 3. Initialize the agent

The `DataScienceAgent` class wraps:

- the dataframe state
- the language model
- the system instruction
- the available tools
- retry configuration
- generated code logs
- tool result logs

### 4. Ask a question

Each call to:

```python
agent.answer(question, question_id)
```

returns a structured dictionary containing the final answer, generated code, output, retry metadata, and logs.

### 5. Run the benchmark

The benchmark loop runs a predefined set of analytical questions and stores each answer record.

The results are converted into a dataframe and written to a Markdown benchmark report.

## Example

Question:

```text
For each stock_id, which feature changed the most in absolute value from the first available date to the last available date?
```

Generated code:

```python
feature_cols = [c for c in df.columns if c.startswith('feature_')]
first_rows = df.sort_values(['stock_id','date']).groupby('stock_id', as_index=True).first()
last_rows = df.sort_values(['stock_id','date']).groupby('stock_id', as_index=True).last()
abs_change = (last_rows[feature_cols] - first_rows[feature_cols]).abs()
max_feat = abs_change.idxmax(axis=1)
max_val = abs_change.max(axis=1)
result = pd.DataFrame({
    'stock_id': abs_change.index,
    'feature_most_changed': max_feat.values,
    'abs_change': max_val.values
}).reset_index(drop=True)
result
```

Example final answer:

```text
For each stock_id, the feature with the largest absolute change from the first available date to the last available date is:

- STK001 → feature_043 with absolute change 9.358163
- STK002 → feature_043 with absolute change 8.182729
- STK003 → feature_050 with absolute change 8.276696
- STK004 → feature_043 with absolute change 7.020201
- STK005 → feature_047 with absolute change 7.757437
```

## Benchmark Questions

The included benchmark evaluates the agent on the following questions:

1. For each `stock_id`, which two feature pairs have the highest absolute correlation?
2. Across the whole dataset, which feature has the highest average absolute correlation with all other features?
3. Which `stock_id` has the highest overall feature volatility, measured as the average standard deviation across all feature columns?
4. For each date, which `stock_id` has the highest average value across all feature columns?
5. Which features show the largest cross-stock differences in average value?
6. Which `stock_id` has the most missing values across all feature columns?
7. Which feature has the strongest upward time trend over the full sample?
8. For each `stock_id`, which feature changed the most in absolute value from the first available date to the last available date?
9. Which dates look most unusual based on the average absolute z-score across all feature values?
10. Which feature best separates the different `stock_id` groups, measured by the ratio of between-stock variance to within-stock variance?

## Installation

Install the core dependencies:

```bash
pip install pandas numpy langchain langchain-openai
```

You also need an OpenAI API key available in your environment or passed into the `ChatOpenAI` client.

Example:

```python
model = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)
```

## Usage

Update the project paths:

```python
from pathlib import Path

project_root = Path(__file__).resolve().parent
data_path = project_root / "data" / "dataframe.csv"
```

Run the script.

The agent will:

1. Load the dataframe.
2. Initialize the LLM with tool access.
3. Run the benchmark questions.
4. Print generated code and answers to the console.
5. Save a Markdown benchmark report.

## Design Philosophy

This project is intentionally not a fully unrestricted code agent.

Instead, it explores a safer and more inspectable pattern for data science automation:

- Give the model enough power to compute real answers.
- Keep the data loaded in memory instead of exposing the filesystem.
- Prevent imports and system operations.
- Require a single explicit result variable.
- Retry failed code, but only up to a fixed limit.
- Log every generated code attempt.
- Separate natural-language answers from executable code.
- Produce benchmark artifacts that can be reviewed after the run.

The result is an agent that behaves more like a careful junior analyst: it can write code, test it, correct mistakes, and explain the answer, while still operating inside a constrained and auditable environment.

## Current Limitations

This project is a local prototype and is not intended to be a hardened production sandbox.

Important limitations:

- The forbidden-pattern check is a simple string-based pre-check, not a full security sandbox.
- The tool uses Python `exec`, so it should only be run in a trusted local environment.
- The dataframe path is currently hardcoded.
- The benchmark assumes a dataset with `date`, `stock_id`, and `feature_*` columns.
- Generated code is constrained to pandas and NumPy.
- Very large result outputs are truncated.

## Future Improvements

Potential next steps:

- Replace string-based code checks with AST-based validation.
- Move paths and model configuration into environment variables or a config file.
- Add unit tests for the code execution guardrails.
- Add timeout handling for long-running generated code.
- Export benchmark results to JSON or CSV in addition to Markdown.
- Add support for multiple datasets.
- Add richer evaluation metrics for benchmark correctness.
- Run the executor in an isolated container or subprocess for stronger sandboxing.

## Example Output

```json
{
  "status": "success",
  "result": "stock_id feature_most_changed  abs_change\n  STK001          feature_043    9.358163\n  STK002          feature_043    8.182729\n  STK003          feature_050    8.276696\n  STK004          feature_043    7.020201\n  STK005          feature_047    7.757437",
  "stdout": ""
}
```

## Why This Project Matters

Many data agents either avoid executing code entirely or execute code with too much freedom.

This project sits in the middle: the agent can perform real dataframe analysis, but every code attempt is constrained, logged, retried intentionally, and converted into a clean final answer.

That makes the workflow practical for experimentation, transparent for debugging, and useful for evaluating how well an LLM can behave as a data science assistant.
