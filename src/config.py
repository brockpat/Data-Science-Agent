# -*- coding: utf-8 -*-

from pathlib import Path
import os
from dotenv import load_dotenv

# Project root:
# Data_Science_Agent/
# ├── .env
# ├── main.py
# └── src/
#     └── config.py

# Calculates base directory structurally relative to this config file
BASE_PATH = Path(__file__).resolve().parents[1]

# Explicitly load .env from the project root
ENV_PATH = BASE_PATH / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DATA_PATH = BASE_PATH / "data" / "dataframe.csv"

OUTPUT_DIR = BASE_PATH / "outputs"
# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MARKDOWN_REPORT_PATH = OUTPUT_DIR / "results.md"

# Fetch API key for LLM
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError(
        f"API_KEY was not found. Expected a .env file at: {ENV_PATH}"
    )

OPENAI_MODEL = "gpt-5.4-mini"

# Number of times the agent can retry code execution on failure
MAX_CODE_ATTEMPTS = 3

# Max characters returned from the code executor
MAX_OUTPUT_CHARS = 12_000