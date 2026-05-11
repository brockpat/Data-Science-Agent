# -*- coding: utf-8 -*-
"""
Benchmark questions and runner.
"""

import time
import pandas as pd


BENCHMARK_QUESTIONS = [
    "For each stock_id, which two feature pairs have the highest absolute correlation?",

    "Across the whole dataset, which feature has the highest average absolute correlation with all other features?",

    "Which stock_id has the highest overall feature volatility, measured as the average standard deviation across all feature columns?",

    "For each date, which stock_id has the highest average value across all feature columns?",

    "Which features show the largest cross-stock differences in average value?",

    "Which stock_id has the most missing values across all feature columns?",

    "Which feature has the strongest upward time trend over the full sample?",

    "For each stock_id, which feature changed the most in absolute value from the first available date to the last available date?",

    "Which dates look most unusual based on the average absolute z-score across all feature values?",

    "Which feature best separates the different stock_id groups, measured by the ratio of between-stock variance to within-stock variance?",
]


def run_questions(
    llm_agent,
    questions: list[str] | None = BENCHMARK_QUESTIONS,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    answers = []

    for i, question in enumerate(questions, start=1):
        question_id = f"Q{i:02d}"

        print("\n" + "#" * 60)
        print(f"RUNNING {question_id}: {question}")
        print("#" * 60)

        answer_record = llm_agent.answer(
            question=question,
            question_id=question_id,
        )

        print("\nFINAL ANSWER")
        print(answer_record["answer"])

        print("\nALL CODE ATTEMPTS")
        for attempt_num, code in enumerate(answer_record["all_code_attempts"], start=1):
            print(f"\n--- Attempt {attempt_num} ---")
            print(code)

        print("\nALL TOOL OUTPUTS")
        for attempt_num, output in enumerate(answer_record["all_tool_outputs"], start=1):
            print(f"\n--- Attempt {attempt_num} ---")
            print(output)

        answers.append(answer_record)
        time.sleep(sleep_seconds)

    return pd.DataFrame(answers)