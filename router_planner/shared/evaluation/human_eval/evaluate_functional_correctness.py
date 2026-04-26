"""
Entry point for HumanEval single-sample evaluation.

Supports both file path input (str) and direct dictionary input.
"""

import json

from .data import HUMAN_EVAL
from .evaluation import evaluate_functional_correctness_item


def entry_point_item(sample_input, problem_file=HUMAN_EVAL):
    """
    Evaluate a single HumanEval sample.

    Args:
        sample_input: Either a file path (str) to a JSON file, or a dict
                      with 'task_id' and 'completion' keys.
        problem_file: Path to the HumanEval problems JSONL file.

    Returns:
        Dict with pass@k scores (e.g., {'pass@1': 1.0}).
    """
    if isinstance(sample_input, str):
        with open(sample_input, "r") as f:
            sample = json.load(f)
    else:
        sample = sample_input

    return evaluate_functional_correctness_item(sample, problem_file)
