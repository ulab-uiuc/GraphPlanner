"""
Unified response evaluation dispatcher.

Routes evaluation to the appropriate metric function based on
the task name and metric type. Supports:
- Exact Match (EM) and Contained Exact Match (CEM)
- F1 score for partial token matching
- BERT score for semantic similarity
- GSM8K and MATH answer extraction and comparison
- Code evaluation via HumanEval and MBPP execution
- CommonGen concept coverage
"""

import re
from .utils import f1_score, exact_match_score, cem_score, evaluate_commongen_coverage
from .evaluation.human_eval.evaluate_functional_correctness import entry_point_item
from .evaluation.mbpp.mbpp_eval import entry_point_item_mbpp
from .math_eval import last_boxed_only_string, remove_boxed, is_equiv


def eval_perf(
    metric: str,
    prediction: str,
    ground_truth: str,
    task_name: str,
    task_id=None,
    humaneval_path: str = "data/HumanEval.jsonl",
    mbpp_path: str = "data/mbpp.jsonl",
) -> float:
    """
    Evaluate model prediction against ground truth using the specified metric.

    Args:
        metric: Metric name (e.g., "em", "cem", "f1_score", "GSM8K", "MATH", "code_eval").
        prediction: Model's predicted output.
        ground_truth: Expected correct answer.
        task_name: Name of the benchmark task.
        task_id: Task identifier (required for code evaluation).
        humaneval_path: Path to HumanEval problems file.
        mbpp_path: Path to MBPP problems file.

    Returns:
        Score as a float (0.0 to 1.0).
    """
    # Override metric for specific QA/MC tasks to use CEM
    if task_name in [
        "natural_qa", "trivia_qa", "squad", "boolq",
        "mmlu", "gpqa", "agentverse-mgsm", "agentverse-logicgrid",
    ]:
        metric = "cem"

    # --- Exact Match ---
    if metric == "em":
        return float(exact_match_score(prediction, ground_truth))

    # --- Contained Exact Match ---
    elif metric == "cem":
        return float(cem_score(prediction, ground_truth))

    # --- Multiple-Choice Exact Match ---
    elif metric == "em_mc":
        return float(exact_match_score(prediction, ground_truth, normal_method="mc"))

    # --- BERT Score ---
    elif metric == "bert_score":
        from .utils import get_bert_score
        return get_bert_score([prediction], [ground_truth])

    # --- GSM8K: extract last number and compare ---
    elif metric == "GSM8K":
        gt = ground_truth.split("####")[-1].replace(",", "").replace("$", "").replace(".", "").strip()
        numbers = re.findall(r"(\-?[0-9\.\,]+)", prediction)
        if not numbers:
            return 0.0
        # Find the last valid number
        final = None
        for candidate in reversed(numbers):
            if candidate not in ["", "."]:
                final = candidate
                break
        if final is None:
            return 0.0
        final = final.replace(",", "").replace("$", "").replace(".", "").strip()
        return 1.0 if final == gt else 0.0

    # --- MATH: extract \\boxed{} and check equivalence ---
    elif metric == "MATH":
        gt_boxed = remove_boxed(last_boxed_only_string(ground_truth))
        try:
            pred_boxed_str = last_boxed_only_string(prediction)
            if pred_boxed_str is not None:
                answer = remove_boxed(pred_boxed_str)
                if is_equiv(answer, gt_boxed):
                    return 1.0
        except Exception:
            return 0.0
        return 0.0

    # --- F1 Score ---
    elif metric == "f1_score" or task_name in ["quac"]:
        return f1_score(prediction, ground_truth)[0]

    # --- CommonGen Coverage ---
    elif metric == "commongen_coverage" or task_name == "commongen":
        return evaluate_commongen_coverage(prediction, ground_truth)

    # --- Code Evaluation (HumanEval / MBPP) ---
    elif metric == "code_eval":
        if task_id is None:
            raise ValueError("task_id is required for code_eval metric")

        is_mbpp = not str(task_id).startswith("HumanEval")

        # Extract code between [BEGIN] and [Done]/[DONE] tags
        code_match = re.search(
            r"\[BEGIN\](.*?)(?:\[DONE\]|\[Done\]|$)", prediction, re.DOTALL | re.IGNORECASE
        )

        if is_mbpp:
            code = code_match.group(1).strip() if code_match else prediction.strip()
            sample = {"task_id": int(task_id), "completion": code}
            result = entry_point_item_mbpp(sample, mbpp_path)
            return result["pass@1"]
        else:
            if code_match:
                raw_code = code_match.group(1).strip()
                # If the code doesn't start with 'def ', indent it as a function body
                code = raw_code if raw_code.lstrip().startswith("def ") else \
                    "    " + raw_code.replace("\n", "\n    ")
            else:
                code = prediction.strip()
            sample = {"task_id": task_id, "completion": code}
            result = entry_point_item(sample, humaneval_path)
            return result["pass@1"]

    # --- Default: unknown metric ---
    else:
        return 0.0
