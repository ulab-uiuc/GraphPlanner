"""
HumanEval functional correctness evaluation.

Provides pass@k estimation and single-sample evaluation functions.
"""

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import itertools

import numpy as np
import tqdm

from .data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from .execution import check_correctness


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimate pass@k for each problem.

    Uses the unbiased estimator: 1 - C(n-c, k) / C(n, k).
    """

    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluate functional correctness of generated samples from a JSONL file.

    Writes results to f"{sample_file}_results.jsonl".
    """
    problems = read_problems(problem_file)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }

    # Save results
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k


def evaluate_functional_correctness_item(
    sample,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluate a single HumanEval sample for functional correctness.

    Args:
        sample: Dict with 'task_id' and 'completion' keys.
        problem_file: Path to HumanEval problems file.

    Returns:
        Dict with pass@k scores (typically only pass@1 for single samples).
    """
    timeout = 3.0
    k = [1, 10, 100]

    problems = read_problems(problem_file)

    completion_id = Counter()
    results = defaultdict(list)

    task_id = sample["task_id"]
    completion = sample["completion"]

    # Execute and check correctness
    result = check_correctness(
        problem=problems[task_id], completion=completion, timeout=timeout
    )
    results[result["task_id"]].append((completion_id[task_id], result))
    completion_id[task_id] += 1

    # Calculate pass@k
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in [1, 10, 100]
        if (total >= k).all()
    }

    return pass_at_k
