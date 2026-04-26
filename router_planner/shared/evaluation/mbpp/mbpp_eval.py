"""
MBPP (Mostly Basic Python Problems) evaluation.

Executes generated code against test assertions in an isolated subprocess
with timeout protection and I/O suppression for safety.
"""

import json
import contextlib
import io
import signal
import multiprocessing
from typing import Dict, Optional


class TimeoutException(Exception):
    """Raised when code execution exceeds the time limit."""
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    """Context manager that raises TimeoutException after the given seconds."""
    def handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    """Redirect stdout and stderr to suppress output during execution."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            yield


def unsafe_execute_mbpp(problem: Dict, completion: str, timeout: float, result):
    """
    Execute MBPP code with test assertions in a controlled environment.

    The completion code is executed first, then each test assertion is run.
    Results are communicated back via the shared `result` list.
    """
    test_list = problem.get("test_list", [])
    try:
        exec_globals = {}
        with swallow_io():
            with time_limit(timeout):
                # Execute the generated function definition
                exec(completion, exec_globals)
                # Run each test assertion
                for test in test_list:
                    exec(test, exec_globals)
        result.append("passed")
    except TimeoutException:
        result.append("timed out")
    except BaseException as e:
        result.append(f"failed: {e}")


def check_mbpp_correctness(
    problem: Dict,
    completion: str,
    timeout: float = 3.0,
    completion_id: Optional[int] = None,
) -> Dict:
    """
    Check if an MBPP completion passes all test assertions.

    Runs the code in an isolated subprocess with a timeout to prevent
    hanging or destructive operations.

    Args:
        problem: Problem dict with 'task_id' and 'test_list' keys.
        completion: Generated Python code string.
        timeout: Maximum execution time in seconds.
        completion_id: Optional ID for tracking results.

    Returns:
        Dict with 'task_id', 'passed' (bool), 'result' (str), 'completion_id'.
    """
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(
        target=unsafe_execute_mbpp, args=(problem, completion, timeout, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return {
        "task_id": problem["task_id"],
        "passed": result[0] == "passed",
        "result": result[0],
        "completion_id": completion_id,
    }


def evaluate_functional_correctness_item_mbpp(sample: dict, problem_file: str) -> dict:
    """
    Evaluate a single MBPP sample against the problem test cases.

    Args:
        sample: Dict with 'task_id' and 'completion' keys.
        problem_file: Path to the MBPP problems JSONL file.

    Returns:
        Dict with 'pass@1' score (1.0 if passed, 0.0 otherwise).
    """
    problems = {}
    with open(problem_file, "r") as f:
        for line in f:
            if line.strip():
                problem = json.loads(line)
                problems[problem["task_id"]] = problem

    task_id = sample["task_id"]
    completion = sample["completion"]

    if task_id not in problems:
        return {"pass@1": 0.0}

    result = check_mbpp_correctness(problems[task_id], completion)
    return {"pass@1": 1.0 if result["passed"] else 0.0}


def entry_point_item_mbpp(sample_input, problem_file: str):
    """
    Entry point for evaluating a single MBPP sample.

    Args:
        sample_input: Either a file path (str) to a JSON file, or a dict
                      with 'task_id' and 'completion' keys.
        problem_file: Path to the MBPP problems JSONL file.

    Returns:
        Dict with 'pass@1' score.
    """
    if isinstance(sample_input, str):
        with open(sample_input, "r") as f:
            sample = json.load(f)
    else:
        sample = sample_input

    return evaluate_functional_correctness_item_mbpp(sample, problem_file)
