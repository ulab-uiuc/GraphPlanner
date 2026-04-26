"""
Task-specific prompt formatting for benchmark evaluation.

Supports: NaturalQA, TriviaQA, MMLU, GPQA, MBPP, HumanEval,
GSM8K, MATH, CommonsenseQA, OpenBookQA, ARC-Challenge, CommonGen.
"""

import ast


def format_qa_prompt(context: str, query: str) -> str:
    """Format prompt for context-based QA tasks."""
    return (
        f"Use the following context to answer the question.\n\n"
        f"## Context:\n{context}\n\n"
        f"## Question:\n{query}"
    )


def format_commongen_prompt(concepts) -> str:
    """Format prompt for the CommonGen concept-to-text task."""
    if isinstance(concepts, str):
        concepts = [c.strip() for c in concepts.split(",")]
    concepts_str = ", ".join(concepts)
    return (
        f"Generate a coherent and grammatically correct paragraph containing "
        f"the following given words (or their variations):\n"
        f"WORDS: {concepts_str}\n\n"
        f"Requirements:\n"
        f"- Use ALL the given words in the paragraph\n"
        f"- The paragraph should be coherent and grammatically correct\n"
        f"- You can use variations of the words (e.g., 'run' -> 'running')\n"
        f"- The paragraph should be at least 2-3 sentences long\n\n"
        f"Paragraph:"
    )


def format_mc_prompt(question: str, choices) -> str:
    """Format prompt for multiple-choice tasks (MMLU, GPQA, etc.)."""
    options = ["A", "B", "C", "D"]
    parsed = ast.literal_eval(choices) if isinstance(choices, str) else choices
    formatted = "".join(f"{options[i]}. {c}\n" for i, c in enumerate(parsed))
    return (
        f"Answer the following multiple-choice question by selecting the correct "
        f"option (A, B, C, or D). You Must put your final answer letter in a parenthesis.\n\n"
        f"## Question:\n{question}\n\n"
        f"## Options:\n{formatted}"
    )


def format_gsm8k_prompt(query: str) -> str:
    """Format prompt for GSM8K math word problems."""
    return f"Answer the following question.\n\nQuestion: {query}"


def format_math_prompt(query: str) -> str:
    """Format prompt for MATH competition problems."""
    return (
        f"Answer the following question. Make sure to put the answer "
        f"(and only the answer) inside \\boxed{{}}.\n\n"
        f"Question: {query}"
    )


def format_commonsense_qa_prompt(query: str, choices) -> str:
    """Format prompt for CommonsenseQA / OpenBookQA / ARC-Challenge."""
    parsed = ast.literal_eval(choices) if isinstance(choices, str) else choices
    labels = parsed["label"]
    texts = parsed["text"]
    choice_text = "".join(f"\n({l}) {t}" for l, t in zip(labels, texts))
    return (
        f"Answer the following multiple-choice question by selecting the correct "
        f"option. You Must put your final answer letter in a parenthesis.\n\n"
        f"Question: {query}\n{choice_text}"
    )


def format_mbpp_prompt(text: str, tests) -> str:
    """Format prompt for MBPP code generation tasks."""
    tests_str = "\n".join(ast.literal_eval(tests) if isinstance(tests, str) else tests)
    return (
        f"You are an expert Python programmer, and here is your task: {text} "
        f"Your code should pass these tests:\n\n{tests_str}\n"
        f" Implement the function (no irrelevant words or comments) "
        f"in this format: [BEGIN] <Your Code> [Done]\n"
    )


def format_humaneval_prompt(prompt: str) -> str:
    """Format prompt for HumanEval code completion tasks."""
    return (
        f"You are an expert Python programmer. Complete the following function:\n\n"
        f"            {prompt}\n\n"
        f"            Implement the function body only. Do not repeat the function "
        f"signature or docstring.\n"
        f"            Put the code in this format: [BEGIN] <Your Code - function body only> [Done]"
    )


def generate_task_query(task_name: str, sample_data: dict) -> str:
    """
    Generate the appropriate evaluation prompt based on the task name.

    Args:
        task_name: Benchmark task identifier.
        sample_data: Dictionary containing query, choices, and other task fields.

    Returns:
        Formatted prompt string.

    Raises:
        ValueError: If the task name is not recognized.
    """
    if task_name in ["natural_qa", "trivia_qa", "agentverse-mgsm"]:
        return sample_data["query"]

    elif task_name in ["mmlu", "agentverse-logicgrid"]:
        return format_mc_prompt(sample_data["query"], sample_data["choices"])

    elif task_name == "gpqa":
        return format_mc_prompt(sample_data["query"], sample_data["choices"])

    elif task_name == "mbpp":
        return format_mbpp_prompt(sample_data["query"], sample_data["choices"])

    elif task_name == "human_eval":
        return format_humaneval_prompt(sample_data["query"])

    elif task_name == "gsm8k":
        return format_gsm8k_prompt(sample_data["query"])

    elif task_name == "commonsense_qa":
        return format_commonsense_qa_prompt(sample_data["query"], sample_data["choices"])

    elif task_name == "math":
        return format_math_prompt(sample_data["query"])

    elif task_name in ["openbook_qa", "arc_challenge"]:
        return format_commonsense_qa_prompt(sample_data["query"], sample_data["choices"])

    elif task_name == "commongen":
        if "concepts" in sample_data and sample_data["concepts"]:
            concepts = sample_data["concepts"]
        else:
            concepts = [w.strip() for w in sample_data["query"].split(",")]
        return format_commongen_prompt(concepts)

    else:
        raise ValueError(f"Unknown task name: {task_name}")
