"""
Standalone LLM agent call functions.

Provides simple planner, executor, and summarizer functions that can be
used independently of the routing environment for testing or baseline runs.
"""

from .utils import model_prompting


def planner(query: str, llm_name: str) -> str:
    """
    Decompose a query into atomic sub-queries.

    Args:
        query: The original user query.
        llm_name: Model identifier for the LLM API.

    Returns:
        Plain text with sub-queries, one per line.
    """
    prompt = f"""You are a query decomposition assistant.

Given the user's query, decide how many sub-queries are needed (possibly just 1).
Rewrite them so that each sub-query is:
- Atomic and answerable independently.
- Non-overlapping (minimal redundancy).
- Collectively covering the user's intent.

Output rules (strict):
- Output ONLY the sub-queries, one per line.
- No numbering, bullets, quotes, or extra text.
- Use the same language as the user's query.
- If the query is already atomic, output it as a single line.
- Do not exceed 8 lines unless absolutely necessary.

User query:
{query}
"""
    return model_prompting(llm_model=llm_name, prompt=prompt)


def executor(query: str, llm_name: str, if_final: bool = False, context: str = "") -> str:
    """
    Execute a query, optionally using context for final answer generation.

    Args:
        query: The question or instruction.
        llm_name: Model identifier for the LLM API.
        if_final: If True, include context in the prompt.
        context: Additional context for final answering.

    Returns:
        Model response text.
    """
    if if_final:
        prompt = f"""You are a helpful assistant.
Given the following context and the user's query, provide a direct, complete, and accurate answer.

Context:
{context}

User query:
{query}

Answer:
"""
    else:
        prompt = query

    return model_prompting(llm_model=llm_name, prompt=prompt)


def summarizer(context: str, llm_name: str) -> str:
    """
    Summarize content into a coherent passage.

    Args:
        context: The content to summarize.
        llm_name: Model identifier for the LLM API.

    Returns:
        Summarized text.
    """
    prompt = f"""You are a professional summarizer.
Your task is to summarize the following content into a fluent, concise, and well-connected passage.
Do not copy long sentences verbatim; instead, paraphrase them naturally.
Keep the summary clear, coherent, and free of bullet points.

Content to summarize:
{context}

Summary:
"""
    return model_prompting(llm_model=llm_name, prompt=prompt)
