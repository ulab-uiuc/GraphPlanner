"""
Shared utility functions for the RouterPlanner framework.

Provides:
- EmbeddingModel: Longformer-based text embedding with mean pooling
- File I/O helpers for JSON and pickle formats
- Text evaluation metrics (F1, EM, CEM, BERT score, CommonGen coverage)
- LLM API client with thread-safe round-robin key rotation
"""

import os
import json
import re
import string
import pickle
from collections import Counter
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import LongformerModel, LongformerTokenizer
from sklearn.decomposition import PCA
from bert_score import score as bert_score_fn
from openai import OpenAI
import itertools
import threading


# ---------------------------------------------------------------------------
# LLM API Client Setup
# ---------------------------------------------------------------------------
# API keys are loaded from the NVIDIA_API_KEYS environment variable.
# Set it as a comma-separated string, e.g.:
#   export NVIDIA_API_KEYS="[YOUR_API_KEY_1],[YOUR_API_KEY_2],[YOUR_API_KEY_3]"

_api_keys_str = os.environ.get("NVIDIA_API_KEYS", "")
API_KEYS = [k.strip() for k in _api_keys_str.split(",") if k.strip()]

if not API_KEYS:
    import warnings
    warnings.warn(
        "NVIDIA_API_KEYS environment variable is not set. "
        "LLM API calls will fail. Set it as a comma-separated list of API keys."
    )

# Create one OpenAI-compatible client per key
clients = [
    OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=key,
        timeout=120,
        max_retries=5,
    )
    for key in API_KEYS
]

# Thread-safe round-robin iterator for load balancing across API keys
_client_cycle = itertools.cycle(clients) if clients else itertools.cycle([None])
_client_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
class EmbeddingModel:
    """
    Text embedding model based on Longformer with mean pooling.

    Uses the AllenAI Longformer (supports up to 4096 tokens) to produce
    fixed-size embeddings by averaging the last hidden state over
    non-padding positions.
    """

    def __init__(self, model_name: str = "allenai/longformer-base-4096", device: str = "cpu"):
        self.device = device
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Compute a single embedding vector for the given text.

        Args:
            text: Input text string.

        Returns:
            1-D tensor of shape [hidden_dim] on CPU.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=4096,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            # Mean pooling over non-padding tokens
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts

        return mean_pooled.squeeze(0).cpu()


# ---------------------------------------------------------------------------
# File I/O Helpers
# ---------------------------------------------------------------------------
def loadjson(filename: str) -> dict:
    """Load and return data from a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def savejson(data: dict, filename: str) -> None:
    """Save a dictionary to a JSON file with indentation."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def loadpkl(filename: str):
    """Load and return an object from a pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def savepkl(data, filename: str) -> None:
    """Save an object to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Text Normalization & Evaluation Metrics
# ---------------------------------------------------------------------------
def normalize_answer(s: str, normal_method: str = "") -> str:
    """
    Normalize text for evaluation by lowercasing, removing articles,
    punctuation, and extra whitespace.

    Args:
        s: Input string.
        normal_method: "mc" for multiple-choice extraction, "" for standard.

    Returns:
        Normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    def mc_remove(text):
        matches = re.findall(r"\(\s*[a-zA-Z]\s*\)", text)
        return matches[-1] if matches else ""

    if normal_method == "mc":
        return mc_remove(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Compute token-level F1 score between prediction and ground truth.

    Returns:
        Tuple of (f1, precision, recall).
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    ZERO = (0.0, 0.0, 0.0)

    if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
        return ZERO
    if norm_gt in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
        return ZERO

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str, normal_method: str = "") -> bool:
    """Check if prediction exactly matches ground truth after normalization."""
    if normal_method == "mc":
        return ground_truth.strip().lower() in normalize_answer(prediction, "mc").strip().lower()
    return normalize_answer(prediction, normal_method) == normalize_answer(ground_truth, normal_method)


def cem_score(prediction: str, ground_truth: str) -> float:
    """
    Contained Exact Match score.
    Returns 1.0 if the normalized ground truth is contained in the prediction.
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    return 1.0 if (norm_pred == norm_gt or norm_gt in norm_pred) else 0.0


def cemf1_score(prediction: str, ground_truth: str) -> float:
    """
    CEM-F1: returns 1.0 on exact/contained match, otherwise token-level F1.
    """
    if cem_score(prediction, ground_truth) == 1.0:
        return 1.0
    return f1_score(prediction, ground_truth)[0]


def get_bert_score(generated: List[str], references: List[str]) -> float:
    """
    Compute average BERT-score F1 between generated texts and references.
    """
    scores = []
    for gen, ref in zip(generated, references):
        _, _, F = bert_score_fn([gen], [ref], lang="en", verbose=False)
        scores.append(F.mean().item())
    return np.mean(scores)


# ---------------------------------------------------------------------------
# CommonGen Evaluation
# ---------------------------------------------------------------------------
def _simple_lemmatize(word: str) -> str:
    """Simple rule-based lemmatization for concept matching."""
    word = word.lower().strip()
    for suffix in ["ing", "ed", "er", "s"]:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)]
    return word


def evaluate_commongen_coverage(prediction: str, ground_truth: str) -> float:
    """
    Evaluate concept coverage for the CommonGen task.

    Args:
        prediction: Generated text.
        ground_truth: Comma-separated concept string.

    Returns:
        Fraction of ground-truth concepts found in the prediction.
    """
    if not prediction or not ground_truth:
        return 0.0

    concepts = [c.strip() for c in ground_truth.split(",") if c.strip()]
    if not concepts:
        return 0.0

    gt_lemmas = [_simple_lemmatize(c) for c in concepts]
    pred_lemmas = [_simple_lemmatize(w) for w in prediction.split()]

    matched = sum(1 for g in gt_lemmas if g in pred_lemmas)
    return matched / len(concepts)


# ---------------------------------------------------------------------------
# Embedding Dimensionality Reduction
# ---------------------------------------------------------------------------
def reduce_embedding_dim(embeddings: np.ndarray, dim: int = 50) -> np.ndarray:
    """Reduce embedding dimensionality using PCA."""
    pca = PCA(n_components=dim)
    return pca.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# LLM API Prompting
# ---------------------------------------------------------------------------
def model_prompting(
    llm_model: str,
    prompt: str,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.1,
    top_p: Optional[float] = 0.7,
) -> str:
    """
    Send a prompt to an LLM via the NVIDIA API (OpenAI-compatible).

    Uses thread-safe round-robin rotation across configured API keys
    for load balancing.

    Args:
        llm_model: Model identifier (e.g., "meta/llama-3.1-8b-instruct").
        prompt: User prompt text.
        max_token_num: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.

    Returns:
        Generated text response.
    """
    # Handle known broken model IDs
    if llm_model == "google/codegemma-7b":
        llm_model = "google/gemma-3-27b-it"

    # Thread-safe client selection
    with _client_lock:
        client = next(_client_cycle)

    if client is None:
        raise RuntimeError("No API keys configured. Set NVIDIA_API_KEYS env var.")

    try:
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_token_num,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
    except Exception as e:
        print(f"API call failed for model {llm_model}: {e}")
        raise

    return completion.choices[0].message.content
