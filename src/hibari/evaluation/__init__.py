"""Evaluation — text metrics (``metrics``) and model-level loss / perplexity (``evaluator``)."""

from hibari.evaluation.evaluator import evaluate_records
from hibari.evaluation.metrics import (
    char_ngram_f,
    distinct_n,
    echo_ratio,
    grammar_heuristic,
    ngram_repetition,
    repeated_sentence,
)

__all__ = ["evaluate_records", "char_ngram_f", "distinct_n", "echo_ratio", "grammar_heuristic",
           "ngram_repetition", "repeated_sentence"]
