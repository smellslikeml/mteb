"""Adversarial paraphrase-selection scoring.

Implements the forced-choice accuracy used by the ParaLux benchmark introduced
in *LuxEmbedder: A Cross-Lingual Approach to Enhanced Luxembourgish Sentence
Embeddings* (Philippy et al., 2024; https://arxiv.org/abs/2412.03331).

For each anchor sentence the model must assign a higher similarity to the true
paraphrase than to an adversarial, minimally edited hard negative. Because the
negatives are near-paraphrases, a single global decision threshold (as used by
average precision in standard pair classification) is easily saturated and hides
whether the model actually ranks the correct paraphrase first. The forced-choice
accuracy below is threshold-free and reported per anchor, which is what makes the
benchmark discriminative for low-resource languages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def adversarial_paraphrase_accuracy(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> float:
    """Fraction of anchors whose paraphrase outscores its adversarial negative.

    Args:
        positive_scores: ``similarity(anchor_i, paraphrase_i)`` for each anchor i.
        negative_scores: ``similarity(anchor_i, not_paraphrase_i)``, aligned by i.

    Returns:
        The forced-choice accuracy in ``[0, 1]``. Ties count as incorrect,
        matching the strict ``>`` comparison used by the paper.
    """
    if len(positive_scores) != len(negative_scores):
        raise ValueError(
            "positive_scores and negative_scores must have equal length; "
            f"got {len(positive_scores)} and {len(negative_scores)}."
        )
    if not positive_scores:
        raise ValueError("Cannot compute accuracy over an empty set of anchors.")
    correct = sum(1 for p, n in zip(positive_scores, negative_scores) if p > n)
    return correct / len(positive_scores)


def split_scores_by_label(
    scores: Sequence[float],
    labels: Sequence[int],
) -> tuple[list[float], list[float]]:
    """Split aligned pair scores into paraphrase (label 1) and adversarial (label 0).

    The two returned lists preserve input order, so the k-th positive and the
    k-th negative share the same anchor when the pairs were built anchor-aligned
    (all paraphrase pairs first, all adversarial pairs second).

    Args:
        scores: One similarity score per ``(anchor, candidate)`` pair.
        labels: Matching labels, ``1`` for a paraphrase pair and ``0`` otherwise.

    Returns:
        A ``(positive_scores, negative_scores)`` tuple.
    """
    positive = [s for s, y in zip(scores, labels) if int(y) == 1]
    negative = [s for s, y in zip(scores, labels) if int(y) == 0]
    return positive, negative
