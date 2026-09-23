"""Tests for the ParaLux adversarial paraphrase-selection integration.

Exercises both the registration wiring (the task is discoverable through the
public ``mteb`` package) and the paper's forced-choice metric flowing through
the standard pair-classification evaluation path.
"""

from __future__ import annotations

import pytest

import mteb
from mteb._evaluators.adversarial_paraphrase_selection import (
    adversarial_paraphrase_accuracy,
    split_scores_by_label,
)
from mteb.tasks.pair_classification.ltz import ParaLuxPairClassification


def test_task_is_registered():
    """The new task is reachable through the existing registry (call-site edit)."""
    task = mteb.get_task("ParaLuxPairClassification")

    assert isinstance(task, ParaLuxPairClassification)
    assert task.metadata.type == "PairClassification"
    assert task.metadata.eval_langs == ["ltz-Latn"]
    assert task.metadata.main_score == "adversarial_accuracy"
    assert task.metadata.license == "cc-by-nc-4.0"


def test_dataset_transform_builds_aligned_pairs():
    """Triples become anchor-aligned pairs: positives first, negatives second."""
    task = ParaLuxPairClassification()
    task.dataset = {
        "test": {
            "anchor": ["a1", "a2"],
            "paraphrase": ["p1", "p2"],
            "not_paraphrase": ["n1", "n2"],
        }
    }

    task.dataset_transform()

    row = task.dataset["test"][0]
    assert row["sentence1"] == ["a1", "a2", "a1", "a2"]
    assert row["sentence2"] == ["p1", "p2", "n1", "n2"]
    assert row["labels"] == [1, 1, 0, 0]


def test_evaluate_reports_adversarial_accuracy():
    """End-to-end: the wired metric shows up as the task's main score."""
    task = ParaLuxPairClassification()
    # Inject an already-transformed synthetic split to avoid a network download.
    task.dataset = {
        "test": [
            {
                "sentence1": ["the cat sat", "a dog ran", "the cat sat", "a dog ran"],
                "sentence2": [
                    "the cat is sitting",
                    "a dog is running",
                    "the cat stood",
                    "a cat ran",
                ],
                "labels": [1, 1, 0, 0],
            }
        ]
    }
    task.data_loaded = True

    model = mteb.get_model("mteb/baseline-random-encoder")
    results = mteb.evaluate(model, task, cache=None, co2_tracker=False)

    scores = results[0].scores["test"][0]
    assert "adversarial_accuracy" in scores
    assert 0.0 <= scores["adversarial_accuracy"] <= 1.0
    # main_score is wired to the paper's forced-choice metric.
    assert scores["main_score"] == scores["adversarial_accuracy"]
    # standard pair-classification metrics still run alongside it.
    assert "max_ap" in scores


def test_adversarial_paraphrase_accuracy_forced_choice():
    # anchors 0 and 2 pick the paraphrase; anchor 1 is fooled by the negative.
    positive = [0.9, 0.4, 0.8]
    negative = [0.5, 0.6, 0.2]
    assert adversarial_paraphrase_accuracy(positive, negative) == pytest.approx(2 / 3)


def test_adversarial_paraphrase_accuracy_ties_count_as_wrong():
    assert adversarial_paraphrase_accuracy([0.5], [0.5]) == 0.0


def test_split_scores_by_label_preserves_order():
    positive, negative = split_scores_by_label([0.9, 0.4, 0.5, 0.6], [1, 1, 0, 0])
    assert positive == [0.9, 0.4]
    assert negative == [0.5, 0.6]


def test_adversarial_paraphrase_accuracy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        adversarial_paraphrase_accuracy([0.1, 0.2], [0.1])
