import pytest

from mteb._evaluators.adversarial_paraphrase_metrics import (
    adversarial_paraphrase_accuracy,
    split_scores_by_label,
)
from mteb._evaluators.retrieval_metrics import calculate_pmrr


def test_p_mrr():
    changed_qrels = {
        "a": ["0"],
    }

    # these are the query: {"doc_id": score}
    original_run = {
        "a-og": {"0": 1, "1": 2, "2": 3, "3": 4},
    }

    new_run = {
        "a-changed": {"0": 1, "1": 2, "2": 3, "3": 4},
    }

    score = calculate_pmrr(
        original_run,
        new_run,
        changed_qrels,
    )
    assert score == 0.0

    # test with a change
    new_run = {
        "a-changed": {"0": 4, "1": 1, "2": 2, "3": 3},
    }

    score = calculate_pmrr(
        original_run,
        new_run,
        changed_qrels,
    )
    assert score == -0.75

    # test with a positive change, flipping them
    new_run = {
        "a-og": {"0": 4, "1": 1, "2": 2, "3": 3},
    }
    original_run = {
        "a-changed": {"0": 1, "1": 2, "2": 3, "3": 4},
    }
    score = calculate_pmrr(
        new_run,
        original_run,
        changed_qrels,
    )
    assert score == 0.75


def test_adversarial_paraphrase_accuracy_forced_choice():
    # anchors 0 and 2 pick the paraphrase; anchor 1 is fooled by the negative.
    positive = [0.9, 0.4, 0.8]
    negative = [0.5, 0.6, 0.2]
    assert adversarial_paraphrase_accuracy(positive, negative) == pytest.approx(2 / 3)


def test_adversarial_paraphrase_accuracy_ties_count_as_wrong():
    assert adversarial_paraphrase_accuracy([0.5], [0.5]) == 0.0


def test_adversarial_paraphrase_accuracy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        adversarial_paraphrase_accuracy([0.1, 0.2], [0.1])


def test_split_scores_by_label_preserves_order():
    positive, negative = split_scores_by_label([0.9, 0.4, 0.5, 0.6], [1, 1, 0, 0])
    assert positive == [0.9, 0.4]
    assert negative == [0.5, 0.6]
