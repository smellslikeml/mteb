import pytest

import mteb
from mteb._evaluators import PairClassificationEvaluator
from mteb.tasks.pair_classification.ltz import ParaLuxPairClassification
from mteb.timing import TimingStack
from tests.mock_tasks import (
    MockPairClassificationTask,
)

TOL = 0.0001


class TestPairClassificationEvaluator:
    def test_accuracy(self):  # noqa: PLR6301
        task = MockPairClassificationTask()
        task.load_data()

        evaluator = PairClassificationEvaluator(
            task.dataset["test"],
            input1_column_name="sentence1",
            input2_column_name="sentence2",
            task_metadata=task.metadata,
            hf_split="test",
            hf_subset="test",
            input1_prompt_type=None,
            input2_prompt_type=None,
            timer=TimingStack(),
        )
        distances = evaluator(
            mteb.get_model("mteb/baseline-random-encoder"),
            encode_kwargs={"batch_size": 32},
        )
        assert distances["cosine_scores"] == pytest.approx(
            [0.7375020980834961, 0.7731508016586304], TOL
        )
        assert distances["euclidean_distances"] == pytest.approx(
            [2.4108424186706543, 2.1905980110168457], TOL
        )
        assert distances["manhattan_distances"] == pytest.approx(
            [11.177837371826172, 10.721406936645508], TOL
        )
        assert distances["similarity_scores"] == pytest.approx(
            [0.7375020384788513, 0.7731509208679199], TOL
        )
        assert distances["dot_scores"] == pytest.approx(
            [7.974165916442871, 8.176445960998535], TOL
        )


class TestParaLuxPairClassification:
    """ParaLux flows its adversarial forced-choice metric through this evaluator."""

    def test_task_is_registered(self):  # noqa: PLR6301
        task = mteb.get_task("ParaLuxPairClassification")

        assert isinstance(task, ParaLuxPairClassification)
        assert task.metadata.type == "PairClassification"
        assert task.metadata.eval_langs == ["ltz-Latn"]
        assert task.metadata.main_score == "adversarial_accuracy"
        assert task.metadata.license == "cc-by-nc-4.0"

    def test_dataset_transform_builds_aligned_pairs(self):  # noqa: PLR6301
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

    def test_evaluate_reports_adversarial_accuracy(self):  # noqa: PLR6301
        """End-to-end: the wired metric shows up as the task's main score."""
        task = ParaLuxPairClassification()
        # Inject an already-transformed synthetic split to avoid a network download.
        task.dataset = {
            "test": [
                {
                    "sentence1": [
                        "the cat sat",
                        "a dog ran",
                        "the cat sat",
                        "a dog ran",
                    ],
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
