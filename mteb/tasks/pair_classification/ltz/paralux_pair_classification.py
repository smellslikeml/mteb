from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import Dataset

from mteb._evaluators import PairClassificationEvaluator
from mteb._evaluators.adversarial_paraphrase_metrics import (
    adversarial_paraphrase_accuracy,
    split_scores_by_label,
)
from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.models_protocols import EncoderProtocol

if TYPE_CHECKING:
    from pathlib import Path

    from mteb.models.models_protocols import MTEBModels
    from mteb.timing import TimingStack
    from mteb.types import EncodeKwargs


class ParaLuxPairClassification(AbsTaskPairClassification):
    """Luxembourgish adversarial paraphrase detection (ParaLux).

    Each example is an ``(anchor, paraphrase, not_paraphrase)`` triple. The
    triples are reshaped into aligned pair-classification rows (all paraphrase
    pairs first, all adversarial pairs second) so the standard pair-classification
    metrics still apply, while ``adversarial_accuracy`` reports the paper's
    forced-choice metric: how often the paraphrase outscores its hard negative.
    """

    metadata = TaskMetadata(
        name="ParaLuxPairClassification",
        description=(
            "ParaLux is a Luxembourgish paraphrase-detection benchmark of "
            "adversarial triples. For each anchor sentence the model must select "
            "the valid paraphrase over a minimally edited, adversarial hard "
            "negative."
        ),
        reference="https://arxiv.org/abs/2412.03331",
        dataset={
            "path": "fredxlpy/ParaLux",
            "revision": "e73a9d222c47572921573ff6ad96cf75442a7083",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ltz-Latn"],
        main_score="adversarial_accuracy",
        date=("2024-01-01", "2024-12-05"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{philippy2024luxembedder,
  author = {Philippy, Fred and Guo, Siwen and Lothritz, Cedric and Klein, Jacques and Bissyandé, Tegawendé F.},
  journal = {arXiv preprint arXiv:2412.03331},
  title = {LuxEmbedder: A Cross-Lingual Approach to Enhanced Luxembourgish Sentence Embeddings},
  year = {2024},
}
""",
    )

    def dataset_transform(self, num_proc: int | None = None) -> None:
        _dataset = {}
        for split in self.metadata.eval_splits:
            split_data = self.dataset[split]
            anchors = list(split_data["anchor"])
            paraphrases = list(split_data["paraphrase"])
            adversarials = list(split_data["not_paraphrase"])
            _dataset[split] = [
                {
                    "sentence1": anchors + anchors,
                    "sentence2": paraphrases + adversarials,
                    "labels": [1] * len(anchors) + [0] * len(adversarials),
                }
            ]
        self.dataset = _dataset

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        timer: TimingStack,
        **kwargs: Any,
    ) -> dict[str, float]:
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        if self.metadata.modalities == ["text"]:
            # for compatibility with v1 version where datasets were stored in a single row
            data_split = (
                Dataset.from_dict(data_split[0]) if len(data_split) == 1 else data_split
            )
        evaluator = PairClassificationEvaluator(
            data_split,
            input1_column_name=self.input1_column_name,
            input2_column_name=self.input2_column_name,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            input1_prompt_type=self.input1_prompt_type,
            input2_prompt_type=self.input2_prompt_type,
            timer=timer,
            **kwargs,
        )
        similarity_scores = evaluator(
            model,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

        if prediction_folder:
            self._save_task_predictions(
                similarity_scores,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        labels = data_split[self.label_column_name]
        scores = self._compute_metrics(similarity_scores, labels)
        # Paper's headline metric: adversarial forced-choice paraphrase selection.
        positive, negative = split_scores_by_label(
            similarity_scores["cosine_scores"], labels
        )
        scores["adversarial_accuracy"] = adversarial_paraphrase_accuracy(
            positive, negative
        )
        return scores
