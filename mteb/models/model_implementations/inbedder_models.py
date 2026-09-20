from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


INBEDDER_CITATION = """@article{peng2024answer,
  title={Answer is All You Need: Instruction-following Text Embedding via Answering the Question},
  author={Peng, Letian and Zhang, Yuwei and Wang, Zilong and Srinivasa, Jayanth and Liu, Gaowen and Wang, Zihan and Shang, Jingbo},
  journal={arXiv preprint arXiv:2402.09642},
  year={2024}
}"""


# QA prompt template the RoBERTa InBedder was trained and evaluated with
# (alpaca_train/train.py::QA_PROMPT_DICT and the MTEB configs in the reference
# repo). The input text comes first, then the instruction posed as a question.
QA_PROMPT = "### Input:\n{input}\n\n### Instruction:\n{instruction}\n\n### Response:"


class InBedderRobertaModel(AbsEncoder):
    """MTEB wrapper for the RoBERTa InBedder instruction-following embedder.

    Implements the "answer the question" mechanism from Peng et al. (2024),
    "Answer is All You Need" (https://arxiv.org/abs/2402.09642): the task
    instruction is treated as a *question* about the input text. Text and
    instruction are wrapped in the QA prompt template the model was trained
    with, followed by ``[MASK]`` answer slots, and the mean of the mask tokens'
    raw last-layer hidden states forms the embedding. Texts that share the same
    (implicit) answer to the instruction land close together in embedding space.

    Faithful to the paper's own MTEB pipeline (reference ``evaluation.py`` ->
    ``MaskededLMEncoder`` with ``output_value=avg_gen_layer_24``): the answer
    tokens are pooled straight from the encoder's last hidden state with no
    ``lm_head`` projection and no z-score normalization. Truncation follows the
    reference (``truncation=True`` with ``truncation_side="left"``) so long
    inputs are trimmed from the front and the trailing answer masks survive.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        n_mask: int = 3,
        max_length: int = 512,
        **_: Any,
    ) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = model_name
        self.revision = revision
        self.n_mask = n_mask
        self.max_length = max_length
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        masked_lm = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)
        # Truncate from the left so the appended answer masks (at the end of the
        # sequence) are never dropped, matching the reference tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, truncation_side="left"
        )
        # The answer representation is read straight off the encoder backbone;
        # the MLM head (projection + vocabulary decoder) is not used.
        self.model = masked_lm.roberta.to(self.device)
        self.model.eval()

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)

    def _pool_answers(
        self, hidden_states: torch.Tensor, answer_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool each example's answer-token hidden states.

        Reproduces the reference ``avg_gen_layer`` aggregation: for every
        example the last-layer hidden states at the ``[MASK]`` positions are
        averaged, with no projection or z-score normalization.
        """
        return torch.stack(
            [
                hidden_states[i][answer_mask[i]].mean(0)
                for i in range(hidden_states.size(0))
            ]
        )

    def _embed_batch(self, texts: list[str], instruction: str) -> np.ndarray:
        prompts = [
            QA_PROMPT.format(input=text, instruction=instruction)
            + self.tokenizer.mask_token * self.n_mask
            for text in texts
        ]
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        answer_mask = inputs.input_ids.eq(self.tokenizer.mask_token_id)
        hidden_states = self.model(**inputs).last_hidden_state
        embeddings = self._pool_answers(hidden_states, answer_mask)
        return embeddings.detach().cpu().float().numpy()

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **_: Any,
    ) -> Array:
        # The MTEB task instruction is the "question" InBedder answers. When a
        # task exposes no instruction the model still encodes the answer to an
        # empty question, matching the reference behavior.
        instruction = self.get_task_instruction(task_metadata, prompt_type) or ""
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for batch in inputs:
                texts = [str(text) for text in batch["text"]]
                embeddings.append(self._embed_batch(texts, instruction))
        return np.concatenate(embeddings, axis=0)


inbedder_roberta_large = ModelMeta(
    loader=InBedderRobertaModel,
    name="KomeijiForce/inbedder-roberta-large",
    model_type=["dense"],
    revision="51efa8290f751aaef1bc219f40f0b0bbec27cdab",
    release_date="2024-02-15",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["PyTorch", "Transformers"],
    n_parameters=355_412_057,
    n_embedding_parameters=52_000_768,
    memory_usage_mb=1355,
    max_tokens=512,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/KomeijiForce/inbedder-roberta-large",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    public_training_code="https://github.com/zhang-yu-wei/InBedder",
    public_training_data="https://huggingface.co/datasets/KomeijiForce/Inbedder-Pretrain-Data",
    training_datasets=None,
    adapted_from="FacebookAI/roberta-large",
    citation=INBEDDER_CITATION,
)
