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


class InBedderRobertaModel(AbsEncoder):
    """MTEB wrapper for the RoBERTa InBedder instruction-following embedder.

    Implements the "answer the question" mechanism from Peng et al. (2024),
    "Answer is All You Need" (https://arxiv.org/abs/2402.09642): the task
    instruction is treated as a *question* about the input text. It is appended
    to the text as a masked-language-modeling prompt whose answer slots are
    ``[MASK]`` tokens, and the pooled hidden states of those answer tokens --
    passed through the MLM head's ``dense -> gelu -> layer_norm`` projection --
    form the embedding. Texts that share the same (implicit) answer to the
    instruction land close together in embedding space.

    Faithful to the reference encode() published on
    https://huggingface.co/KomeijiForce/inbedder-roberta-large . The only
    deviation is ``truncation="only_first"``: truncation is restricted to the
    input text so the appended answer masks are never dropped, which keeps the
    per-example mask count fixed and the answer-pooling reshape well defined.
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        # The answer representation is read from the encoder backbone and the
        # MLM head's projection (dense + layer_norm); the vocabulary decoder is
        # not needed.
        self.model = masked_lm.roberta.to(self.device)
        self.dense = masked_lm.lm_head.dense.to(self.device)
        self.layer_norm = masked_lm.lm_head.layer_norm.to(self.device)
        self.model.eval()

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)
        self.dense.to(device)
        self.layer_norm.to(device)

    def _pool_answers(self, answer_states: torch.Tensor, n_texts: int) -> torch.Tensor:
        """Mean-pool the per-example answer tokens then z-score normalize.

        ``answer_states`` holds the ``n_texts * n_mask`` answer-token vectors in
        row order; they are grouped per example, averaged across the masks, and
        standardized (mean-centered, unit std) exactly as in the reference.
        """
        pooled = answer_states.reshape(n_texts, self.n_mask, -1).mean(1)
        return (pooled - pooled.mean(1, keepdim=True)) / pooled.std(1, keepdim=True)

    def _embed_batch(self, texts: list[str], instruction: str) -> np.ndarray:
        from torch.nn.functional import gelu

        prompts = [instruction + self.tokenizer.mask_token * self.n_mask for _ in texts]
        inputs = self.tokenizer(
            texts,
            prompts,
            padding=True,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        mask = inputs.input_ids.eq(self.tokenizer.mask_token_id)
        answer_states = self.model(**inputs).last_hidden_state[mask]
        answer_states = self.layer_norm(gelu(self.dense(answer_states)))
        embeddings = self._pool_answers(answer_states, len(texts))
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
