"""InBedder: instruction-following text embeddings via answering the question.

Adapted from "Answer is All You Need: Instruction-following Text Embedding via
Answering the Question" (Peng et al., 2024, https://arxiv.org/abs/2402.09642).

The core mechanism ported here is InBedder's "answer the question" view of an
embedding: the user instruction is treated as a *question* about the input
text, the model produces an answer, and the answer representation -- not the raw
text -- becomes the embedding. Two texts that would elicit the same answer to
the instruction therefore land close together in embedding space, which is what
makes the representation follow the instruction.

For the released RoBERTa checkpoint the "answer" is realised as masked-language
tokens appended after the question: the encoder mean-pools the final hidden
states over those answer (mask) positions. Training (the paper's instruction-QA
fine-tuning) and the paper's own evaluation suite are intentionally out of scope
-- this integration wires the released encoder into MTEB's evaluation contract so
the model can be run on MTEB tasks with ``use_instructions=True``.
"""

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


def build_answer_prompt(
    text: str,
    instruction: str,
    mask_token: str,
    num_answer_tokens: int,
) -> str:
    """Format a text + instruction as an InBedder question-with-answer prompt.

    The text is followed by the instruction (the "question") and a slot of
    ``num_answer_tokens`` answer (mask) tokens that the model fills in. When no
    instruction is available the prompt degrades to text + answer slot.

    Args:
        text: The input text to embed.
        instruction: The instruction interpreted as a question about the text.
        mask_token: The tokenizer's mask token used to mark the answer slot.
        num_answer_tokens: Number of answer (mask) tokens to append.

    Returns:
        The formatted prompt string.
    """
    answer_slot = " ".join([mask_token] * max(num_answer_tokens, 1))
    if instruction:
        return f"{text}\n\n{instruction} {answer_slot}"
    return f"{text} {answer_slot}"


def mean_pool_answer_tokens(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Mean-pool hidden states over the answer (mask) positions.

    Implements InBedder's answer pooling: only the positions holding answer
    tokens contribute to the embedding. Rows with no answer token fall back to a
    denominator of one, yielding a zero vector rather than a division by zero.

    Args:
        hidden_states: Final hidden states of shape ``(batch, seq_len, hidden)``.
        input_ids: Token ids of shape ``(batch, seq_len)``.
        mask_token_id: Id of the answer (mask) token.

    Returns:
        Pooled answer embeddings of shape ``(batch, hidden)``.
    """
    answer_mask = (input_ids == mask_token_id).unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * answer_mask).sum(dim=1)
    counts = answer_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class InBedderModel(AbsEncoder):
    """Encoder that embeds the answer to an instruction, following InBedder.

    The instruction is treated as a question about the input text; the model
    fills an answer slot of mask tokens and the mean-pooled hidden states over
    those positions are returned as the embedding.

    To guarantee the appended answer tokens survive truncation of long inputs,
    the tokenizer truncates from the left (dropping the start of the text rather
    than the trailing question + answer slot).
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        device: str | None = None,
        num_answer_tokens: int = 3,
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        """Load the InBedder masked-LM encoder.

        Args:
            model_name: Hugging Face model id of the InBedder checkpoint.
            revision: Model revision (commit hash) to load.
            device: Device to place the model on. Defaults to cuda if available.
            num_answer_tokens: Number of answer (mask) tokens appended per input.
            max_length: Maximum tokenized sequence length.
            **kwargs: Extra keyword arguments forwarded to ``from_pretrained``.
        """
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = model_name
        self.num_answer_tokens = num_answer_tokens
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, revision=revision, **kwargs
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        # Keep the trailing question + answer slot when truncating long inputs.
        self.tokenizer.truncation_side = "left"

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Array:
        """Encode inputs by answering the task instruction about each text.

        Args:
            inputs: Batched inputs to encode.
            task_metadata: Metadata of the task, used to resolve the instruction.
            hf_split: Current dataset split.
            hf_subset: Current dataset subset.
            prompt_type: Whether the input is a query or a document.
            batch_size: Number of prompts encoded per forward pass.
            **kwargs: Ignored additional arguments.

        Returns:
            Answer embeddings as a numpy array of shape ``(n_inputs, hidden)``.
        """
        instruction = self.get_instruction(task_metadata, prompt_type)
        mask_token = self.tokenizer.mask_token or "<mask>"
        texts = [text for batch in inputs for text in batch["text"]]
        prompts = [
            build_answer_prompt(text, instruction, mask_token, self.num_answer_tokens)
            for text in texts
        ]

        embeddings = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoding = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoding, output_hidden_states=True)
            pooled = mean_pool_answer_tokens(
                outputs.hidden_states[-1],
                encoding["input_ids"],
                self.tokenizer.mask_token_id,
            )
            embeddings.append(pooled.cpu().float().numpy())

        return np.concatenate(embeddings, axis=0)


INBEDDER_CITATION = """@article{peng2024answer,
  title={Answer is All You Need: Instruction-following Text Embedding via Answering the Question},
  author={Peng, Letian and Zhang, Yuwei and Wang, Zilong and Srinivasa, Jayanth and Liu, Gaowen and Wang, Zihan and Shang, Jingbo},
  journal={arXiv preprint arXiv:2402.09642},
  year={2024}
}"""

inbedder_roberta_large = ModelMeta(
    loader=InBedderModel,
    loader_kwargs=dict(num_answer_tokens=3, max_length=512),
    name="KomeijiForce/inbedder-roberta-large",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="51efa8290f751aaef1bc219f40f0b0bbec27cdab",
    release_date="2024-02-29",
    n_parameters=int(3.55 * 1e8),
    # roberta-large embeddings: 50265*1024 (word) + 514*1024 (pos) + 1*1024
    # (type) + 2*1024 (LayerNorm).
    n_embedding_parameters=52_000_768,
    memory_usage_mb=None,
    embed_dim=1024,
    max_tokens=514,
    license="mit",
    reference="https://huggingface.co/KomeijiForce/inbedder-roberta-large",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=True,
    adapted_from="FacebookAI/roberta-large",
    public_training_code="https://github.com/zhang-yu-wei/InBedder",
    public_training_data=None,
    training_datasets=None,
    citation=INBEDDER_CITATION,
)
