from mteb.models.answer_logit_pooling import InstructAnswerEncoderWrapper
from mteb.models.model_meta import ModelMeta, ScoringFunction

# InBedder treats the instruction as a question and encodes the answer the model
# predicts at appended [MASK] slots. See mteb/models/answer_logit_pooling.py.
INBEDDER_RELEASE_DATE = "2024-02-15"

roberta_large_inbedder = ModelMeta(
    loader=InstructAnswerEncoderWrapper,
    loader_kwargs=dict(
        n_answer_tokens=3,
        max_seq_length=512,
    ),
    name="BrandonZYW/roberta-large-InBedder",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="f95f96e4a59e0f6b39fd5a3f4a1b4d74823ed03f",
    release_date=INBEDDER_RELEASE_DATE,
    framework=["PyTorch", "Transformers"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    reference="https://huggingface.co/BrandonZYW/roberta-large-InBedder",
    n_parameters=355_359_744,
    n_embedding_parameters=51_471_360,  # 50265 vocab * 1024 hidden
    memory_usage_mb=None,
    embed_dim=50265,
    license="mit",
    max_tokens=512,
    adapted_from="FacebookAI/roberta-large",
    citation="""@inproceedings{peng-etal-2024-answer,
      title={Answer is All You Need: Instruction-following Text Embedding via Answering the Question},
      author={Peng, Letian and Zhang, Yuwei and Wang, Zilong and Srinivasa, Jayanth and Liu, Gaowen and Wang, Zihan and Shang, Jingbo},
      booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
      year={2024},
      url={https://arxiv.org/abs/2402.09642}
    }""",
    public_training_code="https://github.com/zhang-yu-wei/InBedder",
    public_training_data=None,
    training_datasets=None,
)
