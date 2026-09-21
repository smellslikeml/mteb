import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

import mteb
from mteb import AbsTask, EncoderProtocol, TaskMetadata
from mteb.models import DimensionalityReductionWrapper
from tests.task_grid import MOCK_TASK_TEST_GRID_MONOLINGUAL

task_metadata = TaskMetadata(
    name="DummyTask",
    description="Dummy task metadata",
    dataset={"path": "test", "revision": "test"},
    type="Retrieval",
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_10",
)

# Simulate data
task_texts = DataLoader(
    Dataset.from_dict(
        {
            "text": [
                "What is the effect of vitamin C on common cold?",
                "How does exercise affect cardiovascular health?",
            ]
        }
    )
)


@pytest.mark.parametrize("method", ["truncate", "gaussian"])
def test_reduces_dimensionality(method: str):
    model = mteb.get_model("mteb/baseline-random-encoder")
    n_components = 8
    wrapper = DimensionalityReductionWrapper(
        model, n_components=n_components, method=method
    )
    embeddings = wrapper.encode(
        task_texts,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="test",
    )
    assert embeddings.shape[-1] == n_components


def test_gaussian_projection_is_consistent_across_calls():
    # Queries and corpus are encoded in separate calls; the projection basis
    # must be identical so the reduced spaces remain comparable.
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = DimensionalityReductionWrapper(model, n_components=8, method="gaussian")
    kwargs = dict(task_metadata=task_metadata, hf_split="test", hf_subset="test")
    first = wrapper.encode(task_texts, **kwargs)
    second = wrapper.encode(task_texts, **kwargs)
    assert torch.allclose(torch.as_tensor(first), torch.as_tensor(second))


def test_reduction_updates_model_meta():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = DimensionalityReductionWrapper(model, n_components=16, method="gaussian")
    assert wrapper.mteb_model_meta.embed_dim == 16
    assert wrapper.mteb_model_meta.experiment_kwargs["reduction_method"] == "gaussian"
    assert wrapper.mteb_model_meta.experiment_kwargs["n_components"] == 16


@pytest.mark.parametrize("n_components", [0, -4])
def test_invalid_n_components(n_components: int):
    model = mteb.get_model("mteb/baseline-random-encoder")
    with pytest.raises(ValueError):
        DimensionalityReductionWrapper(model, n_components=n_components)


def test_invalid_method():
    model = mteb.get_model("mteb/baseline-random-encoder")
    with pytest.raises(ValueError):
        DimensionalityReductionWrapper(model, n_components=8, method="pca")


def test_n_components_exceeding_dim_raises():
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = DimensionalityReductionWrapper(model, n_components=100_000)
    with pytest.raises(ValueError):
        wrapper.encode(
            task_texts,
            task_metadata=task_metadata,
            hf_split="test",
            hf_subset="test",
        )


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID_MONOLINGUAL)
@pytest.mark.parametrize("method", ["truncate", "gaussian"])
def test_evaluate_with_reduction_on_task(task: AbsTask, method: str):
    model: EncoderProtocol = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = DimensionalityReductionWrapper(model, n_components=8, method=method)
    mteb.evaluate(wrapper, task, cache=None)
