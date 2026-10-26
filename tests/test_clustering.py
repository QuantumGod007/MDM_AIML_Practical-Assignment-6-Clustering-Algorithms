import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clustering_models import load_dataset, standardize, internal_metrics, run_experiment

def test_dataset_properties():
    features, labels = load_dataset()
    assert features.shape == (178, 13)
    assert len(labels) == 178
    assert not features.isna().any().any()

def test_scaling():
    features, _ = load_dataset()
    x = standardize(features)
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(x.std(axis=0), 1, atol=1e-12)

def test_single_cluster_rejected():
    assert internal_metrics(np.ones((8, 3)), np.zeros(8)) is None

def test_reproducible_results(tmp_path):
    metrics, summary, models = run_experiment(tmp_path, make_plots=False)
    assert len(metrics) == 4
    assert summary["recommended_model"] == "K-Means"
    for _, labels, _ in models.values():
        assert len(labels) == 178
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "cluster_assignments.csv").exists()

def test_dbscan_noise_not_scored_as_cluster():
    x = np.array([[0.0], [0.1], [3.0], [3.1], [50.0]])
    result = internal_metrics(x, [0, 0, 1, 1, -1], exclude_noise=True)
    assert result["Noise"] == 1
    assert result["Coverage"] == pytest.approx(.8)
    assert result["Clusters"] == 2
