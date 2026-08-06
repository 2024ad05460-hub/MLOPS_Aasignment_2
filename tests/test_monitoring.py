from cats_dogs_mlops.monitoring import PredictionStore


def test_feedback_updates_post_deployment_accuracy(tmp_path) -> None:
    store = PredictionStore(tmp_path / "predictions.db")
    store.record_prediction("request-0001", "cat", 0.9, {"cat": 0.9, "dog": 0.1}, 12.3)
    store.record_prediction("request-0002", "dog", 0.8, {"cat": 0.2, "dog": 0.8}, 15.0)
    assert store.add_feedback("request-0001", "cat")
    assert store.add_feedback("request-0002", "cat")
    summary = store.performance()
    assert summary.labeled_count == 2
    assert summary.correct_count == 1
    assert summary.accuracy == 0.5
