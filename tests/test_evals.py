"""Smoke tests for the eval harness — the harness itself must be trustworthy."""

import pytest

from evals.harness import run_eval, format_report, approx_tokens


@pytest.fixture(scope="module")
def metrics():
    return run_eval(k=5)


class TestEvalHarness:
    def test_metrics_shape(self, metrics):
        assert metrics["n_questions"] >= 10
        assert set(metrics["retrieval_by_type"]) == {
            "keyword", "paraphrase", "graph_1hop", "graph_2hop"
        }

    def test_metric_ranges(self, metrics):
        r = metrics["retrieval"]
        for key in ("recall_at_1", "recall_at_5", "mrr"):
            assert 0.0 <= r[key] <= 1.0
        assert r["recall_at_5"] >= r["recall_at_1"]

    def test_full_context_control(self, metrics):
        assert metrics["full_context"]["gold_in_context_rate"] == 1.0
        # Control arm must cost more tokens than retrieval — that's its point
        assert metrics["full_context"]["mean_tokens"] > metrics["retrieval"]["mean_tokens"]

    def test_regression_floors(self, metrics):
        # Floors lock in the PPR + RRF-strength gains; deterministic corpus
        assert metrics["retrieval"]["recall_at_5"] >= 0.9
        assert metrics["retrieval_by_type"]["keyword"]["recall_at_5"] == 1.0
        assert metrics["retrieval_by_type"]["graph_2hop"]["recall_at_5"] == 1.0

    def test_report_renders(self, metrics):
        report = format_report(metrics)
        assert "retrieval (all)" in report
        assert "full-context control" in report

    def test_approx_tokens(self):
        assert approx_tokens("") == 1
        assert approx_tokens("a" * 400) == 100

    def test_deterministic(self, metrics):
        again = run_eval(k=5)
        assert again["retrieval"] == metrics["retrieval"]
