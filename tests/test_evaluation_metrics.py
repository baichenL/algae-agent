from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.services.evaluation.metrics import (
    bootstrap_ci,
    classification_metrics,
    retrieval_metrics,
    simple_regret,
    wilson_ci,
)


def test_precision_recall_mrr_and_ndcg_match_hand_calculation():
    result = retrieval_metrics(
        ["a", "x", "b", "y", "z"],
        {"a": 3, "b": 2, "c": 1},
    )
    assert result["precision_at_5"] == pytest.approx(2 / 5)
    assert result["recall_at_5"] == pytest.approx(2 / 3)
    assert result["recall_at_20"] == pytest.approx(2 / 3)
    assert result["mrr_at_10"] == 1.0
    dcg = 7 / math.log2(2) + 3 / math.log2(4)
    idcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert result["ndcg_at_10"] == pytest.approx(dcg / idcg)

    duplicate = retrieval_metrics(["a", "a", "b"], {"a": 1, "b": 1})
    assert duplicate["recall_at_5"] == 1.0
    assert duplicate["precision_at_5"] == 2 / 5


def test_macro_f1_and_confusion_matrix_match_hand_calculation():
    result = classification_metrics(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
        labels=["a", "b"],
    )
    assert result["accuracy"] == 0.75
    assert result["macro_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert result["macro_recall"] == pytest.approx((0.5 + 1.0) / 2)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]


def test_wilson_interval_known_value_and_empty_state():
    low, high = wilson_ci(50, 100)
    assert low == pytest.approx(0.4038, abs=0.0002)
    assert high == pytest.approx(0.5962, abs=0.0002)
    assert wilson_ci(0, 0) is None


def test_bootstrap_interval_is_deterministic_and_contains_mean():
    first = bootstrap_ci([1, 2, 3, 4, 5], seed=77)
    second = bootstrap_ci([1, 2, 3, 4, 5], seed=77)
    assert first == second
    assert first[0] <= 3.0 <= first[1]


def test_simple_regret_is_objective_gap():
    assert simple_regret(10.0, 8.25) == 1.75


def test_interview_rag_candidate_has_50_complete_stable_cases():
    path = Path(__file__).parent / "fixtures" / "interview_rag_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    corpus_root = Path(__file__).parent / "fixtures" / "rag_gold_corpus"
    assert len(cases) == 50
    assert len({case["id"] for case in cases}) == 50
    category_counts = {}
    for case in cases:
        category_counts[case["category"]] = category_counts.get(case["category"], 0) + 1
        assert case["question"]
        assert isinstance(case["relevant_evidence_set"], list)
        assert case["reference_answer"]
        assert case["atomic_claims"]
        assert case["expected_status"]
        assert case["capability_tags"]
        assert case["difficulty"] in {"easy", "medium", "hard"}
        assert "human_reviewed" in case
        for evidence in case["relevant_evidence_set"]:
            assert evidence["evidence_id"]
            assert evidence["source_locator"]
            assert evidence["relevance_grade"] in {0, 1, 2, 3}
            assert "chunk_id" not in evidence
            relative_path, anchor = evidence["source_locator"].split("#", 1)
            source = corpus_root / Path(relative_path)
            assert source.is_file(), evidence["source_locator"]
            headings = {
                line.removeprefix("# ").strip().casefold()
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.startswith("# ")
            }
            assert anchor.casefold() in headings, evidence["source_locator"]
    assert sorted(category_counts.values()) == [10, 10, 10, 10, 10]
