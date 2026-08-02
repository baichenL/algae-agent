from __future__ import annotations

import math
import random
from collections import Counter
from statistics import mean
from typing import Callable, Iterable, Sequence


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def wilson_ci(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return None
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total)
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 2025,
) -> tuple[float, float] | None:
    """Non-parametric percentile bootstrap interval with a fixed seed."""
    observed = [float(value) for value in values]
    if not observed:
        return None
    if len(observed) == 1:
        return observed[0], observed[0]
    rng = random.Random(seed)
    estimates = [
        float(statistic([observed[rng.randrange(len(observed))] for _ in observed]))
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    low = percentile(estimates, alpha)
    high = percentile(estimates, 1.0 - alpha)
    return (float(low), float(high)) if low is not None and high is not None else None


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 2025,
) -> tuple[float, float] | None:
    if len(left) != len(right):
        raise ValueError("paired samples must have the same length")
    return bootstrap_ci(
        [float(a) - float(b) for a, b in zip(left, right)],
        samples=samples,
        seed=seed,
    )


def bootstrap_classification_ci(
    expected: Sequence[str],
    predicted: Sequence[str],
    metric_key: str,
    *,
    labels: Sequence[str] | None = None,
    samples: int = 2000,
    seed: int = 2025,
) -> tuple[float, float] | None:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        indexes = [rng.randrange(len(expected)) for _ in expected]
        result = classification_metrics(
            [expected[index] for index in indexes],
            [predicted[index] for index in indexes],
            labels=labels,
        )
        value = result.get(metric_key)
        if value is not None:
            estimates.append(float(value))
    low = percentile(estimates, 0.025)
    high = percentile(estimates, 0.975)
    return (float(low), float(high)) if low is not None and high is not None else None


def bootstrap_binary_classification_ci(
    expected: Sequence[bool],
    predicted: Sequence[bool],
    metric_key: str,
    *,
    samples: int = 2000,
    seed: int = 2025,
) -> tuple[float, float] | None:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        indexes = [rng.randrange(len(expected)) for _ in expected]
        expected_sample = [bool(expected[index]) for index in indexes]
        predicted_sample = [bool(predicted[index]) for index in indexes]
        tp = sum(a and b for a, b in zip(expected_sample, predicted_sample))
        fp = sum(not a and b for a, b in zip(expected_sample, predicted_sample))
        fn = sum(a and not b for a, b in zip(expected_sample, predicted_sample))
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = (
            safe_ratio(2.0 * precision * recall, precision + recall)
            if precision is not None and recall is not None
            else None
        )
        value = {"precision": precision, "recall": recall, "f1": f1}.get(metric_key)
        if value is not None:
            estimates.append(float(value))
    low = percentile(estimates, 0.025)
    high = percentile(estimates, 0.975)
    return (float(low), float(high)) if low is not None and high is not None else None


def classification_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> dict:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    ordered_labels = list(labels or sorted(set(expected) | set(predicted)))
    confusion = {
        actual: {prediction: 0 for prediction in ordered_labels}
        for actual in ordered_labels
    }
    for actual, prediction in zip(expected, predicted):
        if actual not in confusion:
            confusion[actual] = {item: 0 for item in ordered_labels}
        if prediction not in confusion[actual]:
            for row in confusion.values():
                row.setdefault(prediction, 0)
            if prediction not in ordered_labels:
                ordered_labels.append(prediction)
        confusion[actual][prediction] += 1

    per_class: dict[str, dict[str, float]] = {}
    for label in ordered_labels:
        true_positive = confusion.get(label, {}).get(label, 0)
        false_positive = sum(
            confusion.get(actual, {}).get(label, 0)
            for actual in ordered_labels
            if actual != label
        )
        false_negative = sum(
            confusion.get(label, {}).get(prediction, 0)
            for prediction in ordered_labels
            if prediction != label
        )
        precision = safe_ratio(true_positive, true_positive + false_positive) or 0.0
        recall = safe_ratio(true_positive, true_positive + false_negative) or 0.0
        f1 = safe_ratio(2.0 * precision * recall, precision + recall) or 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion.get(label, {}).values()),
        }

    total = len(expected)
    correct = sum(1 for actual, prediction in zip(expected, predicted) if actual == prediction)
    return {
        "sample_count": total,
        "correct_count": correct,
        "accuracy": safe_ratio(correct, total),
        "macro_precision": mean(item["precision"] for item in per_class.values()) if per_class else None,
        "macro_recall": mean(item["recall"] for item in per_class.values()) if per_class else None,
        "macro_f1": mean(item["f1"] for item in per_class.values()) if per_class else None,
        "labels": ordered_labels,
        "confusion_matrix": [
            [confusion.get(actual, {}).get(prediction, 0) for prediction in ordered_labels]
            for actual in ordered_labels
        ],
        "per_class": per_class,
    }


def retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevance_grades: dict[str, int | float],
    *,
    precision_k: int = 5,
    recall_ks: Sequence[int] = (5, 20),
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> dict[str, float | int | None]:
    """BEIR-compatible binary P/R/MRR and graded nDCG for a single query."""
    relevant_ids = {item_id for item_id, grade in relevance_grades.items() if float(grade) > 0}
    unique_retrieved: list[str] = []
    seen: set[str] = set()
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in seen:
            # Preserve the rank slot but never credit duplicate evidence twice.
            unique_retrieved.append(f"__duplicate__:{rank}:{item_id}")
            continue
        seen.add(item_id)
        unique_retrieved.append(item_id)

    def relevant_count(k: int) -> int:
        return sum(1 for item_id in unique_retrieved[:k] if item_id in relevant_ids)

    result: dict[str, float | int | None] = {
        f"precision_at_{precision_k}": relevant_count(precision_k) / precision_k,
        f"hit_rate_at_{precision_k}": float(relevant_count(precision_k) > 0),
        "relevant_count": len(relevant_ids),
    }
    for k in recall_ks:
        result[f"recall_at_{k}"] = (
            relevant_count(k) / len(relevant_ids) if relevant_ids else None
        )

    reciprocal_rank = 0.0
    for rank, item_id in enumerate(unique_retrieved[:mrr_k], start=1):
        if item_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break
    result[f"mrr_at_{mrr_k}"] = reciprocal_rank if relevant_ids else None

    gains = [float(relevance_grades.get(item_id, 0)) for item_id in unique_retrieved[:ndcg_k]]
    dcg = sum((2.0**gain - 1.0) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_gains = sorted((float(value) for value in relevance_grades.values()), reverse=True)[:ndcg_k]
    idcg = sum(
        (2.0**gain - 1.0) / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, start=1)
    )
    result[f"ndcg_at_{ndcg_k}"] = dcg / idcg if idcg > 0 else None
    return result


def ranking_metrics(ranked_labels: Sequence[str], relevant_labels: set[str]) -> dict[str, float | None]:
    if not relevant_labels:
        return {"hit_at_1": None, "hit_at_3": None, "mrr": None}
    first_rank = next(
        (rank for rank, label in enumerate(ranked_labels, start=1) if label in relevant_labels),
        None,
    )
    return {
        "hit_at_1": float(first_rank == 1),
        "hit_at_3": float(first_rank is not None and first_rank <= 3),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def mean_absolute_error(expected: Sequence[float], predicted: Sequence[float]) -> float | None:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    return mean(abs(float(a) - float(b)) for a, b in zip(expected, predicted)) if expected else None


def root_mean_squared_error(expected: Sequence[float], predicted: Sequence[float]) -> float | None:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    return (
        math.sqrt(mean((float(a) - float(b)) ** 2 for a, b in zip(expected, predicted)))
        if expected
        else None
    )


def simple_regret(optimum: float, selected_best: float) -> float:
    """Best achievable objective minus the best selected objective."""
    return float(optimum) - float(selected_best)


def cohen_kappa(expected: Sequence[str | int | bool], predicted: Sequence[str | int | bool]) -> float | None:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        return None
    total = len(expected)
    observed = sum(a == b for a, b in zip(expected, predicted)) / total
    expected_counts = Counter(expected)
    predicted_counts = Counter(predicted)
    chance = sum(
        (expected_counts[label] / total) * (predicted_counts[label] / total)
        for label in set(expected_counts) | set(predicted_counts)
    )
    return (observed - chance) / (1.0 - chance) if chance < 1.0 else 1.0
