from __future__ import annotations

from dataclasses import dataclass

from comparator import ComparisonReport


@dataclass(frozen=True)
class ScoreResult:
    score: float
    max_score: float
    earned_score: float
    missing_penalty: float
    hallucination_penalty: float


def score_report(
    report: ComparisonReport,
    hallucination_penalty_per_field: float = 2.5,
    required_missing_multiplier: float = 1.5,
) -> ScoreResult:
    max_score = sum(field.weight for field in report.fields)
    earned = 0.0
    missing_penalty = 0.0

    for field in report.fields:
        earned += field.weight * field.credit
        if field.status == "missing" and field.required:
            # Required fields already earn zero credit when missing. This extra
            # penalty makes required semantic loss hurt more than optional loss.
            missing_penalty += field.weight * (required_missing_multiplier - 1.0)

    # Hallucinated fields reduce trust even if the schema fields were present.
    hallucination_penalty = len(report.hallucinated_fields) * hallucination_penalty_per_field
    adjusted = max(0.0, earned - missing_penalty - hallucination_penalty)
    score = round((adjusted / max_score) * 100, 2) if max_score else 0.0

    return ScoreResult(
        score=score,
        max_score=max_score,
        earned_score=round(earned, 2),
        missing_penalty=round(missing_penalty, 2),
        hallucination_penalty=round(hallucination_penalty, 2),
    )
