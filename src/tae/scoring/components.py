from __future__ import annotations

from dataclasses import dataclass


def clip(value: float, low: float = 0, high: float = 1) -> float:
    return max(low, min(high, value))


def score_positive(value: float | None, target: float, weight: float) -> float:
    if value is None or target == 0:
        return 0.0
    return clip(value / target) * weight


def score_inverse(value: float | None, target: float, weight: float) -> float:
    if value is None or target == 0:
        return 0.0
    return clip(1 - (value / target)) * weight


@dataclass(frozen=True)
class ComponentScore:
    name: str
    weight: float
    score: float

    def as_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "weight": self.weight, "score": round(self.score, 2)}

