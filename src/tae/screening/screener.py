from __future__ import annotations

from dataclasses import dataclass

from tae.connectors.yahoo import CompanyProfile
from tae.scoring.engine import ScoreResult


@dataclass(frozen=True)
class ScreenerRow:
    ticker: str
    company: str | None
    industry: str | None
    sector: str | None
    short_term_score: float
    medium_term_score: float
    long_term_score: float
    overall_score: float
    risk_score: float
    recommendation: str


def build_screener_row(profile: CompanyProfile, score: ScoreResult) -> ScreenerRow:
    return ScreenerRow(
        ticker=profile.ticker,
        company=profile.name,
        industry=profile.industry,
        sector=profile.sector,
        short_term_score=score.short_score,
        medium_term_score=score.medium_score,
        long_term_score=score.long_score,
        overall_score=score.overall_score,
        risk_score=score.risk_score,
        recommendation=score.recommendation,
    )

