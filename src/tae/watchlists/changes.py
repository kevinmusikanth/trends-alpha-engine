def score_change_label(
    previous_score: float | None,
    current_score: float,
    threshold: float = 3,
) -> str:
    if previous_score is None:
        return "New opportunity"
    delta = current_score - previous_score
    if delta >= threshold:
        return "Score increasing"
    if delta <= -threshold:
        return "Score decreasing"
    return "Stable"
