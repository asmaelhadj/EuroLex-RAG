"""Evidence threshold: refuse before calling the LLM when retrieval found nothing strong."""

import itertools
from collections.abc import Sequence

from eurolex.schemas import RetrievalResult


def low_evidence(result: RetrievalResult, threshold: float | None) -> bool:
    if threshold is None:
        return False
    return not result.hits or result.hits[0].score < threshold


def choose_threshold(answerable: Sequence[float], unanswerable: Sequence[float]) -> float:
    """Threshold with the best balanced accuracy on dev; ties go to the widest gap.

    `answerable` are top scores of in-scope questions whose gold article was retrieved,
    `unanswerable` those of out-of-scope questions. Scores at or above the threshold are
    answered.
    """
    if not answerable or not unanswerable:
        raise ValueError("need scores for both groups")
    candidates = sorted(set(answerable) | set(unanswerable))
    # midpoints between neighbouring scores, plus both ends
    cuts = (
        [candidates[0] - 1e-9]
        + [(a + b) / 2 for a, b in itertools.pairwise(candidates)]
        + [candidates[-1] + 1e-9]
    )

    def balanced_accuracy(t: float) -> float:
        answered = sum(s >= t for s in answerable) / len(answerable)
        refused = sum(s < t for s in unanswerable) / len(unanswerable)
        return (answered + refused) / 2

    def margin(t: float) -> float:
        return min(abs(s - t) for s in [*answerable, *unanswerable])

    return max(cuts, key=lambda t: (balanced_accuracy(t), margin(t)))
