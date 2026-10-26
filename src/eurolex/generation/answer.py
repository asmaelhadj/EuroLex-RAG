"""Retrieve, decide whether to answer, ask the LLM, check its citations."""

import time

from eurolex.generation.citations import check_citations, resolve
from eurolex.generation.llm import LLMClient
from eurolex.generation.prompts import CANNOT_ANSWER, SYSTEM_PROMPT, build_prompt
from eurolex.generation.refusal import low_evidence
from eurolex.retrieval.pipeline import RetrievalPipeline
from eurolex.schemas import Answer

LOW_EVIDENCE_ANSWER = (
    "I could not find GDPR provisions that clearly answer this question, so I will not guess. "
    "This is general information, not legal advice."
)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class AnswerGenerator:
    def __init__(
        self,
        pipeline: RetrievalPipeline,
        llm: LLMClient,
        *,
        context_k: int = 8,
        threshold: float | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._llm = llm
        self._context_k = context_k
        self._threshold = threshold

    def answer(self, question: str, act_id: str | None = None) -> Answer:
        start = time.perf_counter()
        retrieval = self._pipeline.run(question, act_id=act_id)
        latency = {"retrieval": _ms(start)}

        if low_evidence(retrieval, self._threshold):
            # refuse without calling the LLM
            return Answer(
                question=question,
                answer=LOW_EVIDENCE_ANSWER,
                refused=True,
                refusal_reason="low_evidence",
                retrieval=retrieval,
                latency_ms={**latency, "total": latency["retrieval"]},
            )

        passages = [*retrieval.hits[: self._context_k], *retrieval.support]
        start = time.perf_counter()
        raw = self._llm.generate(build_prompt(question, passages), system=SYSTEM_PROMPT)
        latency["generation"] = _ms(start)
        latency["total"] = round(latency["retrieval"] + latency["generation"], 2)

        check = check_citations(raw.strip(), len(passages))
        refused = check.text.startswith(CANNOT_ANSWER)
        return Answer(
            question=question,
            answer=check.text,
            refused=refused,
            refusal_reason="model" if refused else None,
            citations=resolve(check.valid, passages),
            invalid_citations=check.invalid,
            passages=passages,
            retrieval=retrieval,
            latency_ms=latency,
        )
