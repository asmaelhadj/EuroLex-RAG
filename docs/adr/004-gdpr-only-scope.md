# ADR 004: GDPR-only corpus for now

- Status: accepted (project owner, 2026-10-24)
- Date: 2026-10-24

## Context

SPEC.md puts the GDPR first and the AI Act second. The AI Act raises open questions the GDPR
does not: 13 annexes that SPEC.md has no chunk kind for (Annex III, the high-risk list, is central
to many questions), staged application dates, and fewer public FAQs to build questions from.

## Decision

Index, evaluate and serve the **GDPR only**. The AI Act stays in `configs/ingest.yaml` with
`enabled: false`. Its parser support and count check (113 articles, 180 recitals) are already
verified, so it is ready when we extend.

## Consequences

- All experiments E0 to E7 and the question set cover the GDPR only; the README and results must
  say so.
- The annex decision (ADR 002, consequences) is deferred until the AI Act is enabled.
- Re-enabling the AI Act means: set `enabled: true`, decide on annexes, re-run `make ingest`,
  and extend the question set. Results from a GDPR-only run must not be mixed with results
  from a larger corpus.
