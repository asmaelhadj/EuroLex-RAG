# ADR 009: Answer generation and refusal

- Status: accepted
- Date: 2026-10-27

## Answer and judge models

- Answers: `qwen2.5:7b-instruct` through Ollama (SPEC.md default).
- Judge: `mistral:latest`. It differs from the answer model, as SPEC.md requires. It also
  generated some of the candidate questions; it only grades, it never answers.
- Query rewriting (E5) used `llama3.2:3b`, so no model both wrote questions and answers them.

## Prompt and injection defence

- Retrieved passages go in numbered `<passage n=".." source="..">` blocks, after a system
  prompt that says they are data, not instructions.
- `<` and `>` in passage text and in the question become `(` and `)`, so a passage cannot
  close its block or open a fake one.
- After generation, `[n]` markers that point at no passage are removed and counted.
- Tests: 7 injection cases with a mocked model that obeys any instruction it can see outside
  a passage block. With the bracket defence disabled, 5 of them fail, which shows the tests
  check something real. 3 opt-in tests run the same idea against the real model
  (`RUN_LLM_TESTS=1`); a mock cannot tell whether a real model obeys an instruction that
  stays inside its block.

## Context given to the model

The top 8 ranked passages (articles, since the final pipeline ranks articles only) plus the
3 support recitals (ADR 008). Articles are the citations; recitals explain them.

## Evidence threshold

SPEC.md puts the threshold on the top reranker score. The final retrieval candidate (E1s,
ADR 008) has no reranker, so the threshold is on its top score: the cosine similarity of the
best article. On the E1s dev run:

| Group | n | Top score |
|---|---|---|
| In scope, gold article retrieved | 59 | 0.508 to 0.795 |
| In scope, gold missed | 1 | 0.606 |
| Out of scope | 15 | 0.274 to 0.493 |

`choose_threshold` takes the cut with the best balanced accuracy and, among ties, the widest
margin: **0.5003**. On dev it answers all answerable questions and refuses all out-of-scope
ones, but the margin is small (0.493 vs 0.508), and most dev out-of-scope questions are easy
(weather, maths). Expect some errors on the test split. Plot:
`docs/figures/refusal_threshold_dev.png`, made by
`python -m eurolex.evaluation.threshold runs/20261026-203041_e1s_articles_first_dev`.

Refusals have two causes, both counted: `low_evidence` (threshold, no LLM call) and `model`
(the model replies with the fixed "cannot answer" sentence).

## Metrics (runner, generation on)

Refusal accuracy on out-of-scope questions, false refusal rate on in-scope ones, citation
validity (valid markers / all markers), share of in-scope answers that cite a gold article,
and judge scores for faithfulness and relevance (1 to 5). E7 compares `e7_generation` (no
threshold) with `e7_threshold`.
