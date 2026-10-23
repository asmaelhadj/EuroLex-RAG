# ADR 003: Chunking strategies and chunk ids

- Status: accepted
- Date: 2026-10-22

## Decision

**Article-aware (default, E1+).** One chunk per article if it fits in `max_tokens` (512);
otherwise one chunk per numbered paragraph; a paragraph or unnumbered article that is still too
long is packed into parts at line boundaries (one line = one sentence block or list point).

| Case | Chunk id |
|---|---|
| Whole article | `gdpr:art-6` |
| Paragraph | `gdpr:art-6:p-1` |
| Part of a long paragraph | `gdpr:art-6:p-2:part-1` |
| Part of a long unnumbered article (e.g. definitions) | `gdpr:art-4:part-1` |
| Recital | `gdpr:rct-26` |

**Fixed-size (E0 baseline).** Sliding windows of 256 words with 32 words of overlap over each act
(recitals, then articles), ignoring structure. Ids are `gdpr:fixed-256-32:00042`. A window that
straddles several articles is attributed to the one that contributes the most words, so each
chunk maps to exactly one article id. Crediting every overlapped article would inflate the
baseline's recall.

Ids depend only on document structure and chunker settings, so re-runs give identical ids
(verified: two runs of `make ingest` produce byte-identical JSONL).

## Notes

- Token counts are whitespace words for now. The BGE-M3 tokenizer can be plugged in through the
  `TokenCounter` protocol in Phase 2. Subword counts run about 1.3x higher, still far below
  BGE-M3's 8192-token limit.
- Paragraph text keeps its number prefix ("1. Processing shall…") so whole-article chunks keep
  their internal structure.
- `Chunk` adds an optional `recital_number` to the SPEC.md 5.1 contract; recital chunks
  otherwise carry no locator.
