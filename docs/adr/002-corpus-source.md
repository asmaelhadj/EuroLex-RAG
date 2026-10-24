# ADR 002: Corpus source (CELLAR instead of the EUR-Lex HTML page)

- Status: proposed (awaiting project-owner review)
- Date: 2026-10-22

## Context

SPEC.md Phase 1 asks for the official English text of the GDPR (`32016R0679`) and the AI Act
(`32024R1689`) from EUR-Lex. On 2026-10-22, automated requests to
`https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:...` returned
`202 Accepted` with an empty body and the header `x-amzn-waf-action: challenge`, an AWS WAF bot
challenge. The spec forbids working around access controls.

## Decision

Download from the Publications Office **CELLAR** API, the documented machine-readable access
point behind EUR-Lex, using HTTP content negotiation:

```
GET http://publications.europa.eu/resource/celex/{celex}
Accept: application/xhtml+xml
Accept-Language: eng
```

It redirects (303) to the English XHTML manifestation, which is produced by the same converter
(CONVEX) as the EUR-Lex HTML page and carries the same ELI ids. The downloader:

- never retries or evades a bot challenge; it reports it and stops;
- keeps any file already in `data/raw/`, so a **manually saved EUR-Lex page** (`gdpr_en.html`,
  `ai_act_en.html`) is always used as-is;
- writes `data/raw/manifest.json` with the SHA-256, source URL and retrieval time of each file.

The `source_url` on every chunk still points to the human-readable EUR-Lex page.

## Consequences

- `make ingest` works unattended for a stranger cloning the repo.
- CELLAR serves the text as published in the Official Journal, not a consolidated version.
  The GDPR has not been amended; any later corrigenda are not merged in.
- The AI Act's **annexes** (13, including Annex III's list of high-risk uses) are in the file but
  are not parsed yet: SPEC.md only defines article and recital chunks. Questions whose answer
  is in an annex will have no gold article. Deferred: the AI Act is disabled for now
  (ADR 004), so this is decided when it is re-enabled.
