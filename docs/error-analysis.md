# Error analysis

Ten failure cases from real runs of the final pipeline (E1s retrieval, qwen2.5:7b-instruct,
threshold 0.5003). Every case comes from a saved run or a logged manual test; nothing is
invented. Cases marked **human** come from the 20-answer human review
(`data/eval/answer_review.csv`). The others were found by reading run output and still need
a human check (SPEC.md section 10, task 3): **Reviewed by a human: TODO.**

Runs: `20261027-035624_e7_generation_dev` (dev) and `20261027-051913_e7_threshold_test`
(test). Scores are cosine similarities of the top retrieved article.

| # | Question | Stage | What went wrong |
|---|---|---|---|
| 1 | q-0114 | retrieval | definition article never retrieved |
| 2 | q-0137 | generation | refused although the right article was ranked first |
| 3 | several | retrieval | one long chunk (Art. 47(2)) ranks high for unrelated questions |
| 4 | q-0018 | citation | irrelevant passage cited next to the right one (human) |
| 5 | q-0022 | citation | right rule, wrong article cited (human) |
| 6 | q-0122 | retrieval + citation | neighbouring articles cited instead of Art. 14 (human) |
| 7 | q-0062 | generation | rule stated without its exceptions (human) |
| 8 | q-0126 | generation | a sentence no passage supports (human) |
| 9 | q-0051, q-0052, q-0061 | refusal | out-of-scope questions passed the threshold |
| 10 | "what is gdpr" | refusal | short casual question refused by the threshold |

## 1. Definition article not retrieved (q-0114)

"What is the difference between a data controller and a data processor?" Gold: Art. 4. The
top five chunks were Art. 28(1), Art. 38, Art. 28(3), Art. 29 and Art. 28(4) (scores 0.606
to 0.591); Art. 4 was not in the top 10. Art. 4 holds 26 definitions split into a few long
chunks, so each chunk is about many terms and matches no single question strongly, while
Art. 28 talks about controllers and processors throughout. The model then refused, which
was the safe outcome given the passages.

**Idea:** index each Art. 4 definition as its own chunk.

## 2. Refusal although the gold article was first (q-0137)

"Can I submit an individual complaint directly to the EDPB?" Gold: Art. 77, ranked first
(0.525). The model replied with the fixed "cannot answer" sentence. Art. 77 gives the right
to complain to a supervisory authority but never mentions the EDPB, so the model found no
passage on the exact point. The right answer ("no, to a national supervisory authority")
needs one step of reasoning from what the passage leaves out. This was one of the two false
refusals on dev.

**Idea:** a prompt rule for "the passages say where, and it is not where you asked" cases;
needs its own evaluation, since it could also make the model answer when it should not.

## 3. A "hub" chunk that matches many questions (Art. 47(2))

`gdpr:art-47:p-2:part-1` (the list of what binding corporate rules must contain) appeared
9 times among the top five chunks of the 60 dev questions, and Art. 47 was in the top five
articles of 11 questions, none of which have it as gold. It was ranked first for q-0059 (data processing agreements) and q-0126 (restricting
processing). The list mentions data subjects' rights, complaints, liability, processors and
training, so it is a little similar to almost everything.

**Idea:** split long list paragraphs per point, or down-weight chunks that score high for
many unrelated queries.

## 4. Irrelevant passage cited next to the right one (q-0018, human)

"How long does an organisation generally have to respond to an individual's rights request?"
The answer is correct (one month, extendable by two, Art. 12(3)) but cites [2][3], and [3]
is Art. 66, the urgency procedure for supervisory authorities, which mentions "three
months". The automatic citation check passed because [3] was a passage the model was given.
8 of the 20 reviewed answers had a citation problem of this kind, while automatic citation
validity was 1.000.

**Idea:** check that each cited passage shares key terms with its sentence.

## 5. Right rule, wrong article (q-0022, human)

"What should a controller do when its processor discovers a personal data breach?" The
processor's duty to tell the controller is Art. 33(2), which was passage [1]. The answer
cited [3] (Art. 28(3)) and [2] (Art. 34) for it, and Recital 86 (informing data subjects)
for notifying the supervisory authority. The content was right; the sources were not.

## 6. Neighbouring articles instead of the gold one (q-0122, human)

"What information should I receive if a company obtained my data from another source?"
Gold: Art. 14. Retrieval ranked Art. 15, then Art. 13, above Art. 14, because Arts. 13, 14
and 15 list almost the same items. The answer cited Art. 13 and 15 for Art. 14's content,
and its list missed legitimate interests (Art. 14(2)(b)), withdrawal of consent, automated
decision-making and third-country transfers. Human verdict: partly correct.

## 7. A rule without its exceptions (q-0062, human)

"Are there any restrictions on the type of personal data I can collect from my employees?"
The answer stated Art. 9(1) (special categories are prohibited) as an absolute ban and did
not mention the Art. 9(2) exceptions, such as 9(2)(b) for employment law. For an employer
that is misleading. Art. 5 (data minimisation), one of the three gold articles, was not
retrieved.

## 8. An unsupported sentence (q-0126, human)

"Can I ask a company to stop using my data temporarily while a dispute is resolved?" The
answer added "this is subject to the data being stored only for the duration necessary to
resolve the dispute", which no passage says, and missed Art. 18(1)(d), the ground that fits
a pending objection best. Art. 47 was ranked first (case 3) and cited. The LLM judge gave
this answer 5/5 for faithfulness.

## 9. Out-of-scope questions above the threshold (test: q-0051, q-0052, q-0061)

On the test split the threshold refused 6 of 9 out-of-scope questions. Three scored above
0.5003: the CCPA (0.534), the EU AI Act (0.634) and newsletter frequency (0.504). All three
use GDPR-like words (privacy, personal data, subscribers). qwen refused all three, so the
final refusal accuracy was still 9/9. On dev the margin was already small (in scope at least
0.508, out of scope at most 0.493), so this was expected.

## 10. A short casual question refused (manual test, 2026-10-27)

In the UI, "what is gdpr" was refused by the threshold: top score 0.460. Rephrased, the same
question passes: "What is the GDPR?" 0.627, "What is the General Data Protection Regulation
and what does it cover?" 0.619 (both measured with the served pipeline). The GDPR text
never uses the word "GDPR", and a very short lowercase query has little to match. The dev
set has almost no short, casual questions, so the threshold was never tested on them.

**Idea:** query rewriting or abbreviation expansion, evaluated on dev plus a set of short
questions written by a human (see ADR 009 and the E5 results).

## What the cases have in common

- Retrieval: long multi-topic chunks (Art. 4, Art. 47(2)) and near-identical articles
  (13/14/15) are the main source of wrong context.
- Citations: the automatic check only proves a citation points at a given passage. Whether
  the passage supports the sentence needed a human.
- The LLM judge (mistral) scored every reviewed answer 5/5, so it found none of cases 4-8.
