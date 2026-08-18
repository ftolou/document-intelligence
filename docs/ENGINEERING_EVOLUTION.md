# Engineering evolution

This document records the failure modes and design decisions that shaped the current receipt
intelligence architecture. It is intentionally focused on engineering reasoning rather than a
feature chronology.

## 1. OCR was not the final problem

An earlier CRNN-based receipt project was stopped after the recognition stage because the unresolved
problem had moved beyond character recognition. Receipt semantics remained difficult: product rows,
continuations, discounts, totals, tender amounts, change and tax information could all be present in
the text without being interpreted reliably.

Modern multimodal and language models made semantic interpretation practical enough to revisit the
problem. The new objective was not simply "use an LLM", but to define boundaries around what the
model is allowed to infer and what deterministic code must still authorize.

**Decision:** treat transcription and semantic interpretation as separate stages.

## 2. Correct evidence can still produce the wrong receipt

A representative regression showed why this distinction matters. The source evidence correctly
contained:

- a purchase amount,
- a larger cash payment,
- and returned change.

A monolithic semantic parser nevertheless selected the cash tender as the receipt total. Arithmetic
validation then reported an item-sum mismatch even though the underlying text was correct.

**Failure mode:** OCR/transcription accuracy did not protect against semantic role confusion.

**Decision:** move to bounded specialist extraction and keep purchase total, payment received and
change as explicit concepts with independent validation.

## 3. One canonical evidence representation

Experiments that supplied both raw OCR boxes and model-generated transcription to the semantic
parser introduced duplicated and occasionally conflicting representations of the same receipt.

That increased the amount of context while making source authority less clear.

**Decision:** Paddle provides geometry; Qwen produces one ordered canonical transcription; semantic
stages reason from that bounded evidence representation.

Paddle-recognized text may still be useful operationally, but it is not treated as a second semantic
source of truth.

## 4. Deterministic validation must remain read-only

Early correction attempts mixed deterministic heuristics with semantic mutation. Rules that worked
for one retailer could silently damage another document layout or encode assumptions that were not
present in the source.

**Failure mode:** deterministic repair accumulated document-specific semantic guesses.

**Decision:** deterministic code validates arithmetic, consistency and contracts but does not invent
receipt meaning.

This keeps validators auditable: they answer whether a structured claim is coherent, not what the
receipt "probably meant".

## 5. Generic LLM repair was too broad

A generic repair model could return an empty patch, modify unrelated fields, infer unsupported
values or fix a different problem from the validator failure that triggered correction.

**Decision:** correction is failure-specific and source-evidence bounded.

The correction flow is:

1. Python selects the validation failure, direction, tolerance and allowed patch scope.
2. The correction model searches the bounded evidence for a supported alternative.
3. Python accepts only typed changes inside the allowed scope.
4. Validators run again.
5. The targeted failure must disappear and new regressions are rejected.

A safely accepted correction is retained even if an unrelated issue still requires human review.

## 6. Human review is part of the architecture

No extraction model is treated as the final authority for analytics. Approved data is deliberately
separated from newly extracted data.

**Decision:** human review establishes the source of truth consumed by persistence, retrieval and
RAG-SQL.

This allows the extraction system to remain probabilistic without making downstream analytics
probabilistic by default.

## 7. Retrieval resolves semantics; SQL computes

Natural-language questions introduce another model boundary. Allowing an LLM to directly invent and
execute arbitrary SQL would combine semantic uncertainty with execution authority.

**Decision:** use hybrid retrieval to resolve reviewed product concepts, let the LLM propose a typed
query plan, and let deterministic code validate and execute read-only SQL.

The LLM proposes; deterministic code authorizes.

## 8. Local-first without hard-coding one provider

The reference extraction path is intentionally local:

```text
Paddle geometry -> Qwen transcription -> Gemma extraction/correction
```

The application boundary does not assume that this is the only possible inference provider. An
optional OpenAI one-shot backend bypasses the local extraction stages and rejoins at deterministic
validation, category calibration, publication, review and persistence.

**Decision:** keep provider choice behind typed extraction/application boundaries so model changes do
not require redesigning review, storage or analytics.

## Resulting architecture principles

The current design can be summarized as:

- one canonical evidence representation;
- typed boundaries between model stages;
- deterministic validation without broad semantic mutation;
- bounded, validator-gated correction;
- human-approved data as the analytics trust boundary;
- read-only deterministic execution for RAG-SQL;
- provider-neutral observability and application contracts.

These principles are more important to the project than any individual model version.
