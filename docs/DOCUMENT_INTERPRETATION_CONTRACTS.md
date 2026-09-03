# Generic document interpretation contracts

`receipt_intelligence.interpretation` defines provider-neutral input and result
contracts for evidence-backed interpretation of arbitrary documents. It does
not invoke a model, read a document format, resolve persistent entities, store
results, or define a document-type or predicate registry.

Callers pair a `DocumentSource` with their own bounded
`InterpretationSpecification` in a `DocumentInterpretationRequest`. A
specification can use flat fields or a tree of `InterpretationField` values.
The contract limits the tree to 256 nodes and eight levels so caller input is
bounded.

Classification is caller-bounded too. A specification declares one or more
`ClassificationDimension` values, their flat or hierarchical options, and
selection cardinality. Classified results identify each dimension and selected
option path; the result rejects undeclared dimensions or paths. Confidence, when
provided, is bounded from zero to one. `unsupported` remains the explicit safe
fallback when no caller-defined classification applies.

A `DocumentInterpretation` can contain:

- an explicit classified or unsupported outcome;
- a logical document map and observed mentions;
- document-scoped `CandidateEntity` values, which are not persistent resolved
  entities;
- atomic `CandidateFact` values with exactly one subject, predicate, and
  object, which are not authoritative facts;
- source evidence references with a provider-neutral locator, optional validated
  one-based page metadata, and warning or Human Review signals that can target
  candidate facts.

`LiteralValue.observed` preserves the source content as supplied. A structural
literal type distinguishes text, identifiers, dates, times, date-times, amounts,
measurements, numbers, and booleans. Normalization status distinguishes values
that were not attempted, normalized, failed, or unsafe; amount currency and
measurement units remain available without silently correcting malformed source
content. Normalized temporal values use valid ISO 8601 syntax, normalized numeric
values are finite, and normalized amounts use decimal strings to preserve exact
JSON round trips. Candidate facts refer to evidence by ID, and the result contract
rejects dangling source, evidence, mention, entity, and fact references.

## Deterministic validation boundary

The one-pass workflow returns a `DocumentInterpretationOutcome`, which keeps the
typed model interpretation and `InterpretationValidationResult` together. The
validation status is one of `valid`, `review_required`, or `invalid` and is the
authoritative deterministic acceptance state. Model-produced `ReviewSignal`
values remain unchanged on the interpretation; Core-produced validation issues
remain separate on the validation result.

Every normalized visual page is accounted for by a bounded `PageCoverage`
range. `interpreted`, `blank`, and `irrelevant` are complete handling states;
`unreadable` and `unprocessed_review_required` require review. Missing coverage
also requires review, while duplicate or source-external coverage is invalid.
Range endpoints are checked against the trusted normalized page count before
any range is enumerated.

Visual evidence may use a single page or inclusive page range. Anchors must be
ordered, source-bounded, and attached only to pages declared `interpreted`.
Evidence excerpts and mention text carry `model_observed` provenance: Core
validates their page anchors but does not claim that the text was independently
verified. Deterministic validation never guesses or repairs observed literal
values or their optional normalization.
