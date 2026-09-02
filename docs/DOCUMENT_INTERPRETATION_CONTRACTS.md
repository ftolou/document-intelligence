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
- bounded `PageInterpretation` ranges that account for pages as interpreted,
  blank, irrelevant, unreadable, or explicitly unprocessed;
- a logical document map and observed mentions;
- document-scoped `CandidateEntity` values, which are not persistent resolved
  entities;
- atomic `CandidateFact` values with exactly one subject, predicate, and
  object, which are not authoritative facts;
- source evidence references with a provider-neutral locator, optional validated
  one-based page or page-range metadata, and warning or Human Review signals
  that can target candidate facts.

For visual sources, an evidence excerpt is explicitly tagged
`model_observed`. Core validates its source/page anchor but does not claim that
the reported characters were independently verified. Trusted normalized text
verification would require a separate explicit source capability.

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

`OnePassDocumentInterpreter.interpret` returns one
`DocumentInterpretationOutcome`. Its `interpretation` is the immutable typed
model result. Its `validation` is Core's authoritative deterministic state:
`VALID`, `REVIEW_REQUIRED`, or `INVALID`.

`ValidationIssue` values have deterministic Core provenance and remain separate
from model-produced `ReviewSignal` values. A model review-required signal makes
the authoritative validation state `REVIEW_REQUIRED` without copying that
signal into validation issues. Validation never changes, drops, replaces, or
truncates model signals.

The validator checks the typed interpretation against the trusted normalized
page count. Missing coverage, unreadable pages, and explicitly unprocessed pages
require review. Duplicate or out-of-bounds coverage, invalid evidence ranges,
evidence on non-interpreted pages, and missing required evidence are invalid.
Blank and irrelevant pages are explicit, complete coverage and do not by
themselves require review.

Page-range endpoints are compared with the trusted source bounds before any
enumeration. A model-controlled range such as `1..1000000000` on a one-page
source therefore produces one bounded invalid finding without materializing the
claimed range. Contract construction remains the sole owner of #16's atomic
fact, literal, normalization, confidence, and closed cross-reference invariants;
the deterministic validator does not reimplement them or guess repairs.
