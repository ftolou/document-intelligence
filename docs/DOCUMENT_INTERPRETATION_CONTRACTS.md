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

A `DocumentInterpretation` can contain:

- an explicit classified or unsupported outcome;
- a logical document map and observed mentions;
- document-scoped `CandidateEntity` values, which are not persistent resolved
  entities;
- atomic `CandidateFact` values with exactly one subject, predicate, and
  object, which are not authoritative facts;
- source evidence references and warning or Human Review signals.

`LiteralValue.observed` preserves the source content as supplied. Its
`normalized` value is optional, so ambiguous or malformed observations remain
representable without silent correction. Candidate facts refer to evidence by
ID, and the result contract rejects dangling source, evidence, mention, entity,
and fact references.
