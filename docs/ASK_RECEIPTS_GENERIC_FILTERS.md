# Ask Your Receipts: Generic Typed Filters

## Purpose

The question-analysis contract no longer assumes that every named value is a product. Natural-language wording is translated into a small set of reusable receipt-domain constraints. New wording does not require a new filter implementation.

Examples:

```json
{
  "target_entity": "purchase_item",
  "requested_operation": "list",
  "filters": [
    {
      "filter_id": "f001",
      "field": "merchant",
      "operator": "matches",
      "value": "ARAL"
    }
  ]
}
```

```json
{
  "target_entity": "merchant",
  "requested_operation": "list",
  "filters": [
    {
      "filter_id": "f001",
      "field": "product",
      "operator": "matches",
      "value": "pizza"
    }
  ]
}
```

## Supported capabilities

The application-owned registry in `rag_sql/filter_definitions.py` is the single source of truth for:

- supported filter fields and operators;
- value kinds;
- resolution strategies;
- protected parameter suffixes;
- allowed SQL columns.

The initial registry supports product, merchant, category, purchase date, amount, payment method, currency, and receipt ID.

This is intentionally a controlled domain vocabulary. The LLM may select a registered capability, but it may not invent database fields, SQL columns, operators, or parameter names.

## Resolution flow

```text
Natural-language question
        |
        v
Question analysis v3
        |
        v
Typed QueryFilter objects
        |
        v
QueryFilterResolverRegistry
        |-- product --------> semantic retrieval + candidate resolution
        |-- merchant -------> approved canonical-value catalog
        |-- category -------> approved canonical-value catalog
        |-- payment method -> approved canonical-value catalog
        |-- currency -------> approved canonical-value catalog
        `-- date/amount/id --> deterministic scalar validation
        |
        v
ResolvedQueryFilter objects
        |
        v
Application-owned protected parameters
        |
        v
LLM SQL planning
        |
        v
Deterministic SQL/filter-binding validation
        |
        v
Read-only execution
```

For example, `ARAL`, `aral`, or a longer merchant spelling is resolved against canonical values already present in the approved analytics views. The SQL planner receives the canonical value as an immutable parameter such as `f001_merchant_0`.

## Deterministic boundary

The SQL LLM does not control filter values or their permitted storage mapping.

The application:

1. resolves and normalizes filter values;
2. creates protected named parameters;
3. rejects changed, omitted, or invented protected parameters;
4. verifies that each protected parameter constrains an approved column with the requested operator;
5. executes only through the existing read-only SQL validator and SQLite authorizer.

A merchant parameter bound to `description`, for example, is rejected and sent through the existing SQL-repair path.

## Compatibility

`rag_sql_question_analysis_v3` is the canonical contract. Existing v2 product-only payloads and Python constructors using `SemanticEntity` remain accepted and are migrated internally to product filters with legacy `e001_*` protected parameter names.

This allows a staged rollout without invalidating existing tests, saved diagnostics, or callers.

## Adding a genuinely new domain capability

A new request wording does not require code. A new domain dimension, such as a future `store_city` capability, is added once by:

1. adding one definition to the central capability registry;
2. providing its resolver/catalog source if an existing strategy is insufficient;
3. describing its natural-language meaning in the analyzer prompt;
4. exposing the approved analytics column;
5. adding contract, resolution, binding-validation, and integration tests.

The request variants “in Düsseldorf,” “from stores in Düsseldorf,” and “purchases made in Düsseldorf” would then all map to the same capability.
