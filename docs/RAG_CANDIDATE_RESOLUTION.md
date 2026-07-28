# RAG Steps 5.2–5.3: LLM candidate resolution

Step 5.2 added the precision layer between hybrid retrieval and future SQL generation.
Step 5.3 refines only its decision contract and prompt policy. It does **not** change the
application architecture or the current Ask Your Receipts query engine.

```text
Semantic product phrase
→ hybrid vector + lexical retrieval
→ product-identity deduplication
→ evidence-based LLM candidate resolution
→ resolved item IDs / clarification / not found
```

## Why this layer exists

Hybrid retrieval is intentionally recall-oriented. It can place the correct products near the top
while still returning unrelated vector-only candidates. The resolver asks the local LLM to classify
each retrieved product identity as:

- `selected`
- `uncertain`
- `rejected`

The resolver returns one explicit state:

- `resolved`
- `needs_clarification`
- `not_found`

There is no deterministic semantic fallback. If the model cannot return a valid, complete
classification after the configured attempts, the resolver raises an error.

## Step 5.3 evidence contract

The previous free-form numerical `confidence` field is replaced by categorical evidence:

- `explicit`: the description directly names the requested concept or an unmistakable subtype.
- `strong_contextual`: the description itself or a specific reviewed semantic description strongly
  identifies a recognized synonymous product, brand/model, subtype, or purpose.
- `ambiguous`: the description is truncated, unfamiliar, accessory-like, service-like, or depends
  materially on low-trust metadata.
- `unrelated`: the description clearly denotes a different product.

The contract enforces this mapping:

```text
selected  → explicit | strong_contextual
uncertain → ambiguous
rejected  → unrelated
```

Ambiguous descriptions cannot be silently selected. They produce `needs_clarification` and a concise
scope question.

## Evidence policy

The printed/normalized product description remains primary evidence. Step 7.3.3 also supplies the
approved `category_reason` as `semantic_description_reviewed`. A specific reviewed statement such as
`Vittel is a brand of mineral water` may establish strong contextual evidence for a `water` query.

Category and merchant remain available as `category_low_trust` and `merchant_low_trust`; they may only
corroborate a conclusion and cannot establish one by themselves.

They must not:

- establish selection on their own,
- promote an unfamiliar or truncated description to selected,
- imply that every product from a specialized merchant belongs to the merchant's primary category.

Retrieval ranks and similarity scores are not passed as semantic evidence and are not probabilities.

## Structured result example

```json
{
  "schema_version": "rag_candidate_resolution_v2",
  "status": "needs_clarification",
  "semantic_entity": "Schuhe",
  "decisions": [
    {
      "candidate_id": "c001",
      "decision": "uncertain",
      "evidence_strength": "ambiguous",
      "evidence": "The description is shoe-related but does not clearly identify footwear rather than an accessory."
    },
    {
      "candidate_id": "c002",
      "decision": "selected",
      "evidence_strength": "explicit",
      "evidence": "'Halbschuhe' directly identifies a footwear subtype."
    }
  ],
  "clarification_question": "Should ambiguous shoe-related products or accessories also be included?",
  "notes": []
}
```

After validation, the application maps stable candidate IDs back to the exact SQL `item_ids` retained
by the deduplicated retrieval result.

## Placeholder filtering

Null-like and generic descriptions such as `None`, `Unknown`, and `Product Purchase` remain excluded
from embeddings and candidate-resolution prompts.

Step 7.3.3 changes the canonical embedding policy. Run the incremental item-index rebuild once after
applying v1.27.8; the policy-version content hash detects every stale vector automatically.

## Manual test

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/resolve_rag_candidates.py "Schuhe" `
  --question "Wie viel habe ich für Schuhe ausgegeben?"
```

The output contains the compact retrieval candidates and the structured resolution, including the
selected, uncertain, and rejected SQL item IDs.

## Configuration

```env
RAG_CANDIDATE_RESOLVER_ENABLED=1
RAG_CANDIDATE_MODEL=gemma4:latest
RAG_CANDIDATE_MAX_CANDIDATES=12
RAG_CANDIDATE_NUM_CTX=4096
RAG_CANDIDATE_NUM_PREDICT=1536
RAG_CANDIDATE_TIMEOUT_SECONDS=120
RAG_CANDIDATE_RETRY_COUNT=1
RAG_CANDIDATE_FORMAT_JSON=1
RAG_CANDIDATE_KEEP_ALIVE=
```

## Scope

This remains a reusable entity-resolution contract only. Schema RAG, validated SQL examples,
read-only SQL generation, SQL validation, and query-engine integration remain future steps.
