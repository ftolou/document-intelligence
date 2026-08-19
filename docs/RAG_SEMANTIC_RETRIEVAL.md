# RAG hybrid item retrieval

The receipt database remains the authoritative source. The semantic index is a derived,
rebuildable representation of approved purchase items.

## Retrieval flow

```text
User product concept
→ query embedding
→ dense cosine ranking over reviewed item semantics
→ product-name lexical ranking (FTS5 + compound matching)
→ weighted reciprocal-rank fusion
→ product-identity deduplication
→ evidence-based candidate resolution
```

The dense item document now uses embedding policy
`approved_product_semantics_v2`:

```text
Document type: purchased product
Product description: VITTEL
Reviewed category: Food & Groceries / beverages
Reviewed semantic description: Vittel is a brand of mineral water.
```

The reviewed category and semantic description provide the missing bridge between a printed
brand or abbreviation and the product concept a user asks for. For example, `water` can retrieve
`VITTEL` even when the printed receipt row contains no literal `water` token.

Prices, quantities, dates, merchant names, currency, and receipt totals remain structured SQL
values and are excluded from dense semantic similarity. Product-name lexical scoring remains
separate and retains higher default RRF weight, so exact product names are not replaced by broad
category similarity.

## Embedding provider boundary

Semantic indexing and retrieval depend on the provider-neutral `EmbeddingGateway` application
port. Provider adapters translate their native APIs into the same validated `EmbeddingBatchResult`
contract. The core currently provides adapters for Ollama and the OpenAI embeddings API.

The local reference application keeps Ollama as its default composition. Other deployments can
inject a different gateway without changing item-document construction, indexing, hybrid scoring,
or candidate resolution. Provider credentials and deployment-specific provider selection therefore
stay outside the RAG algorithms.

A single semantic index must use one compatible embedding model and vector dimension. Stored item
vectors and query vectors must be generated with the same model policy. Changing the active model
or output dimension requires rebuilding the derived embedding index; it does not change the
approved receipt data that remains the source of truth.

Provider request diagnostics are exposed through `model_calls`. The previous `ollama_calls` Python
attribute remains available as a compatibility alias for existing callers.

## Reviewed semantic evidence

`category_reason` is treated as a reviewed semantic description after receipt approval. It should
describe what the product is or does:

```text
Vittel is a brand of mineral water.
Elmex is toothpaste used for oral hygiene.
```

Avoid classifier-process prose such as "the model selected this category because...". The review
editor remains the place to correct this field before it is indexed.

The candidate resolver receives the same reviewed semantic description as
`semantic_description_reviewed`. A specific reviewed statement may establish
`strong_contextual` evidence for an unfamiliar brand or abbreviation. Category and merchant remain
low-trust supporting metadata and cannot establish a match by themselves.

## Rebuild the index once after Step 7.3.3

The embedding policy version is part of each content hash. Existing vectors therefore become stale
automatically after applying v1.27.8.

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/rebuild_rag_item_index.py
```

`--force` is not required. The incremental indexer detects the policy/hash change and re-embeds all
eligible approved items once.

## Selective updates after review

The database editor invalidates and re-embeds only affected item IDs when any embedded field changes:

- product description or normalized name,
- parser item type,
- reviewed category path,
- reviewed category reason / semantic description.

Changes to merchant, date, quantity, price, currency, VAT, totals, or review notes do not trigger a
new vector. Embedding failure never rolls back the committed database edit; the stale vector remains
removed and the incremental indexer can repair it later.

## Search manually

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/search_rag_items.py "water" --limit 10
```

The result includes dense and lexical ranks, fused RRF score, grouped item IDs, occurrence count,
category, and the reviewed semantic description used by candidate resolution.

## Configuration

```env
RAG_RETRIEVAL_RRF_K=60
RAG_RETRIEVAL_VECTOR_WEIGHT=1.0
RAG_RETRIEVAL_LEXICAL_WEIGHT=1.5
RAG_RETRIEVAL_DEDUPLICATE=1
```

The lexical branch remains product-name focused and has the larger default weight. The enriched
dense branch improves conceptual recall without replacing deterministic SQL filtering or the LLM
candidate-resolution boundary.
