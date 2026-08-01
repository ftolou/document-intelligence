# Receipt pipeline v7.7 — validator-gated correction coordinator

This bundle integrates the correction experiments into the main receipt pipeline
as an explicit, configurable correction stage. The correction stage runs after
initial receipt assembly and deterministic validation and before semantic-status
derivation and persistence.

The objective is not to run a second unrestricted parser. Python selects a
specific failed validation target and an allowed strategy chain. Gemma performs
bounded source-evidence interpretation or a bounded generic patch request.
Python validates the evidence, constructs or validates typed patches, applies a
candidate to a copy of the last accepted receipt, reruns the complete
validator, and accepts the candidate only when the target disappears without
regressions.

## Production flow

```text
Qwen transcription
    -> Gemma scalar and item extraction
    -> receipt assembly
    -> initial deterministic validation
    -> correction coordinator
         -> failure-specific specialist strategy
         -> bounded generic auto-patch fallback when required
         -> evidence and patch validation
         -> candidate revalidation
         -> atomic accept or discard
    -> final deterministic validation
    -> semantic status / human-review decision
    -> final receipt
```

Accepted corrections are cumulative. Each new strategy starts from the last
accepted state. A rejected candidate is discarded and is never used as input to
the next strategy. When one correction is accepted but unrelated failures
remain, the accepted correction is retained and the receipt remains marked for
review.

## Enabled strategy routes

The production profile is stored in:

```text
correction/config/production.json
```

The principal routes are:

- `ITEM_SUM_RECONCILIATION`
  1. V3 source-item-block recovery
  2. final-total source-evidence recovery
  3. bounded generic auto-patching
- VAT validation failures
  1. VAT v9 source-evidence recovery
  2. bounded generic auto-patching
- total-related reconciliation failures
  1. final-total v2.4 source-evidence recovery
  2. bounded generic auto-patching
- registered remaining failures
  1. bounded generic auto-patching

The generic fallback runs only when no preceding specialist has already
resolved the selected target.

## Correction package

```text
correction/
    profile.py               # configuration contracts and loader
    coordinator.py           # routing, attempts, accepted-state lifecycle
    acceptance.py            # target-resolution and regression gate
    patching.py               # bounded targets, patch schema and application
    evidence.py               # shared literal source-evidence helpers
    strategies/
        item_sum.py           # V3 item evidence -> typed item mutations
        vat.py                # VAT v9 evidence -> typed VAT mutations
        final_total.py        # v2.4 total evidence -> typed total mutation
    config/
        production.json       # enabled routes, prompts, attempts, patch limits
```

## Prompt architecture

Correction prompt text and output schemas are not embedded in Python modules.
They are immutable, hash-verified artifacts in the prompt registry:

```text
prompts/
    manifest.json
    gemma/correction/
        item_sum_source_blocks/v1.0.0/
            template.txt
            schema.json
            metadata.json
        vat_source_evidence/v1.0.0/
            prompt.md
            schema.json
            metadata.json
        final_total_source_evidence/v1.0.0/
            prompt.md
            schema.json
            metadata.json
        generic/v1.0.0/
            prompt.md
            metadata.json
```

The correction profile references prompt IDs and versions. `PromptRegistry`
verifies both prompt and schema hashes before use. Every accepted correction
report records the prompt reference and model metrics.

## Safety and acceptance rules

A candidate is accepted only when all of the following hold:

- the model response satisfies the applicable evidence or patch contract;
- literal values and row references are grounded in the canonical source text;
- every patch operation and path is within the Python-selected scope;
- generic fallback values newly introduced by a patch occur in canonical source
  evidence; unchanged object fields may be preserved;
- generic array removal is limited to exact duplicate elements;
- the selected validation target becomes `passed` or `observed`;
- no previously passed validation check regresses;
- no non-dependency failure is introduced;
- failed-check count and severity do not worsen;
- skipped-check count does not increase.

Specialist strategies do not derive values from reconciliation residuals:

- item recovery inserts source-supported items or changes a uniquely matched
  item `final_price`;
- VAT recovery uses explicit printed VAT evidence;
- final-total recovery uses one explicitly labelled printed total.

## Receipt output artifacts

Each processed receipt writes:

```text
89_receipt_structured_initial.json
89_deterministic_validation_initial.json
90_gemma_correction_report.json
90_correction_round_...json
91_deterministic_validation_final.json
92_receipt_combined_final.json
```

The initial structured receipt is retained so future prompt evaluation runs can
use the actual pre-correction state. The evaluation exporters support both the
new v7.7 filenames and the legacy v7.6 filenames.

## Main script

```text
experiment_batch_paddle_snapped_crops_qwen35_gemma_items_scalars_v7_7_correction_coordinator.py
```

Run help and unit tests:

```bash
python experiment_batch_paddle_snapped_crops_qwen35_gemma_items_scalars_v7_7_correction_coordinator.py --help
python -m unittest discover -s tests -v
```

Relevant runtime controls include:

```text
--gemma-correction / --no-gemma-correction
--gemma-correction-max-rounds
--gemma-correction-retries
--gemma-correction-num-predict
--item-sum-recovery-num-predict
--vat-recovery-num-predict
--final-total-recovery-num-predict
```
