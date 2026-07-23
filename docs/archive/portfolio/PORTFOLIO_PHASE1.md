# Portfolio Phase 1

## Positioning

This project should be presented as:

> A local-first AI document automation prototype that extracts structured receipt data with OCR/LLMs, validates the result deterministically, supports human approval, and produces measurable regression reports.

Do not present it only as a receipt parser. The stronger AI-manager story is that it demonstrates an end-to-end operating model for controlled AI automation.

## Added in Phase 1

| Area | Added capability | Why it matters for AI/KI Manager roles |
|---|---|---|
| README | Portfolio-oriented quick start and positioning | Recruiters understand the business case quickly. |
| Architecture | Mermaid data-flow diagram | Shows you can communicate technical systems clearly. |
| Human review | Editable key fields and approval artifact | Proves controlled automation and traceability. |
| Regression reporting | Script for summary JSON/CSV/Markdown | Proves quality management instead of AI demoing. |
| Screenshot guide | List of views to capture | Helps create a convincing GitHub/LinkedIn/application package. |

## Demo storyline

Use this 3-5 minute demo structure:

1. Start Docker Compose and show the UI.
2. Upload one receipt.
3. Show live timeline, artifacts, extracted JSON, and validation result.
4. Correct one field in the human-review panel.
5. Save the review and show `approved_receipt.json` plus `human_review_record.json`.
6. Run a small batch from `batch_input`.
7. Generate the regression report and show summary metrics.

## What this proves

- You can build a usable AI workflow.
- You understand validation and failure analysis.
- You know that AI outputs require review and auditability.
- You can package a prototype for business/IT stakeholders.
- You can connect AI automation to CI/CD and regression testing.

## Next portfolio upgrade

Phase 2 should add:

- AI use-case dashboard
- governance dashboard
- prompt/version registry
- model/data decision matrix
- simple ROI estimator
