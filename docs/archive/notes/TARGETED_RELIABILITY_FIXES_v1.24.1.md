# v1.24.1 targeted reliability fixes

This release intentionally leaves OCR, VLM, re-OCR, tax extraction, and repair
activation behavior unchanged.  It contains only three scoped changes:

1. A missing `overall_confidence` field is normalized locally to `0.6`, so an
   otherwise structurally valid LLM receipt does not trigger a complete second
   generation.
2. Item categorization now requires explicit evidence terms and a text-certainty
   assessment.  Truncated, unfamiliar, ambiguous, or semantically expanded
   product interpretations are confidence-capped and marked for review.  No
   product abbreviation dictionary is used.
3. Multi-source recovery merges relative item sequences and preserves printed
   receipt order instead of sorting by recovery source or product name.
