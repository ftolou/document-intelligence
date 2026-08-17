"""Stable public extraction pipeline entry points.

Implementation components live under :mod:`receipt_intelligence.extraction`.
"""

from receipt_intelligence.pipeline.integrated_receipt_pipeline import (
    run_integrated_receipt_pipeline,
    run_receipt_extraction,
)

__all__ = ["run_receipt_extraction", "run_integrated_receipt_pipeline"]
