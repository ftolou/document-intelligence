"""Stable public extraction pipeline entry points.

Implementation components live under :mod:`receipt_intelligence.extraction`.
"""

from receipt_intelligence.pipeline.integrated_receipt_pipeline import (
    run_integrated_receipt_pipeline,
)

__all__ = ["run_integrated_receipt_pipeline"]
