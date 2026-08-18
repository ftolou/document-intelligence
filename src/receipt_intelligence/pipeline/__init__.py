"""Stable public extraction pipeline entry point.

Implementation components live under :mod:`receipt_intelligence.extraction`.
"""

from receipt_intelligence.pipeline.integrated_receipt_pipeline import run_receipt_extraction

__all__ = ["run_receipt_extraction"]
