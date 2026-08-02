"""Default independent rule groups in legacy-compatible evaluation order."""

from receipt_intelligence.extraction.validation.rules.currency import CurrencyRule
from receipt_intelligence.extraction.validation.rules.items import ItemRules
from receipt_intelligence.extraction.validation.rules.receipt import ReceiptAmountRule
from receipt_intelligence.extraction.validation.rules.totals import PaymentRule, TotalRules
from receipt_intelligence.extraction.validation.rules.vat import VatRules

__all__ = [
    "CurrencyRule",
    "ItemRules",
    "PaymentRule",
    "ReceiptAmountRule",
    "TotalRules",
    "VatRules",
]
