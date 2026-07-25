"""Phase 9 places extraction algorithms under responsibility-based packages."""

from __future__ import annotations

from pathlib import Path

from receipt_intelligence.extraction.categorization.items import categorize_receipt_items_llm
from receipt_intelligence.extraction.evidence.compact import build_compact_evidence
from receipt_intelligence.extraction.parsing.llm_parser import build_ocr_context
from receipt_intelligence.extraction.repair.patch_correction import run_patch_correction_pass
from receipt_intelligence.extraction.validation.receipt import validate_receipt

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "src" / "receipt_intelligence"
PIPELINE_DIR = PACKAGE_ROOT / "pipeline"
EXTRACTION_DIR = PACKAGE_ROOT / "extraction"


OBSOLETE_SPATIAL_PIPELINE_FILES = [
    EXTRACTION_DIR / "parsing" / "table_assembler.py",
    EXTRACTION_DIR / "parsing" / "table_interpreter.py",
    EXTRACTION_DIR / "repair" / "right_column.py",
    EXTRACTION_DIR / "repair" / "vertical_price_stack.py",
    PACKAGE_ROOT / "prompts" / "main_receipt_parser.txt",
    PACKAGE_ROOT / "prompts" / "spatial_overview.txt",
    PACKAGE_ROOT / "prompts" / "table_interpreter_compact.txt",
    ROOT / "scripts" / "demo_table_interpretation_prompt.py",
    ROOT / "docs" / "TABLE_INTERPRETATION.md",
]

OLD_PIPELINE_MODULES = {
    "llm_receipt_parser_main.py",
    "receipt_compact_evidence_v14.py",
    "receipt_consistency_postprocess_v14.py",
    "receipt_correction_pass_v14.py",
    "receipt_grouped_evidence_v14.py",
    "receipt_item_categorizer_v14.py",
    "receipt_layout_context_v14.py",
    "receipt_patch_correction_v14.py",
    "receipt_region_reocr_v14.py",
    "receipt_reocr_repair_v14.py",
    "receipt_right_column_recovery_v14.py",
    "receipt_table_arbitration_v14.py",
    "receipt_table_assembler_v14.py",
    "receipt_table_interpreter_v14.py",
    "receipt_validation_v14.py",
    "receipt_vertical_price_stack_recovery_v14.py",
    "receipt_visual_evidence_v14.py",
}


def test_versioned_pipeline_modules_are_removed() -> None:
    remaining = sorted(
        path.name for path in PIPELINE_DIR.glob("*.py") if path.name in OLD_PIPELINE_MODULES
    )
    assert remaining == []
    assert sorted(path.name for path in PIPELINE_DIR.glob("*.py")) == [
        "__init__.py",
        "integrated_receipt_pipeline.py",
    ]


def test_responsibility_based_extraction_packages_exist() -> None:
    expected = [
        EXTRACTION_DIR / "evidence" / "compact.py",
        EXTRACTION_DIR / "evidence" / "grouped.py",
        EXTRACTION_DIR / "evidence" / "layout.py",
        EXTRACTION_DIR / "evidence" / "visual.py",
        EXTRACTION_DIR / "parsing" / "llm_parser.py",
        EXTRACTION_DIR / "parsing" / "table_arbitration.py",
        EXTRACTION_DIR / "validation" / "receipt.py",
        EXTRACTION_DIR / "validation" / "consistency.py",
        EXTRACTION_DIR / "repair" / "item_order.py",
        EXTRACTION_DIR / "repair" / "patch_correction.py",
        EXTRACTION_DIR / "repair" / "region_reocr.py",
        EXTRACTION_DIR / "repair" / "reocr.py",
        EXTRACTION_DIR / "categorization" / "items.py",
    ]
    assert not [path for path in expected if not path.is_file()]

    # Importing representative public functions verifies that the moved modules
    # resolve their internal dependencies through the new package paths.
    assert callable(build_ocr_context)
    assert callable(build_compact_evidence)
    assert callable(validate_receipt)
    assert callable(run_patch_correction_pass)
    assert callable(categorize_receipt_items_llm)


def test_active_code_contains_no_versioned_pipeline_imports() -> None:
    forbidden = (
        "receipt_intelligence.pipeline.llm_receipt_parser_main",
        "receipt_intelligence.pipeline.receipt_",
    )
    offenders: list[str] = []

    for path in [*PACKAGE_ROOT.rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
    assert not (ROOT / "scripts" / "run_receipt_folder_v14.py").exists()
    assert not (ROOT / "scripts" / "run_receipt_images_folder_v14.py").exists()
    assert (ROOT / "scripts" / "run_receipt_folder.py").is_file()
    assert (ROOT / "scripts" / "run_receipt_images_folder.py").is_file()

def test_obsolete_spatial_reconstruction_files_are_removed() -> None:
    remaining = [path.relative_to(ROOT) for path in OBSOLETE_SPATIAL_PIPELINE_FILES if path.exists()]
    assert remaining == []
