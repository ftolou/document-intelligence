# Issue #38 owner-authored behavior contract

This directory contains the owner-defined executable contracts for Core issue #38.

## Initial Orchestrator gate

The pre-implementation behavior gate intentionally uses a non-discoverable filename so ordinary repository pytest discovery remains green on the baseline target:

```bash
python scripts/run_tests.py tests/unit/interpretation/behavior_interpretation_outcome_contract.py
```

The file contains 6 test functions / 7 test IDs covering the representative behavior contract. On the pre-implementation target they must collect and execute, then fail as behavioral assertions rather than collection/setup/infrastructure errors.

`interpretation_outcome_support.py` is part of the owner-authored contract semantics and must be integrity-protected together with the gate file.

## Full issue validation

The remaining owner-authored acceptance cases are retained in:

```text
authoritative_interpretation_validation_rules.py
```

They are deliberately non-discoverable before implementation so baseline `pytest` is not made red merely by staging an upcoming issue contract. They should be run explicitly during implementation and become part of ordinary authoritative regression coverage when issue #38 is completed without weakening their assertions.

`test_interpretation_boundaries.py` is discoverable because those architecture/regression guards are expected to be green both before and after implementation.
