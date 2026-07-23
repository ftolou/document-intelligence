# Regression Reporting

The app already stores each job under `var/jobs/<job_id>/`. Phase 1 adds a dependency-free report generator that converts these saved job statuses into quality metrics.

## Generate report

```powershell
python .\scripts\generate_regression_report.py --results-dir .\var\jobs --out-dir .\var\reports\regression_report
```

Outputs:

```text
regression_summary.json
regression_jobs.csv
regression_report.md
```

## Metrics

The report includes:

- total processed jobs
- job states
- import decisions
- human-reviewed count and review rate
- jobs with validation issues
- average absolute total difference
- maximum absolute total difference

## Portfolio interpretation

For an AI/KI Manager portfolio, the regression report is more important than a single successful demo. It shows that you can measure reliability, track failure patterns, and discuss rollout risk.

## Suggested acceptance gate

For a small local test set, define a release gate such as:

```text
- 90% of receipts must finish without pipeline error.
- 80% must be import or needs_review, not reject.
- 100% must produce a final JSON artifact.
- High-risk outputs must require human review.
```

These gates can later be connected to GitHub Actions or GitLab CI.
