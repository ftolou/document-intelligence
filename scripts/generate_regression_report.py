#!/usr/bin/env python3
"""Generate a lightweight regression/quality report from saved receipt jobs.

This script is intentionally dependency-free so it can run locally and in CI.
It reads the Flask app's JSON-backed job store under var/jobs and turns
completed single-image jobs into a management-friendly quality report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iter_jobs(results_dir: Path) -> list[dict[str, Any]]:
    """Find saved single-image jobs robustly.

    Normal app output is `var/jobs/<job_id>/job_status.json`, but local
    ZIPs and copied artifacts are sometimes passed as `outputs/`, project root,
    or a nested extraction directory. Search direct children first, then fall
    back to recursive discovery.
    """
    jobs: list[dict[str, Any]] = []
    candidate_status_paths = sorted(results_dir.glob("*/job_status.json"))
    if not candidate_status_paths:
        nested = results_dir / "var" / "jobs"
        if nested.exists():
            candidate_status_paths = sorted(nested.glob("*/job_status.json"))
    if not candidate_status_paths:
        candidate_status_paths = sorted(results_dir.glob("**/job_status.json"))

    seen: set[str] = set()
    for status_path in candidate_status_paths:
        job = load_json(status_path)
        if not job:
            continue
        if job.get("type") == "batch":
            continue
        job_id = str(job.get("job_id") or status_path.parent.name)
        if job_id in seen:
            continue
        if (
            job.get("type") == "single_in_batch"
            or job.get("filename")
            or isinstance(job.get("result"), dict)
        ):
            job.setdefault("job_id", job_id)
            jobs.append(job)
            seen.add(job_id)
    return jobs


def report_row(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    report = result.get("report") if isinstance(result.get("report"), dict) else {}
    issues = report.get("issues") if isinstance(report.get("issues"), list) else []
    review = job.get("review") if isinstance(job.get("review"), dict) else {}
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    return {
        "job_id": job.get("job_id"),
        "filename": job.get("filename") or Path(str(job.get("image_path", ""))).name,
        "state": job.get("state"),
        "decision": report.get("import_decision") or "n/a",
        "balanced": report.get("balanced"),
        "difference": report.get("difference"),
        "issue_count": len(issues),
        "review_status": review.get("status") or "none",
        "has_human_review": bool(artifacts.get("human_review") or review),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(str(r.get("decision")) for r in rows)
    states = Counter(str(r.get("state")) for r in rows)
    review_count = sum(1 for r in rows if r.get("has_human_review"))
    numeric_diffs: list[float] = []
    for r in rows:
        try:
            if r.get("difference") is not None:
                numeric_diffs.append(abs(float(r.get("difference"))))
        except Exception:
            pass
    return {
        "total_jobs": len(rows),
        "states": dict(states),
        "decisions": dict(decisions),
        "human_reviewed": review_count,
        "review_rate": round(review_count / len(rows), 4) if rows else 0.0,
        "avg_abs_difference": round(mean(numeric_diffs), 4) if numeric_diffs else None,
        "max_abs_difference": round(max(numeric_diffs), 4) if numeric_diffs else None,
        "jobs_with_issues": sum(1 for r in rows if int(r.get("issue_count") or 0) > 0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "job_id",
        "filename",
        "state",
        "decision",
        "balanced",
        "difference",
        "issue_count",
        "review_status",
        "has_human_review",
        "created_at",
        "updated_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Receipt Regression Report",
        "",
        "This report is generated from saved `var/jobs/*/job_status.json` files.",
        "",
        "## Summary",
        "",
        f"- Total jobs: **{summary['total_jobs']}**",
        f"- Human-reviewed jobs: **{summary['human_reviewed']}** ({summary['review_rate']:.0%})",
        f"- Jobs with validation issues: **{summary['jobs_with_issues']}**",
        f"- Average absolute difference: **{summary['avg_abs_difference']}**",
        f"- Maximum absolute difference: **{summary['max_abs_difference']}**",
        "",
        "### Decisions",
        "",
    ]
    if summary["decisions"]:
        for key, value in sorted(summary["decisions"].items()):
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- No jobs found yet.")
    lines.extend(
        [
            "",
            "## Job details",
            "",
            "| Job | File | State | Decision | Balanced | Difference | Issues | Review |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| `{r.get('job_id')}` | {r.get('filename')} | {r.get('state')} | {r.get('decision')} | "
            f"{r.get('balanced')} | {r.get('difference')} | {r.get('issue_count')} | {r.get('review_status')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("var/jobs"))
    parser.add_argument("--out-dir", type=Path, default=Path("var/reports/regression_report"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = iter_jobs(args.results_dir)
    rows = [report_row(job) for job in jobs]
    summary = summarize(rows)

    (args.out_dir / "regression_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.out_dir / "regression_jobs.csv", rows)
    write_markdown(args.out_dir / "regression_report.md", summary, rows)
    print(json.dumps({"out_dir": str(args.out_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
