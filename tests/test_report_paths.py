"""Versioned output defaults and links from nested report folders."""
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

from scripts.analysis import audit_phase123_split_contracts as audit
from scripts.reports import build_study3b_advisor_report as report


def test_default_reports_create_distinct_versions_in_month_folder():
    first = datetime(2026, 9, 29, 1, 2, 3, tzinfo=timezone.utc)
    a = report.default_report_output(first)
    b = report.default_report_output(first + timedelta(microseconds=1))
    assert a.parent == report.ROOT / "docs/reports/2026-09"
    assert a != b
    assert a.name != "滑動視窗三階段實驗成果報告_20260929.md"


def test_links_resolve_from_nested_report_and_root_docs():
    target = report.ROOT / "results/測試 資料/summary.csv"
    for output in (report.ROOT / "docs/example.md", report.ROOT / "docs/reports/2026-10/example.md"):
        resolved = (output.parent / unquote(report.report_link(target, output))).resolve()
        assert resolved == target.resolve()


def test_default_audits_do_not_overwrite_the_dated_historical_json():
    at = datetime(2026, 10, 6, tzinfo=timezone.utc)
    a = audit.default_audit_output(at)
    b = audit.default_audit_output(at + timedelta(microseconds=1))
    assert a.parent == audit.ROOT / "docs/audits/2026-10"
    assert a != b
    assert a.name != "phase123_split_audit_20261006.json"
