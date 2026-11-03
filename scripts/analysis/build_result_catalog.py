"""Build a versioned file inventory without changing research artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = "scripts/analysis/build_result_catalog.py"
POLICY = "config/result_catalog_config.json"
EXCLUDED = "results/catalog/"
REMOVAL_LEDGER = "docs/REMOVED_STOCK_20261012.json"
REMOVAL_LEDGERS = (REMOVAL_LEDGER, "docs/REMOVED_RESULTS_20261012.json")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def matches(path: str, condition: dict) -> bool:
    prefixes = condition.get("prefixes", []) + ([condition["prefix"]] if "prefix" in condition else [])
    paths = condition.get("paths", [])
    if (prefixes or paths) and not (path in paths or any(path.startswith(p) for p in prefixes)):
        return False
    name = Path(path).name
    if "contains_any" in condition and not any(token in path for token in condition["contains_any"]):
        return False
    if "name_contains" in condition and condition["name_contains"] not in name:
        return False
    return not ("name_regex" in condition and not re.search(condition["name_regex"], name))


def validate_policy(policy: dict) -> None:
    if policy["schema_version"] != 1:
        raise ValueError("Unsupported catalog policy version")
    categories = policy["categories"]
    ids = [rule["id"] for rule in policy["rules"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate classification rule identifiers")
    for rule in policy["rules"]:
        if rule["category"] not in categories or rule.get("tier") not in {"override", "fallback"}:
            raise ValueError("Invalid classification rule")
        condition = rule["match"]
        if not condition:
            raise ValueError("Unbounded classification rule")
        if "name_regex" in condition:
            re.compile(condition["name_regex"])
    for run in policy["selected_runs"]:
        path = Path(run["path"])
        if path.is_absolute() or ".." in path.parts or not run["path"].startswith("results/"):
            raise ValueError("Selected run must stay in results")
        if run["category"] not in categories:
            raise ValueError("Unknown selected-run category")


def context_for(path: str, contexts: dict) -> str | None:
    for parent in Path(path).parents:
        candidate = parent.as_posix() + "/manifest.json"
        if candidate in contexts:
            return candidate
    return None


def classify(path: str, policy: dict, contexts: dict) -> dict:
    context = context_for(path, contexts)
    state = contexts.get(context, {}).get("status")
    if state == "failed":
        return {"category": "failed_run", "rule_id": "manifest-failed",
                "reason": "Nearest run manifest records failed", "tags": [], "context": context}
    for tier in ("override", "fallback"):
        for rule in policy["rules"]:
            if rule["tier"] == tier and matches(path, rule["match"]):
                return {"category": rule["category"], "rule_id": rule["id"],
                        "reason": rule["reason"], "tags": list(rule.get("tags", [])), "context": context}
        if tier == "override":
            for run in policy["selected_runs"]:
                if path == run["path"] + "/manifest.json" or path.startswith(run["path"] + "/"):
                    completed = contexts.get(run["path"] + "/manifest.json", {}).get("status") == "completed"
                    return {"category": run["category"] if completed else "needs_review",
                            "rule_id": "selected-run-" + run["id"],
                            "reason": "Selected run manifest completed" if completed else "Selected run lacks completed evidence",
                            "tags": [] if completed else ["completion_unverified"], "context": context}
    return {"category": "needs_review", "rule_id": "unmatched",
            "reason": "No sufficiently specific classification rule", "tags": [], "context": context}


def build_catalog(root: Path, policy: dict, *, generated_at: str | None = None) -> dict:
    root = root.resolve()
    validate_policy(policy)
    paths = []
    skipped = []
    for path in sorted((root / "results").rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(EXCLUDED):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root / "results"):
            skipped.append(relative)
            continue
        if path.is_file():
            paths.append(path)
    contexts = {}
    for path in paths:
        if path.name != "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        try:
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
            contexts[relative] = {"status": manifest.get("status"), "protocol": manifest.get("protocol"),
                                  "config": manifest.get("config"), "target": manifest.get("target"),
                                  "limitations": manifest.get("limitations"), "sha256": digest(path)}
        except (ValueError, OSError) as exc:
            contexts[relative] = {"status": "unreadable", "error": str(exc)}
    files = []
    candidates = defaultdict(list)
    for path in paths:
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        record = {"path": relative, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                  **classify(relative, policy, contexts)}
        files.append(record)
        if stat.st_size:
            candidates[stat.st_size].append(record)
    duplicates = []
    for size, group in sorted(candidates.items()):
        if len(group) < 2:
            continue
        by_hash = defaultdict(list)
        for record in group:
            record["sha256"] = digest(root / record["path"])
            by_hash[record["sha256"]].append(record)
        for checksum, copies in sorted(by_hash.items()):
            if len(copies) < 2:
                continue
            paths_in_group = sorted(record["path"] for record in copies)
            normalized_seeds = {re.sub(r"/seed_\d+/", "/seed_*/", p) for p in paths_in_group}
            normalized_analysis = {re.sub(r"/analysis/[^/]+/", "/analysis/*/", p) for p in paths_in_group}
            if len(normalized_seeds) == 1:
                kind = "repeated_seed_outputs"
            elif len(normalized_analysis) == 1:
                kind = "analysis_snapshot_copies"
            else:
                kind = "cross_path_copies_review_required"
            duplicates.append({"sha256": checksum, "bytes_per_file": size, "paths": paths_in_group,
                               "kind": kind, "action": "retain; check provenance and citations before any consolidation"})
            for record in copies:
                record["tags"].append("exact_duplicate_bytes")
                record["duplicate_sha256"] = checksum
    counts = Counter(record["category"] for record in files)
    totals = {key: {"files": counts[key], "bytes": sum(r["bytes"] for r in files if r["category"] == key)}
              for key in policy["categories"]}
    return {"schema_version": 1, "generated_at_utc": generated_at or datetime.now(timezone.utc).isoformat(),
            "generator": GENERATOR, "policy": POLICY, "policy_content_sha256": hashlib.sha256(
                json.dumps(policy, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "scope": policy["scope"], "excluded_prefixes": [EXCLUDED], "skipped_paths": skipped,
            "limitations": ["Metadata classification is not a numerical validity audit.",
                            "mtime is recorded but never used to decide a file is outdated.",
                            "Exact duplicates do not imply files can be deleted; seed repeats and citations matter.",
                            "Manifest state is recorded evidence, not a process-liveness check."],
            "categories": policy["categories"], "selected_runs": policy["selected_runs"],
            "run_contexts": contexts, "counts": totals, "file_count": len(files),
            "total_bytes": sum(r["bytes"] for r in files), "duplicate_groups": duplicates, "files": files}


def link(target: str, base: Path, root: Path, label: str | None = None) -> str:
    import os
    relative = Path(os.path.relpath(root / target, base)).as_posix()
    text = (label or target).replace("|", r"\|").replace("[", r"\[").replace("]", r"\]")
    if not (root / target).exists():
        for name in REMOVAL_LEDGERS:
            ledger_path = root / name
            if not ledger_path.is_file():
                continue
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            removed = {entry["path"]: entry for entry in ledger["files"]}
            entry = removed.get(target.rstrip("/"))
            if entry:
                replacement = entry.get("replacement_path")
                if replacement and (root / replacement).is_file():
                    return link(replacement, base, root, label or target) + "（來源已整併）"
            if entry or any(target.rstrip("/") == folder or target.startswith(folder + "/")
                            for folder in ledger.get("removed_directories", [])):
                return text + "（已移除；" + link(name, base, root, "移除清冊") + "）"
    return "[" + text + "](" + quote(relative, safe="/.") + ")"


def write_catalog(catalog: dict, root: Path, output: Path, *, refresh_links: bool = False) -> None:
    root, output = root.resolve(), output.resolve()
    allowed = root / "results/catalog"
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Catalog snapshot output must be below results/catalog")
    if refresh_links:
        recorded = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
        if recorded != catalog:
            raise ValueError("Historical link refresh must preserve recorded metadata")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for category, meta in catalog["categories"].items():
        count = catalog["counts"][category]
        rows.append(f"| [{meta['label']}]({category}.md) | {count['files']} | {count['bytes'] / 2**20:.2f} | {meta['note']} |")
        records = [r for r in catalog["files"] if r["category"] == category]
        parts = ["# " + meta["label"], "", meta["note"], "", "[回分類首頁](INDEX.md)；[逐檔 JSON](catalog.json)", ""]
        # Roll up folders to keep the completed runs readable; the JSON remains exhaustive.
        by_folder = defaultdict(list)
        for record in records:
            by_folder[str(Path(record["path"]).parent).replace("\\", "/")].append(record)
        parts.extend(["| 目錄 | 檔案數 |", "|---|---|"])
        for folder, entries in sorted(by_folder.items()):
            parts.append("| " + link(folder, output, root) + f" | {len(entries)} |")
        if category not in {"current_core", "current_extension"}:
            parts.extend(["", "## 逐檔清單", "", "| 檔案 | bytes | 規則 |", "|---|---|---|"])
            for record in records:
                parts.append("| " + link(record["path"], output, root) + f" | {record['bytes']} | {record['rule_id']} |")
        (output / (category + ".md")).write_text("\n".join(parts) + "\n", encoding="utf-8")
    duplicate_lines = ["# 位元組相同的檔案", "", "以下只證明內容相同；保留 seed、快照、引用與來源身份，不自動刪除。", ""]
    for group in catalog["duplicate_groups"]:
        duplicate_lines.extend(["## " + group["sha256"][:16], "", group["kind"] + f"；每份 {group['bytes_per_file']} bytes", ""])
        duplicate_lines.extend("- " + link(path, output, root) for path in group["paths"])
        duplicate_lines.append("")
    (output / "duplicates.md").write_text("\n".join(duplicate_lines), encoding="utf-8")
    main = ["# 結果分類入口", "", "產生時間（UTC）：" + catalog["generated_at_utc"], "",
            f"共 {catalog['file_count']} 份原有檔案；輸出清冊目錄不納入自身盤點。分類依證據與範圍，不按修改時間判定過時。", "",
            "[所有逐檔紀錄](catalog.json)；[內容相同檔案](duplicates.md)", "", "## 已選定的完成 run", "",
            "| 研究 | run | target | manifest 狀態 |", "|---|---|---|---|"]
    for run in catalog["selected_runs"]:
        state = catalog["run_contexts"].get(run["path"] + "/manifest.json", {}).get("status", "未確認")
        main.append(f"| {run['title']} | " + link(run["path"], output, root) + f" | {run['target']} | {state} |")
    main.extend(["", "## 檔案分類", "", "| 類別 | 檔案數 | MiB | 使用限制 |", "|---|---|---|---|", *rows, "",
                 "## 解讀與重建", "", "分類不是全部數值重新驗證；不同 target、協定、部分 seeds 與 pooled／年度口徑仍須分開。", "",
                 "政策：" + link(POLICY, output, root) + "；產生工具：" + link(GENERATOR, output, root), "",
                 "由專案根目錄執行 python scripts/analysis/build_result_catalog.py；每次建立新時間戳快照，更新 catalog/README 導航。", ""])
    if refresh_links:
        main[4:4] = ["歷史快照：原始 catalog.json metadata 保留；已移除檔案的連結改指向移除清冊。", ""]
    (output / "INDEX.md").write_text("\n".join(main), encoding="utf-8")
    if refresh_links:
        return
    relative = output.relative_to(allowed).as_posix()
    (allowed / "README.md").write_text("# 結果清冊\n\n最新產生快照：[分類入口](" + relative +
        "/INDEX.md)；[所有逐檔紀錄](" + relative + "/catalog.json)；[內容相同檔案](" + relative +
        "/duplicates.md)。\n\n舊快照保留於各時間戳資料夾。這是目錄與來源整理，不代表新的實驗成效。\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / POLICY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refresh-links", type=Path, help="只刷新既存快照的 Markdown 連結，保留 catalog.json 與最新導航")
    args = parser.parse_args()
    if args.refresh_links:
        if args.output:
            parser.error("--refresh-links cannot be combined with --output")
        snapshot = args.refresh_links.resolve()
        if not snapshot.is_relative_to(ROOT / "results/catalog") or snapshot == ROOT / "results/catalog":
            parser.error("--refresh-links must name a snapshot below results/catalog")
        recorded = json.loads((snapshot / "catalog.json").read_text(encoding="utf-8"))
        write_catalog(recorded, ROOT, snapshot, refresh_links=True)
        print(json.dumps({"refreshed_links": str(snapshot), "metadata_preserved": True}))
        return 0
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    output = args.output or ROOT / "results/catalog" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    # Check output before scanning any large input.
    if not output.resolve().is_relative_to(ROOT / "results/catalog") or output.resolve() == ROOT / "results/catalog":
        parser.error("--output must be a snapshot below results/catalog")
    catalog = build_catalog(ROOT, policy)
    catalog["policy"] = args.policy.resolve().relative_to(ROOT).as_posix()
    catalog["generator_sha256"] = digest(Path(__file__))
    catalog["policy_sha256"] = digest(args.policy)
    write_catalog(catalog, ROOT, output)
    print(json.dumps({"output": str(output), "files": catalog["file_count"], "categories": catalog["counts"],
                      "duplicate_groups": len(catalog["duplicate_groups"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
