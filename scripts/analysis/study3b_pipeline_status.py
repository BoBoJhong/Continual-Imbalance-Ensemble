"""Read-only progress and fit-cost inventory for the Study 3B full pipeline."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def inspect_run(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    ends = {end for year in config["test_years"]
            for end in range(year - config["pool_size"] - 1, year - 1)}
    per_seed = len(ends) * len(config["classifiers"]) * len(config["sampling"])
    summaries, costs = [], defaultdict(list)
    for seed in config["seeds"]:
        folder = run / f"seed_{seed}"
        audits = [json.loads(path.read_text(encoding="utf-8"))
                  for path in sorted((folder / "cache").glob("*.json"))]
        counts = Counter(a["fs"] == "none" for a in audits)
        for audit in audits:
            costs[(audit["classifier"], audit["sampling"], audit["fs"])].append(audit["fit_predict_seconds"])
        failures = folder / "failures.json"
        summaries.append({
            "seed": seed, "baseline_fits_completed": counts[True], "baseline_fits_expected": per_seed,
            "fs_fits_completed": counts[False], "warnings_count": sum(bool(a["warnings"]) for a in audits),
            "ensemble_years_completed": sum((folder / f"year_{y}/stage12_metrics.csv").exists()
                                            for y in config["test_years"]),
            "fs_years_completed": sum((folder / f"year_{y}/stage3_metrics.csv").exists()
                                      for y in config["test_years"]),
            "seed_completed": (folder / "complete.json").exists(),
            "failures": json.loads(failures.read_text(encoding="utf-8")) if failures.exists() else [],
        })
    progress_path = run / "progress.json"
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(), "run": str(run.resolve()),
        "manifest_status": manifest["status"], "pid": manifest.get("pid"),
        "note": "Manifest status is not an operating-system process liveness check. FS cost is not yet known.",
        "progress": json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {},
        "baseline_fits_completed": sum(s["baseline_fits_completed"] for s in summaries),
        "baseline_fits_expected": per_seed * len(config["seeds"]),
        "fs_fits_completed": sum(s["fs_fits_completed"] for s in summaries),
        "seeds": summaries,
        "observed_fit_predict_costs": [{"classifier": c, "sampling": s, "fs": f,
                                        "count": len(values), "mean_seconds": sum(values) / len(values),
                                        "max_seconds": max(values)}
                                       for (c, s, f), values in sorted(costs.items())],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_run(args.run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
