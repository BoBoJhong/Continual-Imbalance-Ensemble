# Result artifacts

`results/` is intentionally version-controlled. It contains raw per-run outputs,
derived summaries, statistical tables, and report figures needed to audit the
study. Do not hand-edit generated artifacts; rerun the owning experiment or
analysis script.

## Current result families

- `phase1_baseline/`: model-family baselines and yearly outputs.
- `phase2_ensemble/`: static ensembles plus DES/DCS outputs.
- `phase3_feature/`: feature-selection studies and stability diagnostics.
- `phase4_drift/`: drift signals, ROSS variants, and boundary validation.
- `phase5_weighted/`: weighting sweeps and AWE comparison.
- `phase_flexible/`: rolling out-of-time adaptive evaluation.
- `multi_seed/`: per-seed raw rows and aggregated diagnostics.
- `statistical_tests/`: paired tests and sensitivity analyses.
- `thesis_tables/`: manuscript-ready derived tables.
- `professor_report/`: outputs supporting the progress report.

Older top-level folders such as `baseline/`, `ensemble/`, `des/`, `stock/`, and
`medical/` are retained as legacy experiment evidence. They are not the primary
source for current confirmatory claims.

## Validation and provenance

Validate every CSV:

```powershell
python scripts/analysis/validate_result_artifacts.py results --strict
```

Regenerate checksums, runtime versions, Git state, raw-data hashes, and CSV
profiles:

```powershell
python scripts/analysis/generate_result_manifest.py
```

The generated `RESULT_MANIFEST.json` is itself tracked. Its `git.dirty` value
records the state at generation time; generate it again after the final commit
when an exact clean-commit manifest is required.
