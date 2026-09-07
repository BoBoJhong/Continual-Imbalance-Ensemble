# Project Context and Validity Guardrails

## Research identity

This repository studies continual ensemble learning for class-imbalanced, non-stationary data. Its main concerns include temporal distribution shift, old/new model pools, sampling strategies, dynamic ensemble selection, feature selection, drift response, and weighted ensembles.

Treat **DAWCE** and **ROSS** as project-defined methods unless a project document explicitly attributes the name to prior literature. Present established ingredients separately from the novel combination.

## Read first by task

| Task | Primary project files |
|---|---|
| Research scope | `docs/研究方向.md`, `README.md` |
| Literature positioning | core-literature section in `docs/研究方向.md` |
| Dataset provenance and schema | dataset section in `docs/研究方向.md`, `src/data/` |
| Current conclusions | `docs/研究方向.md` |
| DAWCE/ROSS description | Study 3 section in `docs/研究方向.md` |
| Manuscript integration | `thesis/THESIS_FULL.md` |
| Numerical claims | relevant CSV/JSON under `results/` plus its generating script |

Resolve conflicts in favor of reproducible code and current result artifacts, then record documentation drift rather than silently choosing a convenient version.

## Method validity checklist

- State the prediction target, positive class, observation unit, time span, and imbalance ratio.
- Describe chronological train/validation/test boundaries and why they match the deployment story.
- Confirm that all learned preprocessing and selection steps were fit without future information.
- Name baselines fairly and use the same data availability and tuning budget where possible.
- Separate validation-selected configurations from test-set evaluation.
- Record seeds, package versions, configuration files, and the command that produced each result.
- Report per-period behavior when aggregate metrics can hide drift or minority failure.
- For paired tests, explain the pairing unit and dependence assumptions.
- Include ablations that isolate weighting, drift signals, feature selection, and pool composition when claiming their contribution.

## Interpretation boundaries

- A high ROC-AUC does not alone establish useful minority detection.
- A better mean without uncertainty or paired evidence is descriptive, not conclusive.
- A best-on-test configuration is an oracle analysis, not a valid selected model.
- Results on one bankruptcy dataset do not establish cross-domain generalization.
- Synthetic data can test mechanics but cannot by itself validate real-world effectiveness.
