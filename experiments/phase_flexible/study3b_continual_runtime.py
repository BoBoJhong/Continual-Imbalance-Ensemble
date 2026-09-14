"""Persistent Study 3B FIFO runtime for annual continual updates.

This module separates four operations:

``prepare-labels``
    Build the versioned retrospective benchmark label ledger.  The raw CSV is
    never modified.
``initialize``
    Create the first three-model FIFO pool for a prediction feature year.
``update``
    Advance exactly one feature year, retain two models, train one model, and
    atomically replace the pool registry.
``predict``
    Score the current feature-year batch without loading or exporting its
    future event labels.

The bundled label ledger uses the audited annual reconstruction and therefore
supports research replay only.  A real deployment must replace it with an
event ledger carrying authoritative availability timestamps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# joblib otherwise probes the removed Windows `wmic` command on some Python
# versions. This must be set before importing sklearn through project modules.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import xgboost as xgb

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments._shared.common_bankruptcy import US_CSV
from experiments.phase_flexible.rolling_bankruptcy_overlap_ensemble import (
    PROTOCOL_VERSION,
    VALIDATION_FPR_BUDGET,
    FittedWindowModel,
    _fit_window_model,
    _select_f1_threshold,
    _select_threshold_at_fpr,
    build_event_target,
    model_window_ends,
    window_years,
)
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.utils import get_logger, set_seed


STATE_SCHEMA_VERSION = 1
RUNTIME_VERSION = "study3b_fifo_runtime_v1"
FEATURE_COLUMNS = tuple(f"X{index}" for index in range(1, 19))
DEFAULT_LABELS = project_root / "data" / "processed" / "bankruptcy_event_labels_v1.csv"
DEFAULT_STATE_DIR = project_root / "checkpoints" / "study3b_fifo"
DEFAULT_PREDICTION_ROOT = (
    project_root / "results" / "phase_flexible" / "study3b_live_predictions"
)
LABEL_SOURCE = "reconstructed_failed_company_final_observation"
LABEL_EVIDENCE = "annual_aggregate_verified_exact_availability_unverified"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_hash(frame: pd.DataFrame, sort_columns: list[str]) -> str:
    canonical = frame.sort_values(sort_columns).reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raw_snapshot_hash(raw_path: Path, through_feature_year: int) -> str:
    raw = pd.read_csv(raw_path)
    raw["fyear"] = raw["fyear"].astype(int)
    columns = ["company_name", "fyear", *FEATURE_COLUMNS]
    if set(columns) - set(raw.columns):
        raise ValueError("Raw feature snapshot has an incompatible schema.")
    prefix = raw.loc[raw.fyear <= through_feature_year, columns]
    return _frame_hash(prefix, ["company_name", "fyear"])


def _label_snapshot_hash(labels_path: Path, through_available_year: int) -> str:
    labels = pd.read_csv(labels_path)
    labels["label_available_year"] = labels["label_available_year"].astype(int)
    columns = ["company_name", "fyear", "event_year", "target", "label_available_year"]
    if set(columns) - set(labels.columns):
        raise ValueError("Label snapshot has an incompatible schema.")
    prefix = labels.loc[labels.label_available_year <= through_available_year, columns]
    return _frame_hash(prefix, ["company_name", "fyear"])


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


@contextmanager
def _state_lock(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / ".update.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"State is locked by another update or a previous interrupted run: {lock_path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": _now()}))
        yield
    finally:
        if lock_path.exists():
            lock_path.unlink()


def label_manifest_path(labels_path: Path) -> Path:
    return labels_path.with_suffix(".manifest.json")


def prepare_label_ledger(raw_path: Path, output_path: Path) -> dict:
    """Create an explicit, versioned annual label ledger for benchmark replay."""
    raw = pd.read_csv(raw_path)
    derived = build_event_target(raw)
    ledger = derived[["company_name", "fyear", "event_year", "target"]].copy()
    ledger["label_available_year"] = ledger["event_year"]
    ledger["label_source"] = LABEL_SOURCE
    ledger["evidence_status"] = LABEL_EVIDENCE
    if ledger.duplicated(["company_name", "fyear"]).any():
        raise ValueError("Label ledger keys are not unique.")
    manifest_path = label_manifest_path(output_path)
    if output_path.exists() or manifest_path.exists():
        if not output_path.is_file() or not manifest_path.is_file():
            raise FileExistsError("Existing label artifact is incomplete; use a new versioned path.")
        existing = pd.read_csv(output_path)
        try:
            pd.testing.assert_frame_equal(existing, ledger, check_dtype=False)
        except AssertionError as error:
            raise FileExistsError(
                "Refusing to overwrite a different label ledger; choose a new versioned output path."
            ) from error
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing_manifest.get("labels_sha256") != _sha256(output_path)
            or existing_manifest.get("raw_sha256") != _sha256(raw_path)
        ):
            raise ValueError("Existing label ledger provenance is invalid.")
        return existing_manifest
    _atomic_csv(output_path, ledger)
    manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "raw_path": str(raw_path.resolve()),
        "raw_sha256": _sha256(raw_path),
        "labels_path": str(output_path.resolve()),
        "labels_sha256": _sha256(output_path),
        "n_rows": len(ledger),
        "n_positive": int(ledger.target.sum()),
        "feature_year_min": int(ledger.fyear.min()),
        "feature_year_max": int(ledger.fyear.max()),
        "target_definition": "failed company and final observed feature year",
        "label_available_year_rule": "fyear + 1 (annual reconstruction)",
        "label_source": LABEL_SOURCE,
        "evidence_status": LABEL_EVIDENCE,
        "deployment_warning": (
            "This ledger contains retrospectively reconstructed labels for the full benchmark. "
            "The runtime gate prevents future rows from entering an update, but exact filing and "
            "financial-report availability dates are not present in the source data."
        ),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _load_label_ledger(labels_path: Path, raw_path: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = label_manifest_path(labels_path)
    if not labels_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing label ledger or manifest. Run prepare-labels first: {labels_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("labels_sha256") != _sha256(labels_path):
        raise ValueError("Label ledger hash does not match its manifest.")
    if manifest.get("raw_sha256") != _sha256(raw_path):
        raise ValueError("Raw data hash does not match the label-ledger provenance.")
    if manifest.get("label_source") != LABEL_SOURCE:
        raise ValueError("Unsupported label source for this runtime.")
    labels = pd.read_csv(labels_path)
    required = {
        "company_name", "fyear", "event_year", "target", "label_available_year",
        "label_source", "evidence_status",
    }
    if required - set(labels.columns):
        raise ValueError(f"Label ledger is missing columns: {sorted(required - set(labels.columns))}")
    if labels.duplicated(["company_name", "fyear"]).any():
        raise ValueError("Label ledger contains duplicate company-year keys.")
    labels["fyear"] = labels["fyear"].astype(int)
    labels["event_year"] = labels["event_year"].astype(int)
    labels["label_available_year"] = labels["label_available_year"].astype(int)
    labels["target"] = labels["target"].astype(int)
    if not set(labels.target.unique()) <= {0, 1}:
        raise ValueError("Label ledger target must be binary 0/1.")
    return labels, manifest


def _load_training_data(
    raw_path: Path, labels_path: Path, as_of_feature_year: int
) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_path)
    raw["fyear"] = raw["fyear"].astype(int)
    missing_features = set(FEATURE_COLUMNS) - set(raw.columns)
    if missing_features:
        raise ValueError(f"Raw data is missing features: {sorted(missing_features)}")
    labels, manifest = _load_label_ledger(labels_path, raw_path)
    allowed = labels[labels.label_available_year <= as_of_feature_year].copy()
    data = raw[["company_name", "fyear", *FEATURE_COLUMNS]].merge(
        allowed[["company_name", "fyear", "event_year", "target", "label_available_year"]],
        on=["company_name", "fyear"], how="inner", validate="one_to_one",
    )
    required_years = set(range(as_of_feature_year - 6, as_of_feature_year))
    observed = set(data.fyear.unique())
    if not required_years <= observed:
        raise ValueError(
            f"Mature labels are incomplete as of {as_of_feature_year}; "
            f"missing feature years {sorted(required_years - observed)}"
        )
    if (data.label_available_year > as_of_feature_year).any():
        raise AssertionError("Future labels passed the maturity gate.")
    return data, manifest


def _model_artifact_payload(fitted: FittedWindowModel, feature_columns: list[str]) -> dict:
    return {
        "runtime_version": RUNTIME_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_id": fitted.model_id,
        "train_years": list(fitted.train_years),
        "feature_columns": feature_columns,
        "imputer_statistics": fitted.imputer.statistics_.tolist(),
        "scaler_mean": fitted.scaler.mean_.tolist(),
        "scaler_scale": fitted.scaler.scale_.tolist(),
        "audit": fitted.audit,
        "created_at": _now(),
    }


def _persist_model(
    fitted: FittedWindowModel, state_dir: Path, feature_columns: list[str]
) -> dict:
    artifact_id = f"{fitted.model_id}_{uuid4().hex[:12]}"
    artifact_dir = state_dir / "models" / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    model_path = artifact_dir / "model.json"
    preprocessor_path = artifact_dir / "preprocessor.json"
    fitted.model.model.save_model(model_path)
    _atomic_json(preprocessor_path, _model_artifact_payload(fitted, feature_columns))
    return {
        "model_id": fitted.model_id,
        "end_year": fitted.train_years[-1],
        "train_years": list(fitted.train_years),
        "artifact_dir": artifact_dir.relative_to(state_dir).as_posix(),
        "model_sha256": _sha256(model_path),
        "preprocessor_sha256": _sha256(preprocessor_path),
        "created_at": _now(),
    }


class PersistedWindowModel:
    def __init__(self, record: dict, metadata: dict, model: xgb.XGBClassifier):
        self.record = record
        self.metadata = metadata
        self.model = model

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        columns = self.metadata["feature_columns"]
        values = frame[columns].to_numpy(dtype=float, copy=True)
        statistics = np.asarray(self.metadata["imputer_statistics"], dtype=float)
        missing_rows, missing_columns = np.where(np.isnan(values))
        values[missing_rows, missing_columns] = statistics[missing_columns]
        mean = np.asarray(self.metadata["scaler_mean"], dtype=float)
        scale = np.asarray(self.metadata["scaler_scale"], dtype=float)
        transformed = (values - mean) / scale
        return self.model.predict_proba(pd.DataFrame(transformed, columns=columns))[:, 1]


def _resolve_artifact_dir(state_dir: Path, relative_path: str) -> Path:
    root = state_dir.resolve()
    resolved = (state_dir / relative_path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Model artifact path escapes the state directory.")
    return resolved


def _load_model(record: dict, state_dir: Path) -> PersistedWindowModel:
    artifact_dir = _resolve_artifact_dir(state_dir, record["artifact_dir"])
    model_path = artifact_dir / "model.json"
    preprocessor_path = artifact_dir / "preprocessor.json"
    if _sha256(model_path) != record["model_sha256"]:
        raise ValueError(f"Model hash mismatch: {record['model_id']}")
    if _sha256(preprocessor_path) != record["preprocessor_sha256"]:
        raise ValueError(f"Preprocessor hash mismatch: {record['model_id']}")
    metadata = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    if metadata["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError(f"Feature schema mismatch: {record['model_id']}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return PersistedWindowModel(record, metadata, model)


def _registry_path(state_dir: Path) -> Path:
    return state_dir / "registry.json"


def _load_registry(state_dir: Path) -> dict:
    path = _registry_path(state_dir)
    if not path.is_file():
        raise FileNotFoundError(f"State is not initialized: {path}")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("Unsupported state schema version.")
    if registry.get("runtime_version") != RUNTIME_VERSION:
        raise ValueError("Runtime version does not match the checkpoint.")
    if registry.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Checkpoint feature schema does not match the runtime.")
    if len(registry.get("active_models", [])) != 3:
        raise ValueError("Checkpoint must contain exactly three active models.")
    return registry


def _write_registry(state_dir: Path, registry: dict) -> None:
    history_path = (
        state_dir / "history"
        / f"registry_feature_{registry['current_feature_year']}_{_stamp()}.json"
    )
    _atomic_json(history_path, registry)
    _atomic_json(_registry_path(state_dir), registry)


def _validation_thresholds(
    models: list[FittedWindowModel | PersistedWindowModel],
    validation: pd.DataFrame,
) -> tuple[dict, dict]:
    probabilities = []
    for model in models:
        if isinstance(model, FittedWindowModel):
            probabilities.append(model.predict_proba(validation, list(FEATURE_COLUMNS)))
        else:
            probabilities.append(model.predict_proba(validation))
    proba = np.mean(probabilities, axis=0)
    y_true = validation.target.to_numpy()
    f1_threshold = _select_f1_threshold(y_true, proba)
    fpr5_threshold = _select_threshold_at_fpr(y_true, proba, VALIDATION_FPR_BUDGET)
    metrics = compute_metrics(y_true, proba, threshold=f1_threshold)
    return (
        {"f1": f1_threshold, "fpr5": fpr5_threshold},
        {
            "feature_year": int(validation.fyear.iloc[0]),
            "event_year": int(validation.event_year.iloc[0]),
            "n_rows": len(validation),
            "n_positive": int(y_true.sum()),
            **metrics,
        },
    )


def initialize_state(
    state_dir: Path,
    raw_path: Path,
    labels_path: Path,
    as_of_feature_year: int,
    seed: int,
    logger,
) -> dict:
    with _state_lock(state_dir):
        if _registry_path(state_dir).exists():
            raise FileExistsError(
                f"State already exists; use update instead: {_registry_path(state_dir)}"
            )
        set_seed(seed)
        data, label_manifest = _load_training_data(raw_path, labels_path, as_of_feature_year)
        records = []
        fitted_models = []
        for end_year in model_window_ends(as_of_feature_year):
            fitted = _fit_window_model(
                data, list(FEATURE_COLUMNS), window_years(end_year),
                f"seed{seed}_M{end_year}", logger, seed=seed,
            )
            fitted_models.append(fitted)
            records.append(_persist_model(fitted, state_dir, list(FEATURE_COLUMNS)))
        validation = data[data.fyear == as_of_feature_year - 1].copy()
        thresholds, validation_audit = _validation_thresholds(fitted_models, validation)
        created_at = _now()
        registry = {
            "schema_version": STATE_SCHEMA_VERSION,
            "runtime_version": RUNTIME_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "research_status": "retrospective_annual_replay_not_validated_live_deployment",
            "current_feature_year": as_of_feature_year,
            "seed": seed,
            "feature_columns": list(FEATURE_COLUMNS),
            "active_models": records,
            "retired_models": [],
            "thresholds": thresholds,
            "validation_audit": validation_audit,
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": _sha256(raw_path),
            "raw_snapshot_through_feature_year": as_of_feature_year,
            "raw_snapshot_sha256": _raw_snapshot_hash(raw_path, as_of_feature_year),
            "labels_path": str(labels_path.resolve()),
            "labels_sha256": label_manifest["labels_sha256"],
            "labels_snapshot_through_available_year": as_of_feature_year,
            "labels_snapshot_sha256": _label_snapshot_hash(labels_path, as_of_feature_year),
            "label_source": label_manifest["label_source"],
            "label_evidence_status": label_manifest["evidence_status"],
            "runtime_source_sha256": _sha256(Path(__file__)),
            "created_at": created_at,
            "updated_at": created_at,
            "updates": [{
                "action": "initialize",
                "feature_year": as_of_feature_year,
                "added_end_years": list(model_window_ends(as_of_feature_year)),
                "retained_end_years": [],
                "removed_end_years": [],
                "completed_at": created_at,
            }],
        }
        _write_registry(state_dir, registry)
        return registry


def update_state(
    state_dir: Path,
    raw_path: Path,
    labels_path: Path,
    as_of_feature_year: int,
    logger,
) -> dict:
    with _state_lock(state_dir):
        registry = _load_registry(state_dir)
        expected = int(registry["current_feature_year"]) + 1
        if as_of_feature_year != expected:
            raise ValueError(
                f"Updates must advance exactly one year: expected {expected}, got {as_of_feature_year}"
            )
        previous_year = int(registry["current_feature_year"])
        if registry["raw_snapshot_sha256"] != _raw_snapshot_hash(raw_path, previous_year):
            raise ValueError("Previously observed raw feature rows changed; create a new state version.")
        data, label_manifest = _load_training_data(raw_path, labels_path, as_of_feature_year)
        if registry["labels_snapshot_sha256"] != _label_snapshot_hash(labels_path, previous_year):
            raise ValueError("Previously mature labels changed; create a new state version.")

        seed = int(registry["seed"])
        set_seed(seed)
        previous_by_end = {int(record["end_year"]): record for record in registry["active_models"]}
        required = model_window_ends(as_of_feature_year)
        retained_years = tuple(year for year in required if year in previous_by_end)
        added_years = tuple(year for year in required if year not in previous_by_end)
        removed_years = tuple(sorted(set(previous_by_end) - set(required)))
        if len(added_years) != 1 or len(retained_years) != 2 or len(removed_years) != 1:
            raise AssertionError("A yearly FIFO update must add one, retain two and remove one model.")

        active_records = [previous_by_end[year] for year in retained_years]
        persisted_models: list[FittedWindowModel | PersistedWindowModel] = [
            _load_model(record, state_dir) for record in active_records
        ]
        end_year = added_years[0]
        fitted = _fit_window_model(
            data, list(FEATURE_COLUMNS), window_years(end_year),
            f"seed{seed}_M{end_year}", logger, seed=seed,
        )
        active_records.append(_persist_model(fitted, state_dir, list(FEATURE_COLUMNS)))
        persisted_models.append(fitted)
        active_records.sort(key=lambda record: int(record["end_year"]))

        validation = data[data.fyear == as_of_feature_year - 1].copy()
        thresholds, validation_audit = _validation_thresholds(persisted_models, validation)
        completed_at = _now()
        retired_models = list(registry.get("retired_models", []))
        retired_models.extend({
            **previous_by_end[year],
            "retired_at": completed_at,
            "retired_for_feature_year": as_of_feature_year,
            "retirement_reason": "fifo_capacity_3",
        } for year in removed_years)
        registry.update({
            "current_feature_year": as_of_feature_year,
            "active_models": active_records,
            "retired_models": retired_models,
            "thresholds": thresholds,
            "validation_audit": validation_audit,
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": _sha256(raw_path),
            "raw_snapshot_through_feature_year": as_of_feature_year,
            "raw_snapshot_sha256": _raw_snapshot_hash(raw_path, as_of_feature_year),
            "labels_path": str(labels_path.resolve()),
            "labels_sha256": label_manifest["labels_sha256"],
            "labels_snapshot_through_available_year": as_of_feature_year,
            "labels_snapshot_sha256": _label_snapshot_hash(labels_path, as_of_feature_year),
            "label_source": label_manifest["label_source"],
            "label_evidence_status": label_manifest["evidence_status"],
            "runtime_source_sha256": _sha256(Path(__file__)),
            "updated_at": completed_at,
        })
        registry["updates"].append({
            "action": "update",
            "feature_year": as_of_feature_year,
            "added_end_years": list(added_years),
            "retained_end_years": list(retained_years),
            "removed_end_years": list(removed_years),
            "completed_at": completed_at,
        })
        _write_registry(state_dir, registry)
        return registry


def predict_current(
    state_dir: Path,
    raw_path: Path,
    feature_year: int,
    output_root: Path,
) -> Path:
    registry = _load_registry(state_dir)
    if feature_year != int(registry["current_feature_year"]):
        raise ValueError(
            f"State is prepared for feature year {registry['current_feature_year']}, "
            f"not {feature_year}."
        )
    if registry["raw_snapshot_sha256"] != _raw_snapshot_hash(raw_path, feature_year):
        raise ValueError("Observed raw feature rows changed since the state was prepared.")
    # Deliberately load raw features only. status_label is neither selected nor
    # joined, and the event-label ledger is not opened by this function.
    raw = pd.read_csv(raw_path)
    raw["fyear"] = raw["fyear"].astype(int)
    batch = raw[raw.fyear == feature_year].copy()
    if batch.empty:
        raise ValueError(f"No feature rows found for year {feature_year}.")
    models = [_load_model(record, state_dir) for record in registry["active_models"]]
    proba = np.mean([model.predict_proba(batch) for model in models], axis=0)
    predictions = pd.DataFrame({
        "source_row_id": batch.index.astype(int),
        "company_name": batch.company_name.to_numpy(),
        "feature_year": feature_year,
        "predicted_event_year": feature_year + 1,
        "y_proba": proba,
        "y_pred_f1": (proba >= float(registry["thresholds"]["f1"])).astype(int),
        "y_pred_fpr5": (proba >= float(registry["thresholds"]["fpr5"])).astype(int),
        "f1_threshold": float(registry["thresholds"]["f1"]),
        "fpr5_threshold": float(registry["thresholds"]["fpr5"]),
        "state_updated_at": registry["updated_at"],
        "runtime_version": RUNTIME_VERSION,
    })
    run_dir = output_root / f"feature_{feature_year}_{_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.12g")
    manifest = {
        "runtime_version": RUNTIME_VERSION,
        "created_at": _now(),
        "feature_year": feature_year,
        "predicted_event_year": feature_year + 1,
        "n_predictions": len(predictions),
        "n_positive_labels_read": None,
        "test_label_read": False,
        "labels_path_opened": False,
        "raw_sha256": _sha256(raw_path),
        "registry_sha256": _sha256(_registry_path(state_dir)),
        "predictions_sha256": _sha256(predictions_path),
        "active_model_ids": [record["model_id"] for record in registry["active_models"]],
        "thresholds_selected_on_feature_year": registry["validation_audit"]["feature_year"],
        "research_status": registry["research_status"],
    }
    _atomic_json(run_dir / "prediction_manifest.json", manifest)
    return run_dir


def _add_common_state_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--raw", type=Path, default=US_CSV)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--as-of-feature-year", type=int, required=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-labels")
    prepare.add_argument("--raw", type=Path, default=US_CSV)
    prepare.add_argument("--output", type=Path, default=DEFAULT_LABELS)
    initialize = subparsers.add_parser("initialize")
    _add_common_state_args(initialize)
    initialize.add_argument("--seed", type=int, default=42)
    update = subparsers.add_parser("update")
    _add_common_state_args(update)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    predict.add_argument("--raw", type=Path, default=US_CSV)
    predict.add_argument("--feature-year", type=int, required=True)
    predict.add_argument("--output-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    args = parser.parse_args(argv)
    logger = get_logger("Study3BContinualRuntime", console=True, file=False)

    if args.command == "prepare-labels":
        manifest = prepare_label_ledger(args.raw, args.output)
        logger.info(f"Prepared retrospective label ledger: {manifest['labels_path']}")
    elif args.command == "initialize":
        registry = initialize_state(
            args.state_dir, args.raw, args.labels, args.as_of_feature_year, args.seed, logger
        )
        logger.info(
            f"Initialized feature year {registry['current_feature_year']}: "
            f"{[record['model_id'] for record in registry['active_models']]}"
        )
    elif args.command == "update":
        registry = update_state(
            args.state_dir, args.raw, args.labels, args.as_of_feature_year, logger
        )
        logger.info(f"Updated state to feature year {registry['current_feature_year']}")
    else:
        run_dir = predict_current(args.state_dir, args.raw, args.feature_year, args.output_root)
        logger.info(f"Saved label-blind predictions: {run_dir}")


if __name__ == "__main__":
    main()
