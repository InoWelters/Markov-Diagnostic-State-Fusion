"""Case-study state-space helpers for the reproducibility notebook."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

STATE_ORDER = ("H", "O1", "O2", "I1", "I2", "B1", "B2", "F")
INTERNAL_STATE_ORDER = ("H", "O_1", "O_2", "I_1", "I_2", "B_1", "B_2", "F")

ALLOWED_TRANSITIONS = {
    "H": ("H", "O1", "I1", "B1"),
    "O1": ("O1", "O2"),
    "O2": ("O2", "F"),
    "I1": ("I1", "I2"),
    "I2": ("I2", "F"),
    "B1": ("B1", "B2"),
    "B2": ("B2", "F"),
    "F": ("F",),
}

DATA_GENERATING_P0 = (
    (0.89, 0.06, 0.00, 0.03, 0.00, 0.02, 0.00, 0.00),
    (0.00, 0.90, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 0.89, 0.00, 0.00, 0.00, 0.00, 0.11),
    (0.00, 0.00, 0.00, 0.80, 0.20, 0.00, 0.00, 0.00),
    (0.00, 0.00, 0.00, 0.00, 0.81, 0.00, 0.00, 0.19),
    (0.00, 0.00, 0.00, 0.00, 0.00, 0.64, 0.36, 0.00),
    (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.55, 0.45),
    (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
)


@dataclass(frozen=True)
class CaseStudyResult:
    """Container for generated Section 4 case-study tables."""

    state_space: pd.DataFrame
    severity_thresholds: pd.DataFrame
    structural_transition_matrix: pd.DataFrame
    structural_transition_mask: pd.DataFrame
    data_generating_transition_matrix: pd.DataFrame
    evaluation_summary: dict[str, object]
    paths: dict[str, Path]
    cache_hit: bool


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "state_space": output_dir / "case_study_state_space.csv",
        "severity_thresholds": output_dir / "case_study_severity_thresholds.csv",
        "structural_transition_matrix": output_dir / "case_study_structural_transition_matrix.csv",
        "structural_transition_mask": output_dir / "case_study_structural_transition_mask.csv",
        "data_generating_transition_matrix": output_dir / "case_study_data_generating_transition_matrix_p0.csv",
        "metadata": output_dir / "case_study_metadata.json",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(
    *,
    dataset4_path: Path,
    runs_input_path: Path,
    train_fraction: float,
) -> dict[str, object]:
    return {
        "schema": "case_study_v1",
        "state_order": list(STATE_ORDER),
        "internal_state_order": list(INTERNAL_STATE_ORDER),
        "dataset4_path": str(dataset4_path),
        "dataset4_sha256": _file_sha256(dataset4_path),
        "runs_input_path": str(runs_input_path),
        "runs_input_sha256": _file_sha256(runs_input_path),
        "train_fraction": train_fraction,
        "p0_source": "Parent TPM_generation.py benchmark fully observed/transition-time matrix.",
        "severity_source": "Draft Section 4.1.1 / conference Section 3.1, based on Zheng et al. (2024).",
    }


def _cache_matches(metadata_path: Path, expected: dict[str, object]) -> bool:
    if not metadata_path.exists():
        return False
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return existing == expected


def state_space_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state": "H", "fault_mode": "none", "severity": 0, "interpretation": "healthy"},
            {"state": "O1", "fault_mode": "outer race", "severity": 1, "interpretation": "mild outer-race degradation"},
            {"state": "O2", "fault_mode": "outer race", "severity": 2, "interpretation": "advanced outer-race degradation"},
            {"state": "I1", "fault_mode": "inner race", "severity": 1, "interpretation": "mild inner-race degradation"},
            {"state": "I2", "fault_mode": "inner race", "severity": 2, "interpretation": "advanced inner-race degradation"},
            {"state": "B1", "fault_mode": "ball defect", "severity": 1, "interpretation": "mild ball-defect degradation"},
            {"state": "B2", "fault_mode": "ball defect", "severity": 2, "interpretation": "advanced ball-defect degradation"},
            {"state": "F", "fault_mode": "failed", "severity": 3, "interpretation": "absorbing failure state"},
        ]
    )


def severity_threshold_table() -> pd.DataFrame:
    rows = []
    for fault_mode, mild_state, advanced_state in (
        ("outer race", "O1", "O2"),
        ("inner race", "I1", "I2"),
        ("ball defect", "B1", "B2"),
    ):
        rows.extend(
            [
                {
                    "fault_mode": fault_mode,
                    "state": mild_state,
                    "severity_level": 1,
                    "severity_label": "mild / pre-accelerated",
                    "fault_length_interval_mm": "0 <= Delta l < 0.5",
                    "physical_stage": "initial plus slow evolution",
                },
                {
                    "fault_mode": fault_mode,
                    "state": advanced_state,
                    "severity_level": 2,
                    "severity_label": "advanced / accelerated",
                    "fault_length_interval_mm": "0.5 <= Delta l <= 0.8",
                    "physical_stage": "accelerated evolution",
                },
            ]
        )
    return pd.DataFrame(rows)


def structural_transition_mask() -> pd.DataFrame:
    mask = pd.DataFrame(
        0,
        index=pd.Index(STATE_ORDER, name="from_state"),
        columns=pd.Index(STATE_ORDER, name="to_state"),
    )
    for from_state, to_states in ALLOWED_TRANSITIONS.items():
        mask.loc[from_state, list(to_states)] = 1
    return mask


def structural_transition_matrix() -> pd.DataFrame:
    matrix = pd.DataFrame(
        "0",
        index=pd.Index(STATE_ORDER, name="from_state"),
        columns=pd.Index(STATE_ORDER, name="to_state"),
    )
    for from_state, to_states in ALLOWED_TRANSITIONS.items():
        for to_state in to_states:
            matrix.loc[from_state, to_state] = "1" if from_state == "F" and to_state == "F" else f"p_{from_state},{to_state}"
    return matrix


def data_generating_transition_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        DATA_GENERATING_P0,
        index=pd.Index(STATE_ORDER, name="state"),
        columns=pd.Index(STATE_ORDER, name="to_state"),
    )


def _load_dataset4(path: Path) -> pd.DataFrame:
    header = path.open("r", encoding="utf-8").readline()
    separator = ";" if header.count(";") >= header.count(",") else ","
    return pd.read_csv(path, sep=separator)


def evaluation_design_summary(
    *,
    dataset4_path: Path,
    runs_input_path: Path,
    train_fraction: float,
) -> dict[str, object]:
    dataset4 = _load_dataset4(dataset4_path)
    runs = pd.read_csv(runs_input_path, usecols=["bearing_id"])

    n_transition = int(dataset4["bearing_id"].nunique())
    failed = int(dataset4["is_failed"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    censored = int(n_transition - failed)
    censoring_times = sorted(dataset4.loc[~dataset4["is_failed"].astype(str).str.lower().isin({"true", "1", "yes"}), "observed_time"].astype(int).unique())

    n_generated = int(runs["bearing_id"].nunique())
    n_train = int(n_generated * train_fraction)
    n_test = int(n_generated - n_train)

    return {
        "transition_trajectories": n_transition,
        "transition_failed": failed,
        "transition_censored": censored,
        "censoring_times": censoring_times,
        "generated_trajectories": n_generated,
        "training_trajectories": n_train,
        "heldout_trajectories": n_test,
        "train_fraction": train_fraction,
    }


def _read_cached(
    paths: dict[str, Path],
    *,
    dataset4_path: Path,
    runs_input_path: Path,
    train_fraction: float,
) -> CaseStudyResult:
    return CaseStudyResult(
        state_space=pd.read_csv(paths["state_space"]),
        severity_thresholds=pd.read_csv(paths["severity_thresholds"]),
        structural_transition_matrix=pd.read_csv(paths["structural_transition_matrix"]).set_index("from_state"),
        structural_transition_mask=pd.read_csv(paths["structural_transition_mask"]).set_index("from_state"),
        data_generating_transition_matrix=pd.read_csv(paths["data_generating_transition_matrix"]).set_index("state"),
        evaluation_summary=evaluation_design_summary(
            dataset4_path=dataset4_path,
            runs_input_path=runs_input_path,
            train_fraction=train_fraction,
        ),
        paths=paths,
        cache_hit=True,
    )


def build_case_study(
    *,
    dataset4_path: Path,
    runs_input_path: Path,
    output_dir: Path,
    train_fraction: float,
    force: bool = False,
) -> CaseStudyResult:
    paths = _paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_metadata = _metadata(
        dataset4_path=dataset4_path,
        runs_input_path=runs_input_path,
        train_fraction=train_fraction,
    )
    required_outputs = [path for key, path in paths.items() if key != "metadata"]
    if (
        not force
        and all(path.exists() for path in required_outputs)
        and _cache_matches(paths["metadata"], expected_metadata)
    ):
        return _read_cached(
            paths,
            dataset4_path=dataset4_path,
            runs_input_path=runs_input_path,
            train_fraction=train_fraction,
        )

    state_space = state_space_table()
    severity_thresholds = severity_threshold_table()
    transition_matrix = structural_transition_matrix()
    transition_mask = structural_transition_mask()
    p0_matrix = data_generating_transition_matrix()
    evaluation_summary = evaluation_design_summary(
        dataset4_path=dataset4_path,
        runs_input_path=runs_input_path,
        train_fraction=train_fraction,
    )

    state_space.to_csv(paths["state_space"], index=False)
    severity_thresholds.to_csv(paths["severity_thresholds"], index=False)
    transition_matrix.to_csv(paths["structural_transition_matrix"])
    transition_mask.to_csv(paths["structural_transition_mask"])
    p0_matrix.to_csv(paths["data_generating_transition_matrix"])
    paths["metadata"].write_text(json.dumps(expected_metadata, indent=2) + "\n", encoding="utf-8")

    return CaseStudyResult(
        state_space=state_space,
        severity_thresholds=severity_thresholds,
        structural_transition_matrix=transition_matrix,
        structural_transition_mask=transition_mask,
        data_generating_transition_matrix=p0_matrix,
        evaluation_summary=evaluation_summary,
        paths=paths,
        cache_hit=False,
    )
