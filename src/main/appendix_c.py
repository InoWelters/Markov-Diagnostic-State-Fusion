"""Appendix C sensitivity analyses and result table generation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from main import maintenance_cost as mc
from main import state_estimation as se

DEFAULT_LIKELIHOOD_FLOORS = (1e-6, 1e-5, 1e-4, 1e-3)
DEFAULT_EXPONENT_SETTINGS = (
    (0.8, 1.8),
    (1.0, 1.6),
    (1.0, 1.8),
    (1.0, 2.0),
    (1.2, 1.8),
)
DEFAULT_COST_RATIOS = (2.0, 5.0, 10.0)

@dataclass(frozen=True)
class AppendixCResult:
    """
    This class stores the Appendix C output tables, cache status, and output paths.
    """
    heldout_summary: pd.DataFrame
    heldout_differences: pd.DataFrame
    confusion_uncertainty: pd.DataFrame
    state_sensitivity: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    policy_decisions: pd.DataFrame
    cache_hit: bool
    paths: dict[str, Path]

def _metadata(
    *,
    runs_input_path: Path,
    confusion_matrix_path: Path,
    transition_matrix_path: Path,
    prediction_seed: int,
    train_fraction: float,
    alpha: float,
    beta: float,
    epsilon_g: float,
    costs: mc.CostParameters,
    n_boot: int,
    bootstrap_seed: int,
    likelihood_floors: tuple[float, ...],
    exponent_settings: tuple[tuple[float, float], ...],
    cost_ratios: tuple[float, ...],
) -> dict[str, object]:
    """
    This function builds a reproducibility metadata record for the Appendix C analyses.
    """
    # Capture input identities and analysis settings
    return {
        "runs_input_path": str(runs_input_path),
        "runs_input_sha256": se.file_sha256(runs_input_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "confusion_matrix_sha256": se.file_sha256(confusion_matrix_path),
        "transition_matrix_path": str(transition_matrix_path),
        "transition_matrix_sha256": se.file_sha256(transition_matrix_path),
        "prediction_seed": prediction_seed,
        "train_fraction": train_fraction,
        "alpha": alpha,
        "beta": beta,
        "epsilon_g": epsilon_g,
        "preventive_cost": costs.preventive,
        "corrective_cost": costs.corrective,
        "n_boot": n_boot,
        "bootstrap_seed": bootstrap_seed,
        "likelihood_floors": list(likelihood_floors),
        "exponent_settings": [list(setting) for setting in exponent_settings],
        "cost_ratios": list(cost_ratios),
    }

def _cache_matches(metadata_path: Path, expected: dict[str, object]) -> bool:
    """
    This function checks whether an existing metadata file matches the expected cache metadata.
    """
    if not metadata_path.exists():
        return False
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return existing == expected

def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "heldout_summary": output_dir / "appendix_c_heldout_bootstrap_summary.csv",
        "heldout_differences": output_dir / "appendix_c_heldout_bootstrap_differences.csv",
        "confusion_uncertainty": output_dir / "appendix_c_confusion_matrix_uncertainty.csv",
        "state_sensitivity": output_dir / "appendix_c_state_estimation_sensitivity.csv",
        "cost_sensitivity": output_dir / "appendix_c_cost_ratio_sensitivity.csv",
        "policy_decisions": output_dir / "appendix_c_policy_decisions.csv",
        "metadata": output_dir / "appendix_c_metadata.json",
    }

def _read_cached(paths: dict[str, Path]) -> AppendixCResult:
    """
    This function loads all cached Appendix C output tables and wraps them in the result object.
    """
    return AppendixCResult(
        heldout_summary=pd.read_csv(paths["heldout_summary"]),
        heldout_differences=pd.read_csv(paths["heldout_differences"]),
        confusion_uncertainty=pd.read_csv(paths["confusion_uncertainty"]),
        state_sensitivity=pd.read_csv(paths["state_sensitivity"]),
        cost_sensitivity=pd.read_csv(paths["cost_sensitivity"]),
        policy_decisions=pd.read_csv(paths["policy_decisions"]),
        cache_hit=True,
        paths=paths,
    )

def _summary_stats(samples: list[float] | np.ndarray, point_estimate: float) -> dict[str, float]:
    """
    This function summarizes bootstrap samples with a point estimate, mean, standard deviation, and percentile interval.
    """
    # Remove invalid bootstrap draws before calculating intervals
    values = np.asarray(samples, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "point_estimate": point_estimate,
            "bootstrap_mean": float("nan"),
            "bootstrap_sd": float("nan"),
            "ci_2_5": float("nan"),
            "ci_97_5": float("nan"),
        }
    return {
        "point_estimate": point_estimate,
        "bootstrap_mean": float(np.mean(finite)),
        "bootstrap_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "ci_2_5": float(np.percentile(finite, 2.5)),
        "ci_97_5": float(np.percentile(finite, 97.5)),
    }

def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")

def _balanced_accuracy(class_correct: np.ndarray, class_total: np.ndarray) -> float:
    """
    This function computes balanced accuracy from per-class correct and total counts.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = class_correct / class_total
    recalls = recalls[class_total > 0]
    return float(np.mean(recalls)) if len(recalls) else float("nan")

def _state_statistics(evaluation_rows: pd.DataFrame) -> tuple[list[str], dict[str, np.ndarray]]:
    """
    This function aggregates held-out state-estimation counts by bearing for bootstrap resampling.
    """
    # Preserve bearing-level units for bootstrap resampling
    bearing_ids = sorted(evaluation_rows["bearing_id"].astype(str).unique(), key=se.numeric_or_text_key)
    state_index = {state: index for index, state in enumerate(se.TRANSIENT_STATES)}
    rows = []

    for bearing_id in bearing_ids:
        group = evaluation_rows[evaluation_rows["bearing_id"].astype(str).eq(bearing_id)]
        true_states = group["true_state"].to_numpy()
        diagnostic = group["diagnostic_estimate"].to_numpy()
        belief = group["belief_estimate"].to_numpy()
        class_total = np.zeros(len(se.TRANSIENT_STATES), dtype=float)
        diagnostic_class_correct = np.zeros(len(se.TRANSIENT_STATES), dtype=float)
        belief_class_correct = np.zeros(len(se.TRANSIENT_STATES), dtype=float)

        # Count class-level outcomes within this bearing
        for true_state, diagnostic_state, belief_state in zip(true_states, diagnostic, belief):
            idx = state_index[true_state]
            class_total[idx] += 1.0
            diagnostic_class_correct[idx] += float(diagnostic_state == true_state)
            belief_class_correct[idx] += float(belief_state == true_state)

        # Separate severity-specific outcomes
        mild_mask = np.isin(true_states, list(se.MILD_STATES))
        advanced_mask = np.isin(true_states, list(se.ADVANCED_STATES))
        rows.append(
            {
                "n": float(len(group)),
                "diagnostic_correct": float(np.sum(diagnostic == true_states)),
                "belief_correct": float(np.sum(belief == true_states)),
                "class_total": class_total,
                "diagnostic_class_correct": diagnostic_class_correct,
                "belief_class_correct": belief_class_correct,
                "mild_total": float(mild_mask.sum()),
                "advanced_total": float(advanced_mask.sum()),
                "diagnostic_mild_correct": float(np.sum(diagnostic[mild_mask] == true_states[mild_mask])),
                "belief_mild_correct": float(np.sum(belief[mild_mask] == true_states[mild_mask])),
                "diagnostic_advanced_correct": float(
                    np.sum(diagnostic[advanced_mask] == true_states[advanced_mask])
                ),
                "belief_advanced_correct": float(np.sum(belief[advanced_mask] == true_states[advanced_mask])),
            }
        )

    return bearing_ids, {
        "n": np.array([row["n"] for row in rows], dtype=float),
        "diagnostic_correct": np.array([row["diagnostic_correct"] for row in rows], dtype=float),
        "belief_correct": np.array([row["belief_correct"] for row in rows], dtype=float),
        "class_total": np.vstack([row["class_total"] for row in rows]).astype(float),
        "diagnostic_class_correct": np.vstack([row["diagnostic_class_correct"] for row in rows]).astype(float),
        "belief_class_correct": np.vstack([row["belief_class_correct"] for row in rows]).astype(float),
        "mild_total": np.array([row["mild_total"] for row in rows], dtype=float),
        "advanced_total": np.array([row["advanced_total"] for row in rows], dtype=float),
        "diagnostic_mild_correct": np.array([row["diagnostic_mild_correct"] for row in rows], dtype=float),
        "belief_mild_correct": np.array([row["belief_mild_correct"] for row in rows], dtype=float),
        "diagnostic_advanced_correct": np.array([row["diagnostic_advanced_correct"] for row in rows], dtype=float),
        "belief_advanced_correct": np.array([row["belief_advanced_correct"] for row in rows], dtype=float),
    }

def _sum_stats(stats: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
    if indices is None:
        return {key: value.sum(axis=0) if value.ndim > 1 else float(value.sum()) for key, value in stats.items()}
    return {key: value[indices].sum(axis=0) if value.ndim > 1 else float(value[indices].sum()) for key, value in stats.items()}

def _state_metric_values(stats: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """
    This function converts aggregated state-estimation counts into diagnostic and belief-update performance metrics.
    """
    # Sum either all bearings or a bootstrap-selected subset
    selected = _sum_stats(stats, indices)
    n = float(selected["n"])
    return {
        "diagnostic": {
            "accuracy": _safe_rate(float(selected["diagnostic_correct"]), n),
            "balanced_accuracy": _balanced_accuracy(
                selected["diagnostic_class_correct"],
                selected["class_total"],
            ),
            "severity_1_exact_accuracy": _safe_rate(
                float(selected["diagnostic_mild_correct"]),
                float(selected["mild_total"]),
            ),
            "severity_2_exact_accuracy": _safe_rate(
                float(selected["diagnostic_advanced_correct"]),
                float(selected["advanced_total"]),
            ),
        },
        "belief": {
            "accuracy": _safe_rate(float(selected["belief_correct"]), n),
            "balanced_accuracy": _balanced_accuracy(
                selected["belief_class_correct"],
                selected["class_total"],
            ),
            "severity_1_exact_accuracy": _safe_rate(
                float(selected["belief_mild_correct"]),
                float(selected["mild_total"]),
            ),
            "severity_2_exact_accuracy": _safe_rate(
                float(selected["belief_advanced_correct"]),
                float(selected["advanced_total"]),
            ),
        },
    }

def _parse_parameters(parameters: str) -> dict[str, str]:
    """
    This function parses a semicolon-separated policy-parameter string into a dictionary.
    """
    parsed = {}
    for part in str(parameters).split(";"):
        key_value = part.strip().split("=", maxsplit=1)
        if len(key_value) == 2:
            parsed[key_value[0]] = key_value[1]
    return parsed

def _thresholds_for_policy(policy_summary: pd.DataFrame, policy: str) -> mc.Thresholds:
    """
    This function extracts optimized maintenance thresholds for one policy from a policy-summary table.
    """
    row = policy_summary[policy_summary["policy"].eq(policy)]
    if row.empty:
        raise ValueError(f"Policy summary does not contain {policy}.")
    parameters = _parse_parameters(str(row["parameters"].iloc[0]))
    return mc.Thresholds(
        eta_O=float(parameters["eta_O"]),
        eta_I=float(parameters["eta_I"]),
        eta_B=float(parameters["eta_B"]),
    )

def _policy_decisions(
    predictions: pd.DataFrame,
    confusion_matrix_path: Path,
    policy_summary: pd.DataFrame,
    train_fraction: float,
    costs: mc.CostParameters,
) -> pd.DataFrame:
    """
    This function reconstructs per-bearing maintenance decisions for CBR0 and CBR1 on held-out runs.
    """
    # Rebuild held-out runs and recover the optimized thresholds
    probabilities = se.load_confusion_probabilities(confusion_matrix_path)
    runs = mc.prediction_runs(predictions, probabilities)
    _, evaluation_runs = mc.split_runs(runs, train_fraction)
    policy_specs = [
        ("CBR0", "diagnostic", _thresholds_for_policy(policy_summary, "CBR0")),
        ("CBR1", "posterior", _thresholds_for_policy(policy_summary, "CBR1")),
    ]
    rows = []

    # Classify the realized maintenance event for each held-out bearing
    for policy, source, thresholds in policy_specs:
        for bearing_id, run in evaluation_runs.items():
            trigger = mc._first_trigger(run, thresholds, source)
            failure = mc._first_failure(run)
            if failure is not None and (
                trigger is None
                or int(failure["t"]) <= int(trigger["observation"]["t"])
            ):
                event = "CM"
                event_t = int(failure["t"])
                cycle_length = mc.effective_cycle_length(event_t)
                total_cost = costs.corrective
                trigger_state = ""
                trigger_score = float("nan")
            elif trigger is not None:
                event = "PM"
                event_t = int(trigger["observation"]["t"])
                cycle_length = mc.effective_cycle_length(event_t)
                total_cost = costs.preventive
                trigger_state = str(trigger["state"])
                trigger_score = float(trigger["score"])
            else:
                event = "CENSORED"
                event_t = int(run[-1]["t"])
                cycle_length = mc.effective_cycle_length(event_t)
                total_cost = 0.0
                trigger_state = ""
                trigger_score = float("nan")

            rows.append(
                {
                    "policy": policy,
                    "bearing_id": bearing_id,
                    "event": event,
                    "event_t": event_t,
                    "cycle_length": cycle_length,
                    "total_cost": total_cost,
                    "trigger_state": trigger_state,
                    "trigger_score": trigger_score,
                }
            )

    return pd.DataFrame(rows)

def _maintenance_statistics(
    bearing_ids: list[str],
    policy_decisions: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    """
    This function aggregates per-bearing maintenance cost and event counts for each policy.
    """
    # Collapse event-level decisions to one row per bearing and policy
    stats = {}
    for policy in ("CBR0", "CBR1"):
        grouped = (
            policy_decisions[policy_decisions["policy"].eq(policy)]
            .groupby("bearing_id", sort=False)
            .agg(
                total_cost=("total_cost", "sum"),
                total_cycle_length=("cycle_length", "sum"),
                cycles=("bearing_id", "size"),
                preventive_replacements=("event", lambda values: int(np.sum(values.astype(str).eq("PM")))),
                corrective_replacements=("event", lambda values: int(np.sum(values.astype(str).eq("CM")))),
            )
            .reindex(bearing_ids)
            .fillna(0.0)
        )
        stats[policy] = {column: grouped[column].to_numpy(dtype=float) for column in grouped.columns}
    return stats

def _maintenance_metric_values(
    stats: dict[str, dict[str, np.ndarray]],
    indices: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """
    This function converts maintenance aggregates into cost, cycle-length, and replacement metrics.
    """
    # Convert policy aggregates into comparable maintenance metrics
    values = {}
    for policy, arrays in stats.items():
        selected = {
            key: float(value.sum() if indices is None else value[indices].sum())
            for key, value in arrays.items()
        }
        values[policy] = {
            "cost_rate": _safe_rate(selected["total_cost"], selected["total_cycle_length"]),
            "cost_per_bearing": _safe_rate(selected["total_cost"], selected["cycles"]),
            "cycle_length": _safe_rate(selected["total_cycle_length"], selected["cycles"]),
            "preventive_replacements": selected["preventive_replacements"],
            "corrective_replacements": selected["corrective_replacements"],
        }
    return values

def _bootstrap_heldout(
    evaluation_rows: pd.DataFrame,
    policy_decisions: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function bootstraps held-out bearings to quantify uncertainty in state and maintenance metrics.
    """
    # Pre-compute bearing-level aggregates before resampling
    bearing_ids, state_stats = _state_statistics(evaluation_rows)
    maintenance_stats = _maintenance_statistics(bearing_ids, policy_decisions)
    point_state = _state_metric_values(state_stats)
    point_maintenance = _maintenance_metric_values(maintenance_stats)

    samples: dict[tuple[str, str], list[float]] = {}
    diff_samples: dict[tuple[str, str], list[float]] = {}
    rng = np.random.default_rng(seed)
    n_bearings = len(bearing_ids)

    # Resample bearings with replacement
    for _ in range(n_boot):
        indices = rng.integers(0, n_bearings, size=n_bearings)
        state_values = _state_metric_values(state_stats, indices)
        maintenance_values = _maintenance_metric_values(maintenance_stats, indices)

        for method, method_values in state_values.items():
            for metric, value in method_values.items():
                samples.setdefault((method, metric), []).append(value)
        for policy, policy_values in maintenance_values.items():
            for metric, value in policy_values.items():
                samples.setdefault((policy, metric), []).append(value)

        for metric in ("accuracy", "balanced_accuracy", "severity_2_exact_accuracy"):
            diff_samples.setdefault(("belief_minus_diagnostic_label", metric), []).append(
                state_values["belief"][metric] - state_values["diagnostic"][metric]
            )
        for metric in ("cost_rate", "cycle_length", "corrective_replacements"):
            diff_samples.setdefault(("belief_policy_minus_diagnostic_policy", metric), []).append(
                maintenance_values["CBR1"][metric] - maintenance_values["CBR0"][metric]
            )

    method_labels = {
        "diagnostic": "Diagnostic-only label",
        "belief": "Belief-updated label",
        "CBR0": "Diagnostic-evidence policy",
        "CBR1": "Belief-updated policy",
    }
    point_values = {
        **{("diagnostic", key): value for key, value in point_state["diagnostic"].items()},
        **{("belief", key): value for key, value in point_state["belief"].items()},
        **{("CBR0", key): value for key, value in point_maintenance["CBR0"].items()},
        **{("CBR1", key): value for key, value in point_maintenance["CBR1"].items()},
    }

    # Convert bootstrap samples into summary rows
    summary_rows = []
    for key, values in samples.items():
        method, metric = key
        summary_rows.append(
            {
                "method": method_labels.get(method, method),
                "metric": metric,
                **_summary_stats(values, point_values[key]),
            }
        )

    # Compute paired point-estimate differences for the same held-out set
    diff_point_values = {
        ("belief_minus_diagnostic_label", "accuracy"): point_state["belief"]["accuracy"]
        - point_state["diagnostic"]["accuracy"],
        ("belief_minus_diagnostic_label", "balanced_accuracy"): point_state["belief"]["balanced_accuracy"]
        - point_state["diagnostic"]["balanced_accuracy"],
        ("belief_minus_diagnostic_label", "severity_2_exact_accuracy"): point_state["belief"][
            "severity_2_exact_accuracy"
        ]
        - point_state["diagnostic"]["severity_2_exact_accuracy"],
        ("belief_policy_minus_diagnostic_policy", "cost_rate"): point_maintenance["CBR1"]["cost_rate"]
        - point_maintenance["CBR0"]["cost_rate"],
        ("belief_policy_minus_diagnostic_policy", "cycle_length"): point_maintenance["CBR1"]["cycle_length"]
        - point_maintenance["CBR0"]["cycle_length"],
        ("belief_policy_minus_diagnostic_policy", "corrective_replacements"): point_maintenance["CBR1"][
            "corrective_replacements"
        ]
        - point_maintenance["CBR0"]["corrective_replacements"],
    }
    difference_rows = []
    for key, values in diff_samples.items():
        comparison, metric = key
        difference_rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                **_summary_stats(values, diff_point_values[key]),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(difference_rows)

def _wilson_interval(successes: float, total: float, z: float = 1.959963984540054) -> tuple[float, float]:
    """
    This function computes a Wilson binomial confidence interval for class-level recall.
    """
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def _confusion_uncertainty(confusion_matrix_path: Path) -> pd.DataFrame:
    """
    This function summarizes recall uncertainty in the confusion matrix with Wilson intervals.
    """
    counts = pd.read_csv(confusion_matrix_path).set_index("true_state")
    # Estimate one recall interval per true transient class
    rows = []
    for state in se.TRANSIENT_STATES:
        label = se.PAPER_LABEL_BY_STATE[state]
        row = counts.loc[label]
        total = float(row.sum())
        successes = float(row[label])
        point = _safe_rate(successes, total)
        lower, upper = _wilson_interval(successes, total)
        rows.append(
            {
                "class": label,
                "method": "wilson_binomial",
                "n": int(total),
                "recall": point,
                "ci_2_5": lower,
                "ci_97_5": upper,
            }
        )
    return pd.DataFrame(rows)

def _held_out_metrics(
    predictions: pd.DataFrame,
    train_fraction: float,
) -> pd.Series:
    """
    This function computes held-out belief-updated state-estimation metrics for one prediction set.
    """
    evaluation_rows, _, _ = se.held_out_operating_rows(predictions, train_fraction=train_fraction)
    evaluation_rows = se.add_estimate_columns(evaluation_rows)
    metrics = se.state_estimation_metrics(evaluation_rows).set_index("estimator")
    belief = metrics.loc["Belief-updated label"]
    return belief

def _state_sensitivity(
    *,
    runs_input_path: Path,
    confusion_matrix_path: Path,
    transition_matrix_path: Path,
    prediction_seed: int,
    train_fraction: float,
    alpha: float,
    beta: float,
    epsilon_g: float,
    likelihood_floors: tuple[float, ...],
    exponent_settings: tuple[tuple[float, float], ...],
) -> pd.DataFrame:
    """
    This function evaluates state-estimation sensitivity across likelihood floors and fusion exponents.
    """
    # Vary the likelihood floor while holding the exponents fixed
    rows = []
    for floor in likelihood_floors:
        predictions = se.generate_state_estimation_predictions(
            runs_input_path=runs_input_path,
            confusion_matrix_path=confusion_matrix_path,
            transition_matrix_path=transition_matrix_path,
            seed=prediction_seed,
            alpha=alpha,
            beta=beta,
            epsilon_g=floor,
        )
        metrics = _held_out_metrics(predictions, train_fraction)
        rows.append(
            {
                "sensitivity": "likelihood_floor",
                "setting": f"epsilon_G={floor:g}",
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "severity_2_exact_accuracy": metrics["severity_2_exact_rate"],
            }
        )

    # Vary the fusion exponents while holding the likelihood floor fixed
    for candidate_alpha, candidate_beta in exponent_settings:
        predictions = se.generate_state_estimation_predictions(
            runs_input_path=runs_input_path,
            confusion_matrix_path=confusion_matrix_path,
            transition_matrix_path=transition_matrix_path,
            seed=prediction_seed,
            alpha=candidate_alpha,
            beta=candidate_beta,
            epsilon_g=epsilon_g,
        )
        metrics = _held_out_metrics(predictions, train_fraction)
        rows.append(
            {
                "sensitivity": "fusion_exponents",
                "setting": f"alpha={candidate_alpha:g}; beta={candidate_beta:g}",
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "severity_2_exact_accuracy": metrics["severity_2_exact_rate"],
            }
        )
    return pd.DataFrame(rows)

def _cost_ratio_sensitivity(
    policy_decisions: pd.DataFrame,
    *,
    preventive_cost: float,
    cost_ratios: tuple[float, ...],
) -> pd.DataFrame:
    """
    This function recalculates policy cost rates under alternative corrective-to-preventive cost ratios.
    """
    # Reprice the same policy decisions under each cost ratio
    rows = []
    method_by_policy = {
        "CBR0": "Diagnostic-evidence",
        "CBR1": "Corrected-belief",
    }
    for ratio in cost_ratios:
        corrective_cost = preventive_cost * ratio
        for policy in ("CBR0", "CBR1"):
            subset = policy_decisions[policy_decisions["policy"].eq(policy)]
            preventive = int(subset["event"].eq("PM").sum())
            corrective = int(subset["event"].eq("CM").sum())
            total_cost = preventive * preventive_cost + corrective * corrective_cost
            total_cycle_length = float(subset["cycle_length"].sum())
            rows.append(
                {
                    "C_C_over_C_P": ratio,
                    "policy": method_by_policy[policy],
                    "cost_rate": _safe_rate(total_cost, total_cycle_length),
                    "corrective_replacements": corrective,
                }
            )
    return pd.DataFrame(rows)

def _ensure_predictions(
    predictions: pd.DataFrame | None,
    *,
    runs_input_path: Path,
    confusion_matrix_path: Path,
    transition_matrix_path: Path,
    prediction_seed: int,
    alpha: float,
    beta: float,
    epsilon_g: float,
) -> pd.DataFrame:
    """
    This function returns supplied predictions or generates them from the configured inputs.
    """
    if predictions is not None:
        return predictions.copy()
    return se.generate_state_estimation_predictions(
        runs_input_path=runs_input_path,
        confusion_matrix_path=confusion_matrix_path,
        transition_matrix_path=transition_matrix_path,
        seed=prediction_seed,
        alpha=alpha,
        beta=beta,
        epsilon_g=epsilon_g,
    )

def _ensure_policy_summary(
    policy_summary: pd.DataFrame | None,
    predictions: pd.DataFrame,
    *,
    confusion_matrix_path: Path,
    train_fraction: float,
    alpha: float,
    beta: float,
    costs: mc.CostParameters,
) -> pd.DataFrame:
    """
    This function returns a supplied policy summary or computes one from predictions and costs.
    """
    if policy_summary is not None:
        return policy_summary.copy()
    return mc.policy_summary(
        predictions,
        confusion_matrix_path,
        train_fraction=train_fraction,
        alpha=alpha,
        beta=beta,
        costs=costs,
    )

def cached_appendix_c(
    *,
    runs_input_path: Path,
    confusion_matrix_path: Path,
    transition_matrix_path: Path,
    output_dir: Path,
    prediction_seed: int,
    train_fraction: float,
    alpha: float,
    beta: float,
    epsilon_g: float,
    costs: mc.CostParameters,
    predictions: pd.DataFrame | None = None,
    evaluation_rows: pd.DataFrame | None = None,
    policy_summary: pd.DataFrame | None = None,
    n_boot: int = 2000,
    bootstrap_seed: int = 12345,
    likelihood_floors: tuple[float, ...] = DEFAULT_LIKELIHOOD_FLOORS,
    exponent_settings: tuple[tuple[float, float], ...] = DEFAULT_EXPONENT_SETTINGS,
    cost_ratios: tuple[float, ...] = DEFAULT_COST_RATIOS,
    force: bool = False,
) -> AppendixCResult:
    """
    This function runs or loads the cached Appendix C uncertainty and sensitivity analyses.
    """
    # Prepare output locations and reproducibility metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_dir)
    expected_metadata = _metadata(
        runs_input_path=runs_input_path,
        confusion_matrix_path=confusion_matrix_path,
        transition_matrix_path=transition_matrix_path,
        prediction_seed=prediction_seed,
        train_fraction=train_fraction,
        alpha=alpha,
        beta=beta,
        epsilon_g=epsilon_g,
        costs=costs,
        n_boot=n_boot,
        bootstrap_seed=bootstrap_seed,
        likelihood_floors=likelihood_floors,
        exponent_settings=exponent_settings,
        cost_ratios=cost_ratios,
    )
    # Return cached outputs when the inputs and settings are unchanged
    cache_files = [path for key, path in paths.items() if key != "metadata"]
    if (
        not force
        and all(path.exists() for path in cache_files)
        and _cache_matches(paths["metadata"], expected_metadata)
    ):
        return _read_cached(paths)

    # Materialize the prediction and policy inputs needed for Appendix C
    predictions = _ensure_predictions(
        predictions,
        runs_input_path=runs_input_path,
        confusion_matrix_path=confusion_matrix_path,
        transition_matrix_path=transition_matrix_path,
        prediction_seed=prediction_seed,
        alpha=alpha,
        beta=beta,
        epsilon_g=epsilon_g,
    )
    if evaluation_rows is None:
        evaluation_rows, _, _ = se.held_out_operating_rows(predictions, train_fraction=train_fraction)
        evaluation_rows = se.add_estimate_columns(evaluation_rows)
    else:
        evaluation_rows = evaluation_rows.copy()
    policy_summary = _ensure_policy_summary(
        policy_summary,
        predictions,
        confusion_matrix_path=confusion_matrix_path,
        train_fraction=train_fraction,
        alpha=alpha,
        beta=beta,
        costs=costs,
    )

    # Run the held-out uncertainty and sensitivity analyses
    policy_decisions = _policy_decisions(
        predictions,
        confusion_matrix_path,
        policy_summary,
        train_fraction,
        costs,
    )
    heldout_summary, heldout_differences = _bootstrap_heldout(
        evaluation_rows,
        policy_decisions,
        n_boot=n_boot,
        seed=bootstrap_seed,
    )
    confusion_uncertainty = _confusion_uncertainty(confusion_matrix_path)
    state_sensitivity = _state_sensitivity(
        runs_input_path=runs_input_path,
        confusion_matrix_path=confusion_matrix_path,
        transition_matrix_path=transition_matrix_path,
        prediction_seed=prediction_seed,
        train_fraction=train_fraction,
        alpha=alpha,
        beta=beta,
        epsilon_g=epsilon_g,
        likelihood_floors=likelihood_floors,
        exponent_settings=exponent_settings,
    )
    cost_sensitivity = _cost_ratio_sensitivity(
        policy_decisions,
        preventive_cost=costs.preventive,
        cost_ratios=cost_ratios,
    )

    # Persist all output tables and the cache metadata
    frames = {
        "heldout_summary": heldout_summary,
        "heldout_differences": heldout_differences,
        "confusion_uncertainty": confusion_uncertainty,
        "state_sensitivity": state_sensitivity,
        "cost_sensitivity": cost_sensitivity,
        "policy_decisions": policy_decisions,
    }
    for key, frame in frames.items():
        frame.to_csv(paths[key], index=False)
    paths["metadata"].write_text(json.dumps(expected_metadata, indent=2) + "\n", encoding="utf-8")

    return AppendixCResult(
        heldout_summary=heldout_summary,
        heldout_differences=heldout_differences,
        confusion_uncertainty=confusion_uncertainty,
        state_sensitivity=state_sensitivity,
        cost_sensitivity=cost_sensitivity,
        policy_decisions=policy_decisions,
        cache_hit=False,
        paths=paths,
    )
