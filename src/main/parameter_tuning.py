from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from main import state_estimation as se

@dataclass(frozen=True)
class AlphaBetaTuningResult:
    """
    This class stores the outputs and file locations from an alpha-beta tuning run.
    """

    grid: pd.DataFrame
    selected: pd.DataFrame
    cache_hit: bool
    grid_path: Path
    selected_path: Path
    metadata_path: Path

def parameter_grid(min_value: float, max_value: float, step: float) -> list[float]:
    """
    This function builds an inclusive numeric grid for one tuning parameter.
    """

    # Validate grid bounds.
    if min_value < 0 or max_value < min_value or step <= 0:
        raise ValueError("Grid bounds must satisfy 0 <= min_value <= max_value and step > 0.")

    # Accumulate values while allowing for small floating-point round-off error.
    values = []
    current = min_value
    tolerance = step / 1_000_000
    while current <= max_value + tolerance:
        values.append(round(current, 10))
        current += step
    return values

def _metadata(
    *,
    runs_input_path: Path,
    confusion_matrix_path: Path,
    transition_matrix_path: Path,
    seed: int,
    train_fraction: float,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    epsilon_g: float,
) -> dict[str, object]:
    """
    This function creates the cache metadata used to identify a specific tuning setup.
    """

    return {
        "runs_input_path": str(runs_input_path),
        "runs_input_sha256": se.file_sha256(runs_input_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "confusion_matrix_sha256": se.file_sha256(confusion_matrix_path),
        "transition_matrix_path": str(transition_matrix_path),
        "transition_matrix_sha256": se.file_sha256(transition_matrix_path),
        "prediction_seed": seed,
        "train_fraction": train_fraction,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_step": alpha_step,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "beta_step": beta_step,
        "epsilon_g": epsilon_g,
        "selection_objective": "training_state_estimation_accuracy",
    }

def _cache_matches(metadata_path: Path, expected: dict[str, object]) -> bool:
    """
    This function checks whether an existing metadata file matches the requested tuning setup.
    """

    # Missing or unreadable metadata means the cache cannot be trusted.
    if not metadata_path.exists():
        return False
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return existing == expected

def _training_rows(predictions: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, int, int]:
    """
    This function selects the training bearings and returns their prediction rows for tuning.
    """

    # Validate the requested split.
    if train_fraction <= 0 or train_fraction > 1:
        raise ValueError("train_fraction must satisfy 0 < train_fraction <= 1.")

    # Select the first fraction of sorted bearing identifiers for training.
    bearing_ids = sorted(predictions["bearing_id"].astype(str).unique(), key=se.numeric_or_text_key)
    train_count = int(len(bearing_ids) * train_fraction)
    if train_count <= 0:
        raise ValueError("No training bearings selected.")

    # Keep only training rows and require at least one operating observation.
    train_ids = set(bearing_ids[:train_count])
    train_rows = predictions[predictions["bearing_id"].astype(str).isin(train_ids)].copy()
    train_rows = train_rows.sort_values(["bearing_id", "t"], kind="stable")
    operating_count = int(train_rows["true_state"].ne("F").sum())
    if operating_count <= 0:
        raise ValueError("No training operating observations found.")
    return train_rows, train_count, operating_count

def _posterior_accuracy(
    train_rows: pd.DataFrame,
    transition_matrix: tuple[tuple[float, ...], ...],
    confusion_probabilities: dict[str, list[tuple[str, float]]],
    *,
    alpha: float,
    beta: float,
    epsilon_g: float,
) -> float:
    """
    This function evaluates state-estimation accuracy for one alpha-beta parameter pair.
    """

    operating_rows = 0
    correct = 0
    current_bearing_id = None
    posterior_belief = se.one_hot_state("H")

    for row in train_rows.itertuples(index=False):
        # Reset the posterior at the start of each bearing trajectory.
        bearing_id = str(row.bearing_id)
        if bearing_id != current_bearing_id:
            current_bearing_id = bearing_id
            posterior_belief = se.one_hot_state("H")

        # Skip failed observations after recording the absorbing failure belief.
        true_state = str(row.true_state)
        if true_state == "F":
            posterior_belief = se.one_hot_state("F")
            continue

        # Predict the next prior and condition it on the bearing still operating.
        raw_prior = (
            se.one_hot_state("H")
            if int(row.t) == 0
            else se.advance_prior(posterior_belief, transition_matrix)
        )
        operating_prior = se.survival_conditioned_prior(raw_prior)

        # Correct the prior with the sampled diagnostic label.
        transient_belief = se.correct_with_diagnostic(
            operating_prior=operating_prior,
            predicted_state=str(row.predicted_state),
            probabilities_by_true_state=confusion_probabilities,
            alpha=alpha,
            beta=beta,
            epsilon_g=epsilon_g,
        )
        posterior_belief = [
            transient_belief[se.TRANSIENT_STATES.index(state)]
            if state in se.TRANSIENT_STATES
            else 0.0
            for state in se.STATES
        ]

        # Score the maximum-posterior transient state against the true state.
        estimated_state = max(
            se.TRANSIENT_STATES,
            key=lambda state: posterior_belief[se.STATES.index(state)],
        )

        operating_rows += 1
        correct += int(estimated_state == true_state)

    return correct / operating_rows

def _selection_key(row: pd.Series) -> tuple[float, ...]:
    """
    This function ranks grid-search rows by accuracy and deterministic tie-breakers.
    """

    alpha = float(row["alpha"])
    beta = float(row["beta"])
    accuracy = float(row["training_accuracy"])
    return (
        -accuracy,
        abs(alpha - 1.0) + abs(beta - 1.0),
        abs(alpha - beta),
        alpha + beta,
        alpha,
        beta,
    )

def _compute_grid(
    *,
    runs_input_path: Path,
    confusion_matrix_path: Path,
    transition_matrix_path: Path,
    seed: int,
    train_fraction: float,
    alpha_values: list[float],
    beta_values: list[float],
    epsilon_g: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function computes the full alpha-beta grid and selects the best parameter pair.
    """

    # Generate fixed diagnostic predictions so every parameter pair is scored on the same labels.
    base_predictions = se.generate_state_estimation_predictions(
        runs_input_path=runs_input_path,
        confusion_matrix_path=confusion_matrix_path,
        transition_matrix_path=transition_matrix_path,
        seed=seed,
        alpha=1.0,
        beta=1.0,
        epsilon_g=epsilon_g,
    )
    train_rows, train_bearings, train_operating_observations = _training_rows(
        base_predictions,
        train_fraction,
    )

    # Load static model inputs used by each grid evaluation.
    transition_matrix = se.load_transition_matrix(transition_matrix_path)
    confusion_probabilities = se.load_confusion_probabilities(confusion_matrix_path)

    # Evaluate every alpha-beta combination.
    rows = []
    for alpha in alpha_values:
        for beta in beta_values:
            rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "training_bearings": train_bearings,
                    "training_operating_observations": train_operating_observations,
                    "training_accuracy": _posterior_accuracy(
                        train_rows,
                        transition_matrix,
                        confusion_probabilities,
                        alpha=alpha,
                        beta=beta,
                        epsilon_g=epsilon_g,
                    ),
                }
            )

    # Select the best row and mark it in the returned grid.
    grid = pd.DataFrame(rows)
    selected_index = min(grid.index, key=lambda index: _selection_key(grid.loc[index]))
    selected = grid.loc[[selected_index]].reset_index(drop=True)
    grid = grid.assign(selected=False)
    grid.loc[selected_index, "selected"] = True
    return grid, selected

def cached_alpha_beta_grid_search(
    *,
    runs_input_path: Path,
    confusion_matrix_path: Path,
    transition_matrix_path: Path,
    output_dir: Path,
    seed: int,
    train_fraction: float,
    alpha_min: float = 0.0,
    alpha_max: float = 2.0,
    alpha_step: float = 0.2,
    beta_min: float = 0.0,
    beta_max: float = 2.0,
    beta_step: float = 0.2,
    epsilon_g: float = 1e-4,
    force: bool = False,
) -> AlphaBetaTuningResult:
    """
    This function runs or loads a cached grid search for alpha and beta tuning parameters.
    """

    # Define cache output locations.
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "alpha_beta_grid_search.csv"
    selected_path = output_dir / "alpha_beta_selected.csv"
    metadata_path = output_dir / "alpha_beta_grid_search_metadata.json"

    # Build metadata that captures all inputs affecting the search result.
    expected_metadata = _metadata(
        runs_input_path=runs_input_path,
        confusion_matrix_path=confusion_matrix_path,
        transition_matrix_path=transition_matrix_path,
        seed=seed,
        train_fraction=train_fraction,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_step=beta_step,
        epsilon_g=epsilon_g,
    )

    # Return cached outputs when all files exist and the metadata matches.
    if (
        not force
        and grid_path.exists()
        and selected_path.exists()
        and _cache_matches(metadata_path, expected_metadata)
    ):
        return AlphaBetaTuningResult(
            grid=pd.read_csv(grid_path),
            selected=pd.read_csv(selected_path),
            cache_hit=True,
            grid_path=grid_path,
            selected_path=selected_path,
            metadata_path=metadata_path,
        )

    # Compute a new grid search and persist the outputs with their metadata.
    grid, selected = _compute_grid(
        runs_input_path=runs_input_path,
        confusion_matrix_path=confusion_matrix_path,
        transition_matrix_path=transition_matrix_path,
        seed=seed,
        train_fraction=train_fraction,
        alpha_values=parameter_grid(alpha_min, alpha_max, alpha_step),
        beta_values=parameter_grid(beta_min, beta_max, beta_step),
        epsilon_g=epsilon_g,
    )
    grid.to_csv(grid_path, index=False)
    selected.to_csv(selected_path, index=False)
    metadata_path.write_text(json.dumps(expected_metadata, indent=2) + "\n", encoding="utf-8")

    return AlphaBetaTuningResult(
        grid=grid,
        selected=selected,
        cache_hit=False,
        grid_path=grid_path,
        selected_path=selected_path,
        metadata_path=metadata_path,
    )
