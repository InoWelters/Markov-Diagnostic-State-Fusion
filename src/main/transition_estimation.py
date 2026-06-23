from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATE_ORDER = ("H", "O1", "O2", "I1", "I2", "B1", "B2", "F")
TRANSIENT_STATES = STATE_ORDER[:-1]
FAILURE_STATE = "F"

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

FAULT_MODE_TO_STATES = {
    "Outer race": ("O1", "O2"),
    "Inner race": ("I1", "I2"),
    "Ball defect": ("B1", "B2"),
}

STATE_INDEX = {state: index for index, state in enumerate(STATE_ORDER)}
TRANSIENT_INDEX = {state: index for index, state in enumerate(TRANSIENT_STATES)}


@dataclass(frozen=True)
class RightCensoredFit:
    transition_matrix: pd.DataFrame
    expected_transition_counts: pd.DataFrame
    expected_state_summary: pd.DataFrame
    convergence_trace: pd.DataFrame
    prepared_observations: pd.DataFrame
    failed_mode_time_counts: pd.Series
    censored_state_time_counts: pd.Series
    metadata: dict[str, object]


def empty_transition_frame(fill_value: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        fill_value,
        index=pd.Index(STATE_ORDER, name="from_state"),
        columns=pd.Index(STATE_ORDER, name="to_state"),
        dtype=float,
    )


def canonicalize_transition_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0.0).astype(float)


def symmetric_initial_transition_matrix() -> pd.DataFrame:
    """
    Create a symmetric initial transition matrix that respects the allowed Markov-chain transitions.
    """
    matrix = empty_transition_frame(0.0)

    # Initialize all structurally valid transitions with equal probability.
    matrix.loc["H", ["H", "O1", "I1", "B1"]] = 0.25
    for state in ("O1", "O2", "I1", "I2", "B1", "B2"):
        matrix.loc[state, list(ALLOWED_TRANSITIONS[state])] = 0.5
    matrix.loc["F", "F"] = 1.0
    return matrix


def validate_transition_matrix(matrix: pd.DataFrame) -> None:
    """
    Validate that a transition matrix is row-stochastic and contains no forbidden transitions.
    """
    matrix = canonicalize_transition_frame(matrix)

    # Check stochastic rows and structural zeros.
    for state in STATE_ORDER:
        row = matrix.loc[state]
        if abs(float(row.sum()) - 1.0) >= 1e-9:
            raise ValueError(f"Transition-matrix row {state} does not sum to 1.")
        forbidden = [target for target in STATE_ORDER if target not in ALLOWED_TRANSITIONS[state]]
        if float(row.loc[forbidden].abs().sum()) >= 1e-9:
            raise ValueError(f"Transition-matrix row {state} contains structural non-zero entries.")


def _parse_boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    """
    Parse a dataframe column into boolean values while reporting invalid source values.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    # Normalize common string and numeric encodings.
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    parsed = normalized.map(mapping)

    # Fail fast when values cannot be interpreted unambiguously.
    if parsed.isna().any():
        bad_values = sorted(series.loc[parsed.isna()].astype(str).unique().tolist())
        raise ValueError(f"Could not parse boolean values in {column_name}: {bad_values}")
    return parsed.astype(bool)


def load_mode_augmented_censored_data(path: Path) -> pd.DataFrame:
    """
    Load and validate the mode-augmented censored dataset used by the EM estimator.
    """
    frame = pd.read_csv(path, sep=";")
    required_columns = {
        "bearing_id",
        "observed_state",
        "observed_time",
        "is_failed",
        "fault_mode",
    }

    # Verify the required input schema.
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required dataset-4 columns: {missing}")

    # Normalize column types and missing fault-mode labels.
    prepared = frame.copy()
    prepared["observed_state"] = prepared["observed_state"].astype(str).str.strip()
    prepared["observed_time"] = pd.to_numeric(prepared["observed_time"], errors="raise").astype(int)
    prepared["is_failed"] = _parse_boolean_series(prepared["is_failed"], "is_failed")
    prepared["fault_mode"] = prepared["fault_mode"].astype("string").str.strip()
    prepared["fault_mode"] = prepared["fault_mode"].mask(
        prepared["fault_mode"].isna() | prepared["fault_mode"].eq(""),
        pd.NA,
    )

    # Enforce the known state and fault-mode vocabularies.
    unknown_states = sorted(set(prepared["observed_state"]) - set(STATE_ORDER))
    if unknown_states:
        raise ValueError(f"Unknown observed states in dataset 4: {unknown_states}")

    unknown_modes = sorted(set(prepared["fault_mode"].dropna()) - set(FAULT_MODE_TO_STATES))
    if unknown_modes:
        raise ValueError(f"Unknown fault modes in dataset 4: {unknown_modes}")

    # Check consistency between censoring status and observed fault-mode labels.
    if prepared.loc[prepared["is_failed"], "fault_mode"].isna().any():
        raise ValueError("Failed rows must include an observed fault mode.")
    if prepared.loc[~prepared["is_failed"], "fault_mode"].notna().any():
        raise ValueError("Censored rows must not include a future fault-mode label.")

    return prepared


def grouped_observation_counts(prepared: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Aggregate prepared observations into sufficient count tables for failed and censored records.
    """
    # Count failures by observed fault mode and failure time.
    failed_mode_time_counts = (
        prepared.loc[prepared["is_failed"]]
        .groupby(["fault_mode", "observed_time"], sort=True)
        .size()
        .astype(float)
        .rename("count")
    )

    # Count censored observations by observed endpoint state and censoring time.
    censored_state_time_counts = (
        prepared.loc[~prepared["is_failed"]]
        .groupby(["observed_state", "observed_time"], sort=True)
        .size()
        .astype(float)
        .rename("count")
    )
    return failed_mode_time_counts, censored_state_time_counts


def _expected_count_outputs(
    expected_counts_array: np.ndarray,
    observed_log_likelihood: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Convert expected transition counts into labeled outputs and per-state summary statistics.
    """
    expected_counts = pd.DataFrame(
        expected_counts_array,
        index=pd.Index(STATE_ORDER, name="from_state"),
        columns=pd.Index(STATE_ORDER, name="to_state"),
    )

    # Summarize expected visits and exits per source state.
    expected_source_counts = expected_counts.sum(axis=1).rename("expected_source_count")
    expected_exit_counts = pd.Series(0.0, index=STATE_ORDER, name="expected_exit_count")
    for state in TRANSIENT_STATES:
        exit_targets = [target for target in ALLOWED_TRANSITIONS[state] if target != state]
        expected_exit_counts.loc[state] = float(expected_counts.loc[state, exit_targets].sum())

    # Convert exit counts into empirical exit probabilities where possible.
    exit_probabilities = pd.Series(0.0, index=STATE_ORDER, name="exit_probability")
    positive_source_counts = expected_source_counts > 0.0
    exit_probabilities.loc[positive_source_counts] = (
        expected_exit_counts.loc[positive_source_counts]
        / expected_source_counts.loc[positive_source_counts]
    )
    exit_probabilities.loc[FAILURE_STATE] = 0.0
    expected_state_summary = pd.concat(
        [expected_source_counts, expected_exit_counts, exit_probabilities],
        axis=1,
    )
    return expected_counts, expected_state_summary, float(observed_log_likelihood)


def _expected_counts_from_mode_labeled_failures(
    failed_mode_time_counts: pd.Series,
    transition_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Compute expected transition counts for failed observations with known fault modes.
    """
    matrix = canonicalize_transition_frame(transition_matrix)
    expected_counts_array = np.zeros((len(STATE_ORDER), len(STATE_ORDER)), dtype=float)
    observed_log_likelihood = 0.0

    if failed_mode_time_counts.empty:
        return _expected_count_outputs(expected_counts_array, observed_log_likelihood)

    for fault_mode, (stage1_state, stage2_state) in FAULT_MODE_TO_STATES.items():
        if fault_mode not in failed_mode_time_counts.index.get_level_values(0):
            continue

        # Restrict the chain to the healthy state and the two degradation stages for this fault mode.
        counts_for_mode = failed_mode_time_counts.xs(fault_mode, level=0)
        local_states = ("H", stage1_state, stage2_state)
        local_positions = [STATE_INDEX[state] for state in local_states]
        local_grid = np.ix_(local_positions, local_positions)

        q_block = matrix.loc[list(local_states), list(local_states)].to_numpy(dtype=float)
        r_vector = matrix.loc[list(local_states), FAILURE_STATE].to_numpy(dtype=float)
        max_tau = int(counts_for_mode.index.max())

        # Compute forward probabilities before the absorbing failure transition.
        forward = np.zeros((max_tau, len(local_states)), dtype=float)
        forward[0, 0] = 1.0
        for step in range(1, max_tau):
            forward[step] = forward[step - 1] @ q_block

        # Compute exact future absorption probabilities for each remaining horizon.
        exact_absorption = np.zeros((max_tau + 1, len(local_states)), dtype=float)
        exact_absorption[1] = r_vector
        for remaining_steps in range(2, max_tau + 1):
            exact_absorption[remaining_steps] = q_block @ exact_absorption[remaining_steps - 1]

        for tau_f, multiplicity in counts_for_mode.items():
            tau_f = int(tau_f)
            likelihood = float(forward[tau_f - 1] @ r_vector)
            if likelihood <= 0.0 or not np.isfinite(likelihood):
                raise ValueError("Non-positive likelihood for a mode-labeled failed observation.")

            observed_log_likelihood += float(multiplicity) * float(np.log(likelihood))
            scale = float(multiplicity) / likelihood

            # Attribute expected pre-failure transitions across all possible hidden paths.
            for step in range(tau_f - 1):
                continuation = exact_absorption[tau_f - step - 1]
                transition_slice = (forward[step][:, None] * q_block) * continuation[None, :]
                expected_counts_array[local_grid] += scale * transition_slice

            # Attribute the final transition into the absorbing failure state.
            failure_transition_slice = scale * (forward[tau_f - 1] * r_vector)
            expected_counts_array[local_positions, STATE_INDEX[FAILURE_STATE]] += failure_transition_slice

    return _expected_count_outputs(expected_counts_array, observed_log_likelihood)


def _expected_counts_from_censored_endpoints(
    censored_state_time_counts: pd.Series,
    transition_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Compute expected transition counts for right-censored observations with known endpoint states.
    """
    matrix = canonicalize_transition_frame(transition_matrix)
    expected_counts_array = np.zeros((len(STATE_ORDER), len(STATE_ORDER)), dtype=float)
    observed_log_likelihood = 0.0

    if censored_state_time_counts.empty:
        return _expected_count_outputs(expected_counts_array, observed_log_likelihood)

    q_block = matrix.loc[list(TRANSIENT_STATES), list(TRANSIENT_STATES)].to_numpy(dtype=float)
    max_time = int(censored_state_time_counts.index.get_level_values(1).max())

    # Compute forward probabilities over the non-absorbing state space.
    forward = np.zeros((max_time + 1, len(TRANSIENT_STATES)), dtype=float)
    forward[0, TRANSIENT_INDEX["H"]] = 1.0
    for step in range(1, max_time + 1):
        forward[step] = forward[step - 1] @ q_block

    for (endpoint_state, observed_time), multiplicity in censored_state_time_counts.items():
        observed_time = int(observed_time)
        endpoint_position = TRANSIENT_INDEX[endpoint_state]
        likelihood = float(forward[observed_time, endpoint_position])
        if likelihood <= 0.0 or not np.isfinite(likelihood):
            raise ValueError("Non-positive likelihood for a censored endpoint observation.")

        observed_log_likelihood += float(multiplicity) * float(np.log(likelihood))
        scale = float(multiplicity) / likelihood

        # Compute backward probabilities conditioned on the observed endpoint state.
        endpoint_vector = np.zeros(len(TRANSIENT_STATES), dtype=float)
        endpoint_vector[endpoint_position] = 1.0
        backward = np.zeros((observed_time + 1, len(TRANSIENT_STATES)), dtype=float)
        backward[0] = endpoint_vector
        for remaining_steps in range(1, observed_time + 1):
            backward[remaining_steps] = q_block @ backward[remaining_steps - 1]

        # Attribute expected transitions across hidden paths that end in the observed state.
        for step in range(observed_time):
            continuation = backward[observed_time - step - 1]
            transition_slice = (forward[step][:, None] * q_block) * continuation[None, :]
            expected_counts_array[
                np.ix_(
                    [STATE_INDEX[state] for state in TRANSIENT_STATES],
                    [STATE_INDEX[state] for state in TRANSIENT_STATES],
                )
            ] += scale * transition_slice

    return _expected_count_outputs(expected_counts_array, observed_log_likelihood)


def expected_counts(
    failed_mode_time_counts: pd.Series,
    censored_state_time_counts: pd.Series,
    transition_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Combine expected counts and log-likelihood terms from failed and censored observations.
    """
    # Run the E-step separately for failure and censoring contributions.
    failed_counts, _, failed_log_likelihood = _expected_counts_from_mode_labeled_failures(
        failed_mode_time_counts,
        transition_matrix,
    )
    censored_counts, _, censored_log_likelihood = _expected_counts_from_censored_endpoints(
        censored_state_time_counts,
        transition_matrix,
    )

    # Merge both expected-count components into one transition-count table.
    combined_counts = failed_counts.to_numpy(dtype=float) + censored_counts.to_numpy(dtype=float)
    return _expected_count_outputs(
        combined_counts,
        float(failed_log_likelihood + censored_log_likelihood),
    )


def _m_step(
    expected_transition_counts: pd.DataFrame,
    previous_transition_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Update transition probabilities from expected counts while preserving structural constraints.
    """
    updated_matrix = empty_transition_frame(0.0)
    previous_matrix = canonicalize_transition_frame(previous_transition_matrix)

    # Normalize each row over only its allowed outgoing transitions.
    for state in STATE_ORDER:
        allowed_targets = list(ALLOWED_TRANSITIONS[state])
        if state == FAILURE_STATE:
            updated_matrix.loc[state, FAILURE_STATE] = 1.0
            continue

        row_counts = expected_transition_counts.loc[state, allowed_targets]
        source_count = float(row_counts.sum())
        if source_count > 0.0:
            updated_matrix.loc[state, allowed_targets] = row_counts / source_count
        else:
            updated_matrix.loc[state, allowed_targets] = previous_matrix.loc[state, allowed_targets]

    return updated_matrix


def estimate_right_censored_endpoint_em(
    data: pd.DataFrame,
    *,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> RightCensoredFit:
    """
    Estimate a right-censored endpoint transition matrix with the EM algorithm.
    """
    prepared = data.copy()

    # Confirm that the input dataframe has already been prepared and validated.
    if not {"is_failed", "observed_state", "observed_time", "fault_mode"}.issubset(prepared.columns):
        raise ValueError("Input data must be prepared with load_mode_augmented_censored_data().")

    # Initialize sufficient statistics and the starting transition matrix.
    failed_mode_time_counts, censored_state_time_counts = grouped_observation_counts(prepared)
    current_matrix = symmetric_initial_transition_matrix()
    previous_log_likelihood: float | None = None
    convergence_rows = []
    converged = False

    # Iterate between expected-count computation and transition-probability updates.
    for iteration in range(1, max_iter + 1):
        transition_counts, state_summary, log_likelihood = expected_counts(
            failed_mode_time_counts,
            censored_state_time_counts,
            current_matrix,
        )
        updated_matrix = _m_step(transition_counts, current_matrix)
        max_parameter_change = float(
            np.max(
                np.abs(
                    updated_matrix.to_numpy(dtype=float)
                    - current_matrix.to_numpy(dtype=float)
                )
            )
        )
        delta_log_likelihood = (
            np.nan
            if previous_log_likelihood is None
            else float(log_likelihood - previous_log_likelihood)
        )
        convergence_rows.append(
            {
                "iteration": iteration,
                "log_likelihood": log_likelihood,
                "delta_log_likelihood": delta_log_likelihood,
                "max_parameter_change": max_parameter_change,
            }
        )

        # Stop when both likelihood and parameter updates are within tolerance.
        current_matrix = updated_matrix
        if (
            previous_log_likelihood is not None
            and abs(delta_log_likelihood) <= tol
            and max_parameter_change <= tol
        ):
            converged = True
            break
        previous_log_likelihood = float(log_likelihood)

    # Recompute final summaries for the returned fitted model.
    transition_counts, state_summary, final_log_likelihood = expected_counts(
        failed_mode_time_counts,
        censored_state_time_counts,
        current_matrix,
    )
    validate_transition_matrix(current_matrix)

    # Store fit diagnostics and dataset-level metadata.
    metadata = {
        "n_bearings": int(prepared["bearing_id"].nunique()),
        "n_rows": int(len(prepared)),
        "n_failed": int(prepared["is_failed"].sum()),
        "n_censored": int((~prepared["is_failed"]).sum()),
        "termination_reason": "converged" if converged else "max_iter_reached",
        "converged": bool(converged),
        "n_iterations": int(len(convergence_rows)),
        "final_log_likelihood": float(final_log_likelihood),
        "tol": float(tol),
        "max_iter": int(max_iter),
    }

    return RightCensoredFit(
        transition_matrix=current_matrix,
        expected_transition_counts=transition_counts,
        expected_state_summary=state_summary,
        convergence_trace=pd.DataFrame(convergence_rows),
        prepared_observations=prepared,
        failed_mode_time_counts=failed_mode_time_counts,
        censored_state_time_counts=censored_state_time_counts,
        metadata=metadata,
    )


def transition_matrix_for_state_filter(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Rename transition-matrix state labels into the format expected by the state filter.
    """
    label_map = {
        "H": "H",
        "O1": "O_1",
        "O2": "O_2",
        "I1": "I_1",
        "I2": "I_2",
        "B1": "B_1",
        "B2": "B_2",
        "F": "F",
    }
    renamed = matrix.rename(index=label_map, columns=label_map)
    renamed.index.name = "state"
    return renamed
