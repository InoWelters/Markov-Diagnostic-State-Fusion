"""State-estimation helpers for diagnostic and belief-updated inference."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import pandas as pd

STATES = ("H", "O_1", "O_2", "I_1", "I_2", "B_1", "B_2", "F")
TRANSIENT_STATES = tuple(state for state in STATES if state != "F")
MILD_STATES = {"O_1", "I_1", "B_1"}
ADVANCED_STATES = {"O_2", "I_2", "B_2"}

PAPER_LABEL_BY_STATE = {
    "H": "H",
    "O_1": "O1",
    "O_2": "O2",
    "I_1": "I1",
    "I_2": "I2",
    "B_1": "B1",
    "B_2": "B2",
    "F": "F",
}
STATE_BY_PAPER_LABEL = {value: key for key, value in PAPER_LABEL_BY_STATE.items()}

def file_sha256(path: Path) -> str:
    """
    This function computes the SHA-256 checksum for a file.
    """
    digest = hashlib.sha256()

    # Stream file contents into the checksum digest.
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def belief_columns(states: tuple[str, ...] = STATES) -> list[str]:
    return [f"b_{PAPER_LABEL_BY_STATE[state]}" for state in states]

def validate_transition_matrix(matrix: tuple[tuple[float, ...], ...]) -> None:
    """
    This function verifies that a transition matrix has the expected shape and row probabilities.
    """
    if len(matrix) != len(STATES):
        raise ValueError("Transition matrix must have one row per state.")

    # Check row length and stochasticity for every state.
    for row_index, row in enumerate(matrix):
        if len(row) != len(STATES):
            raise ValueError(f"Transition matrix row {row_index} has the wrong length.")
        if abs(sum(row) - 1.0) >= 1e-9:
            raise ValueError(f"Transition matrix row {row_index} does not sum to 1.")

def load_transition_matrix(path: Path) -> tuple[tuple[float, ...], ...]:
    """
    This function loads and validates a transition matrix from a CSV file.
    """
    frame = pd.read_csv(path)
    state_column = "state" if "state" in frame.columns else "from_state"
    if list(frame[state_column]) != list(STATES):
        raise ValueError(f"Transition matrix must use state order: {STATES}")

    # Convert the CSV rows into the internal tuple-of-tuples representation.
    rows: list[tuple[float, ...]] = []
    for _, row in frame.iterrows():
        values = []
        for state in STATES:
            column = state if state in frame.columns else PAPER_LABEL_BY_STATE[state]
            values.append(float(row[column]))
        rows.append(tuple(values))

    # Validate the loaded matrix before returning it.
    matrix = tuple(rows)
    validate_transition_matrix(matrix)
    return matrix

def load_confusion_probabilities(confusion_matrix_path: Path) -> dict[str, list[tuple[str, float]]]:
    """
    This function converts confusion-matrix counts into prediction probabilities by true state.
    """
    frame = pd.read_csv(confusion_matrix_path)
    if "true_state" not in frame.columns:
        raise ValueError("Confusion matrix must include a true_state column.")

    # Normalize each true-state row into diagnostic-label probabilities.
    prediction_columns = [column for column in frame.columns if column != "true_state"]
    probabilities_by_true_state: dict[str, list[tuple[str, float]]] = {}
    for _, row in frame.iterrows():
        true_label = str(row["true_state"])
        counts = [float(row[column]) if pd.notna(row[column]) else 0.0 for column in prediction_columns]
        total = sum(counts)
        if total <= 0:
            raise ValueError(f"Confusion matrix row for {true_label} has no positive counts.")
        probabilities_by_true_state[true_label] = [
            (predicted_label, count / total)
            for predicted_label, count in zip(prediction_columns, counts)
        ]
    return probabilities_by_true_state

def one_hot_state(state: str) -> list[float]:
    return [1.0 if candidate == state else 0.0 for candidate in STATES]

def advance_prior(
    prior: list[float],
    transition_matrix: tuple[tuple[float, ...], ...],
) -> list[float]:
    return [
        sum(prior[row_index] * transition_matrix[row_index][column_index] for row_index in range(len(STATES)))
        for column_index in range(len(STATES))
    ]

def normalize(probabilities: list[float]) -> list[float]:
    """
    This function rescales a probability vector so its entries sum to one.
    """
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("Cannot normalize probabilities with non-positive total.")
    return [probability / total for probability in probabilities]

def survival_conditioned_prior(raw_prior: list[float]) -> list[float]:
    return normalize([raw_prior[STATES.index(state)] for state in TRANSIENT_STATES])

def sample_diagnostic_label(
    true_state: str,
    probabilities_by_true_state: dict[str, list[tuple[str, float]]],
    rng: random.Random,
) -> str:
    """
    This function samples a diagnostic prediction label conditional on the true state.
    """
    true_label = PAPER_LABEL_BY_STATE[true_state]

    # Handle failed-state observations when no confusion row is provided.
    if true_label == "F" and true_label not in probabilities_by_true_state:
        return "F"
    if true_label not in probabilities_by_true_state:
        raise ValueError(f"Confusion matrix has no row for true state {true_label}.")

    # Draw a predicted label and map it back to the internal state notation.
    predicted_labels, probabilities = zip(*probabilities_by_true_state[true_label])
    sampled_label = rng.choices(predicted_labels, weights=probabilities, k=1)[0]
    if sampled_label not in STATE_BY_PAPER_LABEL:
        raise ValueError(f"Unknown predicted state label: {sampled_label}")
    return STATE_BY_PAPER_LABEL[sampled_label]

def diagnostic_likelihoods(
    predicted_state: str,
    probabilities_by_true_state: dict[str, list[tuple[str, float]]],
    epsilon_g: float,
) -> list[float]:
    """
    This function computes diagnostic likelihoods for all transient true states.
    """
    predicted_label = PAPER_LABEL_BY_STATE[predicted_state]
    likelihoods = []

    # Read the probability of the observed diagnostic label under each true state.
    for true_state in TRANSIENT_STATES:
        true_label = PAPER_LABEL_BY_STATE[true_state]
        row_probabilities = dict(probabilities_by_true_state[true_label])
        likelihoods.append(max(row_probabilities.get(predicted_label, 0.0), epsilon_g))
    return likelihoods

def correct_with_diagnostic(
    operating_prior: list[float],
    predicted_state: str,
    probabilities_by_true_state: dict[str, list[tuple[str, float]]],
    alpha: float,
    beta: float,
    epsilon_g: float,
) -> list[float]:
    """
    This function updates an operating-state prior using diagnostic evidence and weighting parameters.
    """
    # Convert the diagnostic output into state-wise likelihoods.
    likelihoods = diagnostic_likelihoods(
        predicted_state=predicted_state,
        probabilities_by_true_state=probabilities_by_true_state,
        epsilon_g=epsilon_g,
    )

    # Combine prior and diagnostic likelihoods in log space for numerical stability.
    log_scores = []
    for prior_probability, likelihood in zip(operating_prior, likelihoods):
        if prior_probability <= 0 and alpha > 0:
            log_scores.append(float("-inf"))
            continue
        if likelihood <= 0 and beta > 0:
            log_scores.append(float("-inf"))
            continue

        prior_term = 0.0 if alpha == 0 else alpha * math.log(prior_probability)
        likelihood_term = 0.0 if beta == 0 else beta * math.log(likelihood)
        log_scores.append(prior_term + likelihood_term)

    # Convert log scores back into normalized probabilities.
    max_log_score = max(log_scores)
    if max_log_score == float("-inf"):
        raise ValueError("Diagnostic correction produced zero probability for every transient state.")

    unnormalized = [
        0.0 if log_score == float("-inf") else math.exp(log_score - max_log_score)
        for log_score in log_scores
    ]
    return normalize(unnormalized)

def generate_state_estimation_predictions(
    runs_input_path: Path,
    confusion_matrix_path: Path,
    transition_matrix_path: Path,
    *,
    seed: int,
    alpha: float,
    beta: float,
    epsilon_g: float,
) -> pd.DataFrame:
    """
    This function generates diagnostic and belief-updated state predictions for bearing runs.
    """
    # Load model inputs and run observations.
    transition_matrix = load_transition_matrix(transition_matrix_path)
    confusion_probabilities = load_confusion_probabilities(confusion_matrix_path)
    runs = pd.read_csv(runs_input_path)
    required_columns = {"bearing_id", "t", "state"}
    if not required_columns.issubset(runs.columns):
        raise ValueError("Runs input must contain bearing_id, t, and state columns.")

    # Initialize random sampling and recurrent belief state.
    rng = random.Random(seed)
    rows = []
    current_bearing_id = None
    posterior_belief = one_hot_state("H")
    transient_b_columns = belief_columns(TRANSIENT_STATES)

    # Process each bearing trajectory in time order.
    for row in runs.itertuples(index=False):
        bearing_id = str(row.bearing_id)
        if bearing_id != current_bearing_id:
            current_bearing_id = bearing_id
            posterior_belief = one_hot_state("H")

        true_state = str(row.state)
        t = int(row.t)
        predicted_state = sample_diagnostic_label(true_state, confusion_probabilities, rng)

        # Update the belief vector unless the bearing has failed.
        if true_state == "F":
            posterior_belief = one_hot_state("F")
        else:
            raw_prior = one_hot_state("H") if t == 0 else advance_prior(posterior_belief, transition_matrix)
            operating_prior = survival_conditioned_prior(raw_prior)
            corrected_transient_belief = correct_with_diagnostic(
                operating_prior=operating_prior,
                predicted_state=predicted_state,
                probabilities_by_true_state=confusion_probabilities,
                alpha=alpha,
                beta=beta,
                epsilon_g=epsilon_g,
            )
            transient_belief_by_column = dict(zip(transient_b_columns, corrected_transient_belief))
            posterior_belief = [
                transient_belief_by_column.get(column, 0.0)
                for column in belief_columns()
            ]

        # Store the true state, sampled diagnostic label, and posterior belief vector.
        rows.append(
            {
                "bearing_id": bearing_id,
                "t": t,
                "true_state": true_state,
                "predicted_state": predicted_state,
                "operating": int(true_state != "F"),
                **{column: value for column, value in zip(belief_columns(), posterior_belief)},
            }
        )

    return pd.DataFrame(rows)

def numeric_or_text_key(value: object) -> tuple[int, int | str]:
    """
    This function creates a sortable key that orders numeric identifiers before text identifiers.
    """
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)

def held_out_operating_rows(
    predictions: pd.DataFrame,
    *,
    train_fraction: float,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    This function splits bearing predictions into training identifiers and held-out operating rows.
    """
    # Partition bearing identifiers deterministically.
    bearing_ids = sorted(predictions["bearing_id"].astype(str).unique(), key=numeric_or_text_key)
    train_count = int(len(bearing_ids) * train_fraction)
    train_ids = bearing_ids[:train_count]
    evaluation_ids = bearing_ids[train_count:]

    # Keep only operating observations from held-out bearings.
    operating = predictions[predictions["true_state"].ne("F")].copy()
    evaluation_rows = operating[operating["bearing_id"].astype(str).isin(evaluation_ids)].copy()
    if evaluation_rows.empty:
        raise ValueError("No held-out operating rows found.")
    return evaluation_rows, train_ids, evaluation_ids

def add_estimate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds diagnostic-only and belief-updated point estimates to a prediction frame.
    """
    result = frame.copy()
    transient_columns = belief_columns(TRANSIENT_STATES)

    # Use the sampled diagnostic label as the diagnostic-only estimate.
    result["diagnostic_estimate"] = result["predicted_state"]

    # Use the maximum posterior belief among transient states as the belief-updated estimate.
    result["belief_estimate"] = (
        result[transient_columns]
        .idxmax(axis=1)
        .map({f"b_{PAPER_LABEL_BY_STATE[state]}": state for state in TRANSIENT_STATES})
    )
    return result

def _safe_mean(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")

def balanced_accuracy(frame: pd.DataFrame, estimate_column: str) -> float:
    """
    This function computes the mean recall across transient states for one estimator.
    """
    recalls = []

    # Average only over states that appear in the evaluation frame.
    for state in TRANSIENT_STATES:
        rows = frame[frame["true_state"].eq(state)]
        if rows.empty:
            continue
        recalls.append(_safe_mean(rows[estimate_column].eq(rows["true_state"])))
    return float(sum(recalls) / len(recalls))

def state_estimation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """
    This function summarizes overall and severity-specific state-estimation metrics.
    """
    rows = []

    # Compute the same metric set for each estimator variant.
    for estimator, estimate_column in [
        ("Diagnostic-only label", "diagnostic_estimate"),
        ("Belief-updated label", "belief_estimate"),
    ]:
        true_state = frame["true_state"]
        estimate = frame[estimate_column]
        mild_mask = true_state.isin(MILD_STATES)
        advanced_mask = true_state.isin(ADVANCED_STATES)
        rows.append(
            {
                "estimator": estimator,
                "operating_observations": len(frame),
                "accuracy": _safe_mean(estimate.eq(true_state)),
                "balanced_accuracy": balanced_accuracy(frame, estimate_column),
                "severity_1_exact_rate": _safe_mean(estimate[mild_mask].eq(true_state[mild_mask])),
                "severity_2_detection_rate": _safe_mean(estimate[advanced_mask].isin(ADVANCED_STATES)),
                "severity_2_exact_rate": _safe_mean(estimate[advanced_mask].eq(true_state[advanced_mask])),
            }
        )
    return pd.DataFrame(rows)

def per_state_recall(frame: pd.DataFrame) -> pd.DataFrame:
    """
    This function computes diagnostic and belief-updated recall for each transient state.
    """
    rows = []

    # Build one recall row per transient true state.
    for state in TRANSIENT_STATES:
        state_rows = frame[frame["true_state"].eq(state)]
        rows.append(
            {
                "state": PAPER_LABEL_BY_STATE[state],
                "held_out_observations": len(state_rows),
                "diagnostic_recall": _safe_mean(state_rows["diagnostic_estimate"].eq(state_rows["true_state"])),
                "belief_updated_recall": _safe_mean(state_rows["belief_estimate"].eq(state_rows["true_state"])),
            }
        )
    return pd.DataFrame(rows)

def confusion_counts(frame: pd.DataFrame, estimate_column: str) -> pd.DataFrame:
    """
    This function counts true-versus-estimated labels for a selected estimator.
    """
    # Initialize a full transient-state confusion table.
    labels = [PAPER_LABEL_BY_STATE[state] for state in TRANSIENT_STATES]
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=int)

    # Accumulate observed true-estimate pairs.
    for true_state, estimate in zip(frame["true_state"], frame[estimate_column]):
        counts.loc[PAPER_LABEL_BY_STATE[true_state], PAPER_LABEL_BY_STATE[estimate]] += 1
    counts.index.name = "true_state"
    counts.columns.name = "predicted_state"
    return counts

def row_percentages(counts: pd.DataFrame) -> pd.DataFrame:
    """
    This function converts confusion-count rows into row-wise percentages.
    """
    row_totals = counts.sum(axis=1).replace(0, pd.NA)
    return counts.div(row_totals, axis=0).fillna(0.0) * 100.0
