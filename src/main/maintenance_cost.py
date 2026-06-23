"""Maintenance policy cost evaluation and threshold search utilities."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from itertools import product

import pandas as pd

from main import state_estimation as se


MODE_ORDER = ("O", "I", "B")
MODE_SECOND_STAGE = {
    "O": "O_2",
    "I": "I_2",
    "B": "B_2",
}

from main import state_estimation as se


@dataclass(frozen=True)
class CostParameters:
    """
    This class stores the unit costs used when evaluating maintenance policies.
    """

    inspection: float = 2.0
    preventive: float = 10.0
    corrective: float = 50.0


@dataclass(frozen=True)
class Thresholds:
    """
    This class stores the replacement thresholds for the second-stage fault modes.
    """

    eta_O: float
    eta_I: float
    eta_B: float


@dataclass(frozen=True)
class PolicyEvaluation:
    """
    This class stores the aggregate cost and event outcomes for one evaluated policy.
    """

    policy: str
    thresholds: Thresholds
    parameters: dict[str, str]
    total_cost: float
    total_cycle_length: int
    cycles: int
    event_counts: Counter[str]

    @property
    def cost_rate(self) -> float:
        return self.total_cost / self.total_cycle_length

    @property
    def average_cost(self) -> float:
        return self.total_cost / self.cycles

    @property
    def average_cycle_length(self) -> float:
        return self.total_cycle_length / self.cycles


def threshold_grid() -> list[float]:
    return [round(index / 10, 10) for index in range(11)]


def effective_cycle_length(event_t: int) -> int:
    return max(int(event_t), 1)


def diagnostic_evidence_from_label(
    predicted_state: str,
    probabilities_by_true_state: dict[str, list[tuple[str, float]]],
) -> dict[str, float]:
    """
    This function converts one diagnostic label into normalized evidence over transient states.
    """

    # Handle failed-state labels separately because maintenance decisions use transient evidence.
    if predicted_state == "F":
        return {state: 0.0 for state in se.TRANSIENT_STATES}

    # Collect the diagnostic probability of the predicted label under each possible true state.
    predicted_label = se.PAPER_LABEL_BY_STATE[predicted_state]
    values = {}
    for true_state in se.TRANSIENT_STATES:
        true_label = se.PAPER_LABEL_BY_STATE[true_state]
        row = dict(probabilities_by_true_state[true_label])
        values[true_state] = row.get(predicted_label, 0.0)

    # Normalize the evidence so that it can be compared directly with posterior beliefs.
    total = sum(values.values())
    if total <= 0:
        raise ValueError(f"Cannot normalize diagnostic evidence for predicted state {predicted_state}.")
    return {state: value / total for state, value in values.items()}


def _posterior_from_row(row: pd.Series) -> dict[str, float]:
    return {
        state: float(row[f"b_{se.PAPER_LABEL_BY_STATE[state]}"])
        for state in se.TRANSIENT_STATES
    }


def prediction_runs(
    predictions: pd.DataFrame,
    probabilities_by_true_state: dict[str, list[tuple[str, float]]],
) -> OrderedDict[str, list[dict[str, object]]]:
    """
    This function groups prediction rows into ordered bearing-level runs with diagnostic and posterior scores.
    """

    # Sort the data so that each bearing's observations are processed in time order.
    runs: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    for _, row in predictions.sort_values(["bearing_id", "t"], kind="stable").iterrows():
        bearing_id = str(row["bearing_id"])
        predicted_state = str(row["predicted_state"])

        # Store the fields needed by the threshold policy evaluator.
        observation = {
            "bearing_id": bearing_id,
            "t": int(row["t"]),
            "true_state": str(row["true_state"]),
            "predicted_state": predicted_state,
            "diagnostic": diagnostic_evidence_from_label(
                predicted_state,
                probabilities_by_true_state,
            ),
            "posterior": _posterior_from_row(row),
        }
        runs.setdefault(bearing_id, []).append(observation)
    return runs


def split_runs(
    runs: OrderedDict[str, list[dict[str, object]]],
    train_fraction: float,
) -> tuple[OrderedDict[str, list[dict[str, object]]], OrderedDict[str, list[dict[str, object]]]]:
    """
    This function splits bearing-level runs into deterministic training and evaluation subsets.
    """

    bearing_ids = sorted(runs, key=se.numeric_or_text_key)
    train_count = int(len(bearing_ids) * train_fraction)
    train_ids = bearing_ids[:train_count]
    evaluation_ids = bearing_ids[train_count:]
    return (
        OrderedDict((bearing_id, runs[bearing_id]) for bearing_id in train_ids),
        OrderedDict((bearing_id, runs[bearing_id]) for bearing_id in evaluation_ids),
    )


def _score(observation: dict[str, object], state: str, source: str) -> float:
    """
    This function reads a state score from either diagnostic evidence or posterior belief.
    """

    values = observation[source]
    if not isinstance(values, dict):
        raise TypeError(f"Observation field {source} is not a score dictionary.")
    return float(values.get(state, 0.0))


def _first_trigger(
    run: list[dict[str, object]],
    thresholds: Thresholds,
    source: str,
) -> dict[str, object] | None:
    """
    This function finds the earliest preventive-maintenance trigger in one bearing run.
    """

    # Map each fault mode to the threshold that should trigger preventive replacement.
    selected = None
    threshold_by_mode = {
        "O": thresholds.eta_O,
        "I": thresholds.eta_I,
        "B": thresholds.eta_B,
    }

    # Search each mode until the score first exceeds its mode-specific threshold.
    for mode in MODE_ORDER:
        state = MODE_SECOND_STAGE[mode]
        eta = threshold_by_mode[mode]
        for observation in run:
            if observation["true_state"] == "F":
                break
            score = _score(observation, state, source)
            if score > eta:
                candidate = {
                    "observation": observation,
                    "state": state,
                    "score": score,
                }

                # Keep the earliest trigger, using the score to break ties at the same time.
                if selected is None:
                    selected = candidate
                else:
                    selected_t = int(selected["observation"]["t"])
                    candidate_t = int(observation["t"])
                    if candidate_t < selected_t or (
                        candidate_t == selected_t and score > float(selected["score"])
                    ):
                        selected = candidate
                break
    return selected


def _first_failure(run: list[dict[str, object]]) -> dict[str, object] | None:
    """
    This function returns the first observed failure in one bearing run.
    """

    for observation in run:
        if observation["true_state"] == "F":
            return observation
    return None


def evaluate_threshold_policy(
    runs: OrderedDict[str, list[dict[str, object]]],
    *,
    policy: str,
    thresholds: Thresholds,
    source: str,
    costs: CostParameters,
    parameters: dict[str, str],
) -> PolicyEvaluation:
    """
    This function evaluates one threshold policy over a collection of bearing runs.
    """

    # Accumulate cost, cycle length, and event counts across all bearing life cycles.
    total_cost = 0.0
    total_cycle_length = 0
    event_counts: Counter[str] = Counter()

    for run in runs.values():
        trigger = _first_trigger(run, thresholds, source)
        failure = _first_failure(run)

        # Corrective maintenance occurs when failure arrives before any preventive trigger.
        if failure is not None and (
            trigger is None
            or int(failure["t"]) <= int(trigger["observation"]["t"])
        ):
            total_cost += costs.corrective
            total_cycle_length += effective_cycle_length(int(failure["t"]))
            event_counts["CM"] += 1
        elif trigger is not None:
            total_cost += costs.preventive
            total_cycle_length += effective_cycle_length(int(trigger["observation"]["t"]))
            event_counts["PM"] += 1
        else:
            total_cycle_length += effective_cycle_length(int(run[-1]["t"]))
            event_counts["CENSORED"] += 1

    # Return derived metrics through PolicyEvaluation properties.
    return PolicyEvaluation(
        policy=policy,
        thresholds=thresholds,
        parameters=parameters,
        total_cost=total_cost,
        total_cycle_length=total_cycle_length,
        cycles=len(runs),
        event_counts=event_counts,
    )


def _optimization_key(evaluation: PolicyEvaluation) -> tuple[float, ...]:
    """
    This function builds the deterministic tie-breaking key used to select threshold policies.
    """

    thresholds = evaluation.thresholds
    return (
        evaluation.cost_rate,
        evaluation.cost_rate,
        evaluation.average_cost,
        -evaluation.average_cycle_length,
        -thresholds.eta_O,
        -thresholds.eta_I,
        -thresholds.eta_B,
    )


def optimize_threshold_policy(
    train_runs: OrderedDict[str, list[dict[str, object]]],
    evaluation_runs: OrderedDict[str, list[dict[str, object]]],
    *,
    policy: str,
    source: str,
    costs: CostParameters,
    train_fraction: float,
    alpha: float | None = None,
    beta: float | None = None,
) -> PolicyEvaluation:
    """
    This function selects thresholds on training runs and evaluates the selected policy on held-out runs.
    """

    # Initialize the exhaustive grid search over all threshold triples.
    best_train_evaluation: PolicyEvaluation | None = None
    best_thresholds: Thresholds | None = None
    eta_values = threshold_grid()

    # Evaluate every combination of outer-race, inner-race, and ball-defect thresholds.
    for eta_O, eta_I, eta_B in product(eta_values, repeat=3):
        thresholds = Thresholds(eta_O=eta_O, eta_I=eta_I, eta_B=eta_B)
        parameters = policy_parameters(
            thresholds,
            train_fraction=train_fraction,
            train_cycles=len(train_runs),
            evaluation_cycles=len(evaluation_runs),
            alpha=alpha,
            beta=beta,
        )
        candidate = evaluate_threshold_policy(
            train_runs,
            policy=policy,
            thresholds=thresholds,
            source=source,
            costs=costs,
            parameters=parameters,
        )
        if best_train_evaluation is None or _optimization_key(candidate) < _optimization_key(best_train_evaluation):
            best_train_evaluation = candidate
            best_thresholds = thresholds

    if best_thresholds is None:
        raise ValueError(f"No threshold candidates were evaluated for {policy}.")

    # Re-evaluate the selected thresholds on the held-out evaluation runs.
    parameters = policy_parameters(
        best_thresholds,
        train_fraction=train_fraction,
        train_cycles=len(train_runs),
        evaluation_cycles=len(evaluation_runs),
        alpha=alpha,
        beta=beta,
    )
    return evaluate_threshold_policy(
        evaluation_runs,
        policy=policy,
        thresholds=best_thresholds,
        source=source,
        costs=costs,
        parameters=parameters,
    )


def _format_number(value: float | int) -> str:
    return f"{float(value):.10f}".rstrip("0").rstrip(".")


def policy_parameters(
    thresholds: Thresholds,
    *,
    train_fraction: float,
    train_cycles: int,
    evaluation_cycles: int,
    alpha: float | None = None,
    beta: float | None = None,
) -> dict[str, str]:
    """
    This function formats policy and split settings for reporting in the summary table.
    """

    # Include belief-update tuning parameters only for policies that use them.
    parameters: dict[str, str] = {}
    if alpha is not None:
        parameters["alpha"] = _format_number(alpha)
    if beta is not None:
        parameters["beta"] = _format_number(beta)

    # Record threshold and data-split settings for reproducible policy summaries.
    parameters.update(
        {
            "eta_O": _format_number(thresholds.eta_O),
            "eta_I": _format_number(thresholds.eta_I),
            "eta_B": _format_number(thresholds.eta_B),
            "train_fraction": _format_number(train_fraction),
            "train_cycles": str(train_cycles),
            "evaluation_cycles": str(evaluation_cycles),
        }
    )
    return parameters


def parameter_string(parameters: dict[str, str]) -> str:
    return "; ".join(f"{key}={value}" for key, value in parameters.items())


def evaluation_row(evaluation: PolicyEvaluation, costs: CostParameters) -> dict[str, object]:
    """
    This function converts a policy evaluation into one flat row for the output summary table.
    """

    return {
        "policy": evaluation.policy,
        "parameters": parameter_string(evaluation.parameters),
        "cost_rate": evaluation.cost_rate,
        "ECC": evaluation.average_cost,
        "ECL": evaluation.average_cycle_length,
        "total_cost": evaluation.total_cost,
        "total_cycle_length": evaluation.total_cycle_length,
        "cycles": evaluation.cycles,
        "preventive_replacements": evaluation.event_counts["PM"],
        "corrective_replacements": evaluation.event_counts["CM"],
        "censored_cycles": evaluation.event_counts["CENSORED"],
        "inspections": 0,
        "inspection_cost": costs.inspection,
        "preventive_cost": costs.preventive,
        "corrective_cost": costs.corrective,
    }


def policy_summary(
    predictions: pd.DataFrame,
    confusion_matrix_path,
    *,
    train_fraction: float,
    alpha: float,
    beta: float,
    costs: CostParameters,
) -> pd.DataFrame:
    """
    This function compares diagnostic-only and belief-updated maintenance policies in one summary table.
    """

    # Build bearing-level runs from the state-estimation predictions and diagnostic confusion data.
    probabilities = se.load_confusion_probabilities(confusion_matrix_path)
    runs = prediction_runs(predictions, probabilities)
    train_runs, evaluation_runs = split_runs(runs, train_fraction)

    # Optimize the diagnostic-only policy and the posterior-belief policy under the same cost settings.
    cbr0 = optimize_threshold_policy(
        train_runs,
        evaluation_runs,
        policy="CBR0",
        source="diagnostic",
        costs=costs,
        train_fraction=train_fraction,
    )
    cbr1 = optimize_threshold_policy(
        train_runs,
        evaluation_runs,
        policy="CBR1",
        source="posterior",
        costs=costs,
        train_fraction=train_fraction,
        alpha=alpha,
        beta=beta,
    )
    return pd.DataFrame([evaluation_row(cbr0, costs), evaluation_row(cbr1, costs)])
