# Source-of-Truth Decision

## Working Assumption

Use `paper/1MIAC15___conference_version.pdf` as the current manuscript target.

The paper's result sections are:

1. `4.1 Transition-matrix estimation`
2. `4.2 State-estimation performance`
3. `4.3 Maintenance-cost performance`

The current notebook scope covers items 1, 2, and 3: right-censored endpoint
transition-matrix estimation, diagnostic-only versus belief-updated state
estimation, and the CBR0/CBR1 maintenance-cost comparison shown in the
conference paper.

## Reproducibility Caveats

- The executed right-censored EM fit converges in `908` iterations with
  log-likelihood `-341.7368791146924`. The conference PDF text says `899`
  iterations, but the old backup estimator also returns `908` with the current
  code/data, so `899` should be treated as stale text.
- The state-estimation table in the conference PDF is reproduced by using the
  paper-encoded rounded transition matrix in
  `data/transition_matrices/right_censored_endpoint.csv`. Using the
  full-precision EM matrix changes the state-estimation metrics slightly.

## Matching Existing Backup Material

The closest existing backup output is:

```text
2. Temporal Belief Updating/results/section_3_4_right_censored_endpoint/accuracy_first/
```

That output matches the paper-facing state-estimation numbers more closely than the broader
`transition_matrix_comparison/cost/` folder because it selects inference weights
for state-estimation accuracy first, then tunes maintenance thresholds for cost.

## Supplementary Material

The broader transition-matrix comparison folder contains useful analysis across:

- fully observed / transition-time TPM;
- mode-labeled run-to-failure TPM;
- right-censored endpoint TPM;
- CBR0, CBR1, CBR1M, and MLC policies.

Those results should remain supplementary unless the manuscript is updated to
make the three-TPM comparison and learned maintenance classifier part of the
main result story.

## Open Decision for Later

The current single notebook intentionally reproduces only the conference-version
PDF. Add the broader three-TPM comparison only if the manuscript is revised to
make it part of the main result story or as an explicitly marked supplementary
section.
