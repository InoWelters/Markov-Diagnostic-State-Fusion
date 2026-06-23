# Paper Reproducible Results

This folder is the clean working area for manuscript-facing notebooks.
The two original top-level folders remain the backup/source archive and should not
be edited while this folder is being built.

## Current Source-of-Truth Decision

The working source of truth is:

```text
paper/1MIAC15___conference_version.pdf
```

The current implemented scope covers the conference-version result sections. It
is aligned with the focused right-censored endpoint workflow:

- right-censored endpoint transition-matrix estimation from dataset 4;
- diagnostic-only versus belief-updated state-estimation results;
- CBR0 and CBR1 maintenance-cost results from the paper.

The broader three-transition-matrix comparison, CBR1M, and MLC results are useful
supplementary material, but they are not treated as the primary manuscript target
unless the paper is revised to include them.

## Structure

```text
notebooks/
  paper_results.ipynb          # single executed paper-results notebook

data/
  raw/
    transition_estimation/     # dataset 4 retained for right-censored context
    diagnostics/               # diagnostic confusion matrix
  generated/                   # generated simulation input copied for reproducibility
  transition_matrices/         # fitted right-censored endpoint matrix

src/
  paper_results/

results/
  tables/
  figures/

paper/
  1MIAC15___conference_version.pdf

docs/
  source_of_truth.md
```

## Notes

- Run `notebooks/paper_results.ipynb` to rebuild the paper-facing outputs.
- The notebook uses compact local modules instead of the old broad comparison
  runners.
- Only dataset 4 is retained from the transition-estimation inputs. The earlier
  observation-regime datasets are deliberately not copied into this clean scope.
- The transition-estimation section reports the full-precision EM fit. The
  state-estimation section intentionally uses the paper-encoded rounded TPM in
  `data/transition_matrices/right_censored_endpoint.csv`, because that is what
  reproduces the conference-version state-estimation table.
- Generated result tables and figures are written under `results/` when the
  notebook is executed.
