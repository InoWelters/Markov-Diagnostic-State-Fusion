# Paper Reproducible Results

This repository contains the notebook, data, and generated outputs used to
reproduce the study's results.

## Contents

```text
main.ipynb                  # executed paper-results notebook
data/                       # raw inputs, generated inputs, transition matrices
src/main/                   # local analysis modules used by the notebook
results/                    # generated tables and figures
paper/                      # manuscript PDF
```

## Reproduce

Open `main.ipynb` and run the cells from top to bottom. The notebook writes its
tables and figures into `results/`.

## Scope

The current workflow covers the conference-version result sections:

- right-censored endpoint transition-matrix estimation;
- diagnostic-only versus belief-updated state estimation;
- CBR0 and CBR1 maintenance-cost results.

Broader comparison results such as CBR1M, MLC, and the full three-TPM analysis
are retained as supplementary material.
