# Optimization-project

Final project (Optimization for Data Science, Autumn 2026):
**Can a prior shrink the variance of EGGROLL?**
A low-rank prior, a sharp threshold, and why a better estimate need not train faster.

- `report/report.pdf` — the final report (5 pages, references, appendix). Source: `report/report.tex`, `report/appendix.tex`.
- `code/` — all experiments (numpy; scikit-learn for the data; matplotlib for the figures).
- `prior_notes/` — our earlier draft and notes that this project starts from.

## Reproduce

Each script writes a `results_*` file next to it; `figs.py` and `tables.py` then build every figure and table.

```
cd code
python3 exp_illustration.py    # Figure 1                         (~1 min)
python3 exp_risk.py            # Figure 2, Table 2                (~10 min)
python3 exp_recovery.py        # Table 3: one SVD vs recovery [3] (~5 min)
python3 exp_descent.py         # Figure 3a, Figure 4, Table 4     (~15 min)
python3 exp_nn_wide.py         # Figure 3b, Table 5               (~5 min)
python3 exp_nn_curvature.py    # Table 6                          (~1 min)
python3 exp_nn_wide_train.py   # Figure 3c, Table 7               (~1 h)
python3 exp_nn.py              # small network, finite differences
python3 check_fd.py            # finite differences vs exact slopes
python3 figs.py                # writes report/fig_*.pdf
python3 tables.py              # writes report/tab_*.tex
cd ../report && pdflatex report && pdflatex report
```

`code/shrink.py` holds the method: the ES estimate, Algorithm 1 (optimal singular-value shrinkage with
sign-flip calibration), two-level shrinkage, the error formula of Theorem 3, and the normalized
iterative hard thresholding used for the comparison with Allen et al. (2026).
