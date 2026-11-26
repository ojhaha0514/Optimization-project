# Optimization-project

Final project (Optimization for Data Science, Autumn 2026):
**Can a prior shrink the variance of EGGROLL? Spectral shrinkage of evolution-strategy gradients.**

- `report/report.pdf` — the final report (5 pages + appendix). Source: `report/report.tex`, `report/appendix.tex`.
- `code/` — all experiments (numpy only; scikit-learn for the data; matplotlib for figures).
- `prior_notes/` — our earlier draft and notes that this project starts from.

## Reproduce

```
cd code
python3 exp_risk.py            # Figure 1, Table 1   (~10 min)
python3 exp_descent.py         # Figure 2            (~5 min)
python3 exp_nn_wide.py         # Figure 3, Table 3   (~5 min)
python3 exp_nn.py              # small network, finite differences
python3 exp_nn_wide_train.py   # training comparison (slow, ~1 h)
python3 check_fd.py            # finite differences vs exact directional derivatives
python3 figs.py                # writes report/fig_*.pdf
cd ../report && pdflatex report && pdflatex report
```

`code/shrink.py` holds the method: ES estimate, optimal singular-value shrinkage, two-level shrinkage,
sign-flip calibration, and the risk formula of Theorem 3.
