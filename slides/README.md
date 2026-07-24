# Presentation: bias, ESS, and debiasing in MP-IS

A ~30 minute Beamer talk covering the empirical MP-IS-vs-HMC bias
investigation on `explore/mp-is-bias-empirical` (the Order-A/Order-B bug,
ESS, jackknife), with the newer BR-SNIS and Particle Gibbs/PGAS prototypes
from `explore/particle-mcmc` presented at the end as further work.

- `mp_is_debiasing_talk.tex` — the talk (Beamer, `metropolis` theme).
- `figures/` — plots generated from real experiment output (see below);
  committed as PDFs so the talk builds without re-running any experiments.
- `src/generate_figures.py` — regenerates `figures/*.pdf` from
  `src/results.json` (copied from `explore/mp-is-bias-empirical`'s
  `experiments/mp_is_bias/results.json`) plus numbers transcribed directly
  from `marginal_ess.py`, `jackknife_orderB.py`, and
  `experiments/particle_gibbs/pg_validate.py` output on the two experiment
  branches (`explore/mp-is-bias-empirical`, `explore/particle-mcmc`) — no
  fabricated numbers anywhere in the deck.

## Building

```
cd slides
pdflatex mp_is_debiasing_talk.tex
pdflatex mp_is_debiasing_talk.tex   # second pass for the outline/ToC
```

Needs a TeX distribution with the `beamer`, `metropolis` theme, and `tikz`
packages (Ubuntu: `texlive-latex-base texlive-latex-recommended
texlive-latex-extra texlive-fonts-recommended texlive-pictures`).

To regenerate the figures from scratch: `python3 src/generate_figures.py`
(needs `matplotlib`; `src/results.json` is already checked in, so this
doesn't require re-running the experiments).
