# Presentation: TMC, MP-RWS, MP-IW, and the source term trick

A ~30 minute Beamer talk on how `alan` actually does inference: Tensor Monte
Carlo (Aitchison 2019) as the precursor, Massively-Parallel Reweighted
Wake-Sleep (Heap et al., UAI 2023) as the training algorithm built on top of
it, the resulting MP-IW bound, and the source term trick (Bowyer, Heap &
Aitchison, UAI 2024) for getting posterior samples/marginals/moments out of
the same computation via autodiff. The empirical follow-up work (MP-IS-vs-HMC
bias, the Order-A/Order-B bug, ESS, jackknife, and the BR-SNIS/Particle
Gibbs prototypes) is condensed into four slides at the end, not the main
content.

- `tmc_mprws_source_term_talk.tex` — the talk (Beamer, `metropolis` theme).
- `figures/bias_vs_K.pdf` — the one empirical plot used (from the
  MP-IS-vs-HMC bias sweep); the other three plots from the earlier full
  empirical deck aren't used here but are left in `figures/` in case a
  longer version of the talk wants them back.
- `src/generate_figures.py` / `src/results.json` — regenerates the figures
  from real experiment output (see the file for provenance details). No
  fabricated numbers anywhere in the deck; the algorithmic content (TMC /
  MP-RWS / MP-IW / source term trick) is grounded directly in `alan`'s own
  source (`src/alan/Sample.py`'s `J_tensor`/`grad()` mechanism for the
  source term trick in particular) plus the two cited papers.

## Building

```
cd slides
pdflatex tmc_mprws_source_term_talk.tex
pdflatex tmc_mprws_source_term_talk.tex   # second pass for the outline/ToC
```

Needs a TeX distribution with the `beamer`, `metropolis` theme, and `tikz`
packages (Ubuntu: `texlive-latex-base texlive-latex-recommended
texlive-latex-extra texlive-fonts-recommended texlive-pictures`).

To regenerate the figures from scratch: `python3 src/generate_figures.py`
(needs `matplotlib`; `src/results.json` is already checked in, so this
doesn't require re-running the experiments).
