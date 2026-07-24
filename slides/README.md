# Presentation: TMC, MP-RWS, MPIW, the source term trick, and QEM

A ~30 minute Beamer talk on how `alan` actually does inference, five pillars
in sequence:

1. **Tensor Monte Carlo** (Aitchison, 2019) — the $K$-per-latent,
   plate-contraction trick.
2. **Massively-Parallel RWS** (Heap et al., UAI 2023) — turns that estimator
   into a wake/sleep training loop, proves $\hat P_{\text{MP}}(z)$ unbiased.
3. **MPIW** (Bowyer, Heap & Aitchison, UAI 2024) — reframes the same
   estimator as a general-purpose posterior-expectation tool (global vs.
   massively-parallel importance weighting), independent of the RWS
   training loop.
4. **The source term trick** (same paper) — gets posterior
   samples/marginals/moments out of the same computation via autodiff,
   instead of hand-written backward traversals.
5. **QEM** (Heap, Bowyer & Aitchison, 2025) — uses MPIW + the source term
   trick as a gradient-free E-step (with a closed-form, exponential-family
   M-step) to train $Q$ faster than MP-VI/MP-RWS and, provably,
   reparameterization-invariantly.

The empirical follow-up work (MP-IS-vs-HMC bias, the Order-A/Order-B bug,
ESS, jackknife, and the BR-SNIS/Particle Gibbs prototypes) is condensed into
four slides at the end, not the main content.

- `tmc_mprws_source_term_talk.tex` — the talk (Beamer, `metropolis` theme).
- `figures/bias_vs_K.pdf` — the one empirical plot used (from the
  MP-IS-vs-HMC bias sweep); the other three plots from the earlier full
  empirical deck aren't used here but are left in `figures/` in case a
  longer version of the talk wants them back.
- `src/generate_figures.py` / `src/results.json` — regenerates the figures
  from real experiment output (see the file for provenance details).

No fabricated numbers anywhere in the deck. The QEM section (algorithm,
results, reparameterization-invariance claim) is grounded directly in
arXiv:2503.08264 ("Massively Parallel Expectation Maximization For
Approximate Posteriors", Heap, Bowyer & Aitchison, 2025); the MPIW and
source-term-trick sections in arXiv:2310.17374; the TMC/MP-RWS sections in
the UAI'23 MP-RWS paper plus `alan`'s own source
(`src/alan/Sample.py`'s `J_tensor`/`grad()` mechanism for the source term
trick in particular).

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
