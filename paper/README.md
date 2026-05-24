# Academic Paper Draft

This directory contains the first submission-style manuscript draft derived from
`docs/manuscript_v2_tightened.md`.

Files:

- `manuscript.tex`: LaTeX article draft with abstract, introduction, methods,
  results, discussion, conclusions, and figure/table structure.
- `wrr_manuscript.tex`: WRR/AGU-oriented version with Key Points, a concise
  abstract, Plain Language Summary, keywords, and Open Research section.
- `references.bib`: bibliography scaffold based on the current reference notes,
  expanded with recent 2020+ work on porous-media ML, pore-scale modeling,
  computational microfluidics, neural operators, diffusion models, and learned
  Lagrangian/particle simulators.
- `wrr_submission_notes.md`: AGU/WRR-specific requirements and remaining
  submission cleanup items.
- `figures/`: PNG/PDF renderings of the manuscript figures for LaTeX
  compilation.
- `../scripts/make_reader_friendly_figures.py`: generates the reader-facing
  workflow and mechanism-selection figures used in the WRR-oriented draft.
- `../scripts/make_transport_behavior_figures.py`: generates the held-out
  breakthrough, coverage, and dilution comparison figure from the outer-split
  benchmark summaries.
- `../scripts/render_svg_pngs.js`: renders SVG result figures to PNG with
  `sharp` so the manuscript uses the full figure canvas rather than cropped
  QuickLook thumbnails.

Build from this directory with:

```bash
tectonic manuscript.tex
```

or for the WRR-oriented draft:

```bash
tectonic wrr_manuscript.tex
```

The current draft is intentionally thesis-first. It argues that the strongest
AI-era update to the 2019 training-trajectory method is validation-driven
selection among physics kernels, learned transition rules, and mixtures, rather
than black-box replacement of the original physics-informed sampler.

The WRR draft now opens from the broad non-Fickian transport problem, introduces
an actual three-dimensional Bentheimer pore-space figure, and ends the
Introduction with three explicit questions. Methods are written as reproducible
procedural detail, and Results are organized as answers to those questions.
Figures now carry the primary evidence, with tables serving as numerical
support. A held-out behavior figure now shows breakthrough timing summaries,
crossing coverage, and dilution index so readers can see the transport
differences behind the scalar scores.
