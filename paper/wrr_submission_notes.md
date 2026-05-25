# WRR-Style Submission Notes

Target journal family: AGU / Water Resources Research style.

Official AGU preparation points checked on May 17, 2026:

- Manuscript order should include title page, Key Points, Abstract, Plain
  Language Summary, Keywords, main text, Acknowledgments, Open Research,
  References, Tables, and Figures.
- Key Points: one to three complete statements; each no more than 140
  characters.
- Abstract: one paragraph, fewer than 250 words, no figure or table references.
- Plain Language Summary: optional but strongly encouraged for WRR-type
  submissions; no more than 200 words.
- Open Research section is required for data and software availability.

Current draft status:

- `wrr_manuscript.tex` includes three Key Points, all under 140 characters.
- Abstract is under 250 words.
- Plain Language Summary is under 200 words.
- `Open Research` section lists the GitHub repository
  `diogobolster/Training-Trajectories-Version-2`; large generated artifacts
  need a DOI-bearing review archive before WRR submission.
- The file compiles to `wrr_manuscript.pdf` using local `tectonic`.

Remaining WRR-oriented tightening:

- Replace placeholder author affiliations with confirmed affiliations and
  corresponding author details.
- Decide whether figures should stay embedded for review readability or be
  moved after the references to match AGU manuscript-order guidance exactly.
- Convert current PNG figure renderings into publication-quality vector or
  high-resolution raster figures.
- Archive code, processed trajectory data, OpenFOAM case files, and figure
  generation scripts in a DOI-bearing repository before submission/review.
- Check whether the official AGU LaTeX class should be used for final upload
  once the manuscript text stabilizes.

Useful AGU links:

- Text and graphics requirements:
  https://www.agu.org/publications/authors/journals/text-graphics-requirements
- Plain Language Summary guidance:
  https://www.agu.org/Publish-with-AGU/Publish/Author-Resources/Plain-Language-Summary
