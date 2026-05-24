# Reference Audit

Date: 2026-05-24

## Local Consistency

- BibTeX entries in `paper/references.bib`: 26.
- Unique citation keys used in `paper/wrr_manuscript.tex`: 26.
- Missing BibTeX entries: none.
- Uncited BibTeX entries: none.

## Crossref DOI Check

All DOI-bearing entries resolved through Crossref and matched the intended title, journal, volume, and year or online/publication metadata. The audit prompted three cleanup edits:

- Added article number `W01202` to `Bijeljic2006`.
- Added issue `7` to `Dentz2016`.
- Added issue `3` and article number `22` to `Zhu2025`.

The `Dentz2023` entry is retained as 2023 because Springer lists the journal citation as `Transport in Porous Media 146, 5-53 (2023)`, although Crossref records an online publication date in 2022.

## Non-DOI Conference Entries

The following entries are conference proceedings or OpenReview/PMLR/NeurIPS entries with stable URLs rather than DOI fields:

- `Ho2020`
- `Li2021`
- `SanchezGonzalez2020`
- `Song2021`
- `Toshev2023`

These entries are cited in the machine-learning motivation paragraph and should remain in the bibliography unless the target journal requests DOI-only substitutions.
