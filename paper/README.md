# Working paper (English draft)

`main.tex` is the English working-paper draft on the stage F experiment. It includes the three exploratory robustness supplements of contract Section 20.21 (mean reversion, a fine for every false report, firm-specific washing costs), which finished on 18 October 2026. Red bracketed text in the PDF marks the only remaining placeholder: the author line (name, affiliation and e-mail), which the author fills in before circulating.

Build from this directory:

```sh
python make_figures.py
latexmk -pdf main.tex
```

`make_figures.py` reads only committed analysis files under `reports/`. Every entry in `references.bib` was checked on 20 October 2026 against the publisher page, its DOI record at Crossref or the official text; the commit message of that check lists the source of each entry.
