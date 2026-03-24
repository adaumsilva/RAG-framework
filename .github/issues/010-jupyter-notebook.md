---
title: "Jupyter Notebook Example & Getting-Started Guide"
labels: ["good first issue", "documentation"]
---

## Description

Add a Jupyter notebook (`examples/basic_rag.ipynb`) that walks through the full pipeline step-by-step with prose explanations, and a `docs/getting-started.md` guide.

## Motivation

A notebook lowers the barrier to entry for data scientists and ML practitioners who prefer interactive environments over scripts. Good documentation is what turns visitors into contributors.

## Acceptance criteria

- [ ] `examples/basic_rag.ipynb` — notebook that runs end-to-end using built-in components (no API key needed)
- [ ] Each notebook section has a markdown cell explaining what the code does and why
- [ ] Notebook covers: install → load → chunk → embed → ingest → query → inspect results
- [ ] All cells execute without errors (`jupyter nbconvert --to notebook --execute examples/basic_rag.ipynb`)
- [ ] `docs/getting-started.md` with: installation options, quickstart code, explanation of each ABC, link to `CONTRIBUTING.md`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Files to touch

- `examples/basic_rag.ipynb` — new notebook
- `docs/getting-started.md` — new doc page
- `examples/README.md` — add notebook to the table

## Resources

- `examples/basic_rag.py` — the script version to convert to a notebook
- [nbformat docs](https://nbformat.readthedocs.io/) if creating the notebook programmatically
