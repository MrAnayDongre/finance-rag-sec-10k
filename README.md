# Assignment 2: End-to-end-NLP-System-Building

## Project
finance-rag-sec-10k

This repository contains our CS5170 Assignment 2 project: an end-to-end retrieval-augmented question answering system for the finance domain using public SEC 10-K filings.

## Files

- `src/finance_rag_local.py`
  Local version for path checks, structure validation, and lightweight setup checks.

- `notebooks/finance_rag_kaggle.ipynb`
  Kaggle notebook file for the main tested runtime.

## Summary

The system answers factual finance questions by combining:
- conservative QA generation from SEC filings
- dense retrieval over an augmented knowledge base
- a closed-book TF-IDF baseline
- a local Mistral 7B reader on Kaggle
- extractive heuristics and supported-span postprocessing

## Data sources

- PleIAs SEC parquet data
- ConvFinQA
- FinQA mirror corpus

## Execution environment

This project was developed and tested primarily on Kaggle.

The final hybrid system uses dense retrieval together with a local Mistral 7B reader, which required GPU resources that we did not have available for reliable local testing. For that reason, the full end-to-end pipeline was executed in a Kaggle notebook environment with model files mounted through Kaggle input datasets.

The code may be adapted to run outside Kaggle, but the provided paths and tested setup assume Kaggle.

## Expected Kaggle paths

- Staff test questions:
  `/kaggle/input/staff-test/questions.txt`

- Mistral model:
  `/kaggle/input/models/mistral-ai/mistral/pytorch/7b-instruct-v0.1-hf/1`

- Qwen model:
  `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1`

- Output package:
  `/kaggle/working/BID/`

- Output zip:
  `/kaggle/working/BID.zip`

## Expected local paths

- Local output root:
  `./outputs`

- Optional local staff test path:
  `./data/staff-test/questions.txt`

- Optional local model paths:
  `./models/mistral`
  `./models/qwen`

## Notes

This repository is for Assignment 2: End-to-end-NLP-System-Building.

The GitHub repository is intended for code transparency, documentation, and reproducibility. The full tested runtime was Kaggle, not local hardware.
