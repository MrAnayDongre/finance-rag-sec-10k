# Assignment 2: End-to-end-NLP-System-Building

## Project
**finance-rag-sec-10k**

This repository contains our CS5170 Assignment 2 project: an end-to-end retrieval-augmented question answering system for the finance domain using public SEC 10-K filings.

## Overview

The goal of this project is to answer factual finance questions using a retrieval-augmented generation pipeline built from public data only. The system is centered on SEC 10-K filings and is strengthened with additional public finance QA resources.

The pipeline includes:
- conservative QA generation from SEC 10-K filings
- external finance QA integration
- dense retrieval over an augmented knowledge base
- a closed-book TF-IDF baseline
- a local generative reader
- extractive heuristics and supported-span postprocessing

## Repository contents

- `src/finance_rag_local.py`  
  Local Python version of the project. This version is useful for path validation, structure checks, and local execution in environments where dependencies and model files are already available.

- `notebooks/finance_rag_kaggle.ipynb`  
  Kaggle notebook version of the project. This is the main tested runtime for the full end-to-end pipeline.

- `configs/paths_example.json`  
  Example path configuration for Kaggle and local environments.

- `requirements.txt`  
  Python dependencies used by the project.

- `outputs/expected_kaggle_run.txt`  
  A lightweight record of a representative successful Kaggle run.

## Problem setting

This project builds a finance-domain RAG system using public SEC 10-K filings as the main source of truth. The assignment required building an end-to-end NLP system from scratch, including:
- collecting or preparing raw data
- creating training and evaluation QA data
- building a retrieval-augmented system
- generating submission-ready outputs

Our chosen domain is finance, with an emphasis on factual company and filing information extracted from SEC documents.

## Data sources

This project uses only public data sources.

### Main source
- **PleIAs SEC parquet data**  
  Used to stream public SEC filings for years 2022, 2023, and 2024.

### External finance sources
- **ConvFinQA**  
  Used as external finance QA data to broaden question-answer coverage.

- **FinQA mirror corpus**  
  Used only as additional retrieval context in the knowledge base.

## System design

### 1. SEC QA generation
The pipeline first streams public SEC filings and extracts the most useful 10-K sections:
- Item 1
- Item 1A
- Item 7
- Item 7A
- Item 8

From these sections, the system generates conservative, high-precision QA pairs using:
- metadata templates
- regex-based extraction
- sentence-level templates

### 2. External finance integration
The system supplements SEC-generated QA pairs with:
- ConvFinQA question-answer examples
- FinQA mirror corpus contexts for retrieval only

This improves the retrieval base without shifting the project away from the SEC-centered domain.

### 3. Retrieval and answering
The final hybrid system combines:
- a **closed-book TF-IDF nearest-neighbor baseline**
- **dense retrieval** using `intfloat/e5-base-v2`
- **FAISS** for passage retrieval
- a **local Mistral 7B reader**
- question-specific extractive heuristics
- supported-span filtering on final answers

### 4. Final answer policy
At inference time, the system follows this logic:
1. metadata heuristics
2. extractive heuristics
3. closed-book fallback for confident cases
4. dense retrieval plus local reader otherwise
5. supported short-span postprocessing

## Execution environment

This project was developed and tested primarily on Kaggle.

The final hybrid system uses dense retrieval together with a local Mistral 7B reader, which required GPU resources that we did not have available for reliable local testing. For that reason, the full end-to-end pipeline was executed and validated in a Kaggle notebook environment with model files mounted through Kaggle input datasets.

The repository also includes a local Python version, but the full tested runtime for the project was Kaggle.

## Expected Kaggle paths

The Kaggle notebook assumes the following paths:

- Mistral model  
  `/kaggle/input/models/mistral-ai/mistral/pytorch/7b-instruct-v0.1-hf/1`

- Qwen model  
  `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1`

- Output package  
  `/kaggle/working/BID/`

- Output zip  
  `/kaggle/working/BID.zip`

## Expected local paths

The local script is organized around these optional paths:

- Output root  
  `./outputs`

- Optional staff test path  
  `./data/staff-test/questions.txt`

- Optional local model paths  
  `./models/mistral`  
  `./models/qwen`

## How the data is handled

The project does not store large generated datasets or submission artifacts in the repository.

### Automatically downloaded public data
When the pipeline runs, it fetches public data from:
- PleIAs SEC parquet data
- ConvFinQA
- FinQA mirror corpus

### Not automatically provided
These must already exist in the execution environment:
- local Mistral model files
- local Qwen model files

## How to run the project

### Option 1: Run in Kaggle
This is the recommended and fully tested path.

#### Steps
1. Open the Kaggle notebook:
   - `notebooks/finance_rag_kaggle.ipynb`

2. Make sure the required model inputs are available at:
   - `/kaggle/input/models/mistral-ai/mistral/pytorch/7b-instruct-v0.1-hf/1`
   - `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1`

3. Run the notebook end to end.

4. The generated submission package will be written to:
   - `/kaggle/working/BID/`

5. The zipped submission package will be written to:
   - `/kaggle/working/BID.zip`


### Option 2: Run locally
This is possible only if you already have the required dependencies, model files, and enough hardware.

#### Install dependencies

    pip install -r requirements.txt

#### Run the local script

    python src/finance_rag_local.py

#### Important note about local execution
The local script is included for compatibility and reproducibility, but the full end-to-end system was not primarily validated on local hardware. The strongest and most reliable project run was the Kaggle version.

## Output structure

When the pipeline runs successfully, it creates a submission package with the structure below:

    BID/
    ├── report.pdf
    ├── github_url.txt
    ├── contributions.md
    ├── data/
    │   ├── train/
    │   │   ├── questions.txt
    │   │   └── reference_answers.txt
    │   └── test/
    │       ├── questions.txt
    │       └── reference_answers.txt
    ├── system_outputs/
    │   └── system_output_1.txt
    └── README.md

Additional analysis artifacts are written under:
- `artifacts/`

## Representative run summary

A lightweight summary of a successful Kaggle run is included at:

- `outputs/expected_kaggle_run.txt`

This file is included only as a reference and is not itself part of the final submission package.

## Notes on reproducibility

- The system uses only public source data.
- The raw public datasets are downloaded automatically by the pipeline.
- Final outputs depend on access to the required model files and environment setup.
- Exact intermediate counts may vary slightly if sampling behavior or runtime conditions change.
- The official staff test file is not stored in this repository.

## Final submission notes

For the actual class submission, the following files must be present in the final package:
- `report.pdf`
- `github_url.txt`
- `contributions.md`
- `data/train/questions.txt`
- `data/train/reference_answers.txt`
- `data/test/questions.txt`
- `data/test/reference_answers.txt`
- `system_outputs/system_output_1.txt`
- `README.md`

## Team

- Anay Dongre
- Jay Kurade

