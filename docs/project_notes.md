# Assignment 2: End-to-end-NLP-System-Building

Project name: finance-rag-sec-10k

This repository contains our CS5170 Assignment 2 project on retrieval-augmented question answering over SEC 10-K filings.

The full end-to-end system was developed and tested primarily on Kaggle because the final setup used dense retrieval together with a local Mistral 7B reader, which required GPU resources that we did not have available for reliable local testing.

For that reason, the Kaggle version is the main tested runtime, while the local version in this repository is intended mainly for structure, configuration, and lightweight setup checks.
