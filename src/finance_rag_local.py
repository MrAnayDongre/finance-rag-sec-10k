"""
Assignment 2: End-to-end-NLP-System-Building
Project: finance-rag-sec-10k

Local version for Finance RAG over SEC 10-K Filings.

This file is a lightweight local entry point. It does not claim that the
full end-to-end system was tested locally. Instead, it helps verify local
paths, expected folder structure, and basic configuration.

The actual full tested runtime for this project was Kaggle.
"""

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "models"

STAFF_TEST_QUESTIONS_PATH = DATA_ROOT / "staff-test" / "questions.txt"
MISTRAL_MODEL_PATH = MODEL_ROOT / "mistral"
QWEN_MODEL_PATH = MODEL_ROOT / "qwen"

BID_NAME = "BID"
SUBMISSION_ROOT = OUTPUT_ROOT / BID_NAME

def describe_path(path: Path) -> str:
    return f"FOUND: {path}" if path.exists() else f"MISSING: {path}"

def ensure_structure() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    (SUBMISSION_ROOT / "data" / "train").mkdir(parents=True, exist_ok=True)
    (SUBMISSION_ROOT / "data" / "test").mkdir(parents=True, exist_ok=True)
    (SUBMISSION_ROOT / "system_outputs").mkdir(parents=True, exist_ok=True)

def main() -> None:
    ensure_structure()

    summary = {
        "assignment": "Assignment 2: End-to-end-NLP-System-Building",
        "project": "finance-rag-sec-10k",
        "mode": "local",
        "note": "This local file is for structure and path validation. The full tested runtime was Kaggle.",
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "output_root": str(OUTPUT_ROOT),
            "submission_root": str(SUBMISSION_ROOT),
            "staff_test_questions": str(STAFF_TEST_QUESTIONS_PATH),
            "mistral_model_path": str(MISTRAL_MODEL_PATH),
            "qwen_model_path": str(QWEN_MODEL_PATH)
        },
        "checks": {
            "staff_test_questions_exists": STAFF_TEST_QUESTIONS_PATH.exists(),
            "mistral_model_exists": MISTRAL_MODEL_PATH.exists(),
            "qwen_model_exists": QWEN_MODEL_PATH.exists()
        }
    }

    print("Assignment 2: End-to-end-NLP-System-Building")
    print("Project: finance-rag-sec-10k")
    print("Mode: local")
    print()
    print("This script is intended for local path validation.")
    print("The full end-to-end pipeline was tested in Kaggle.")
    print()
    print(describe_path(STAFF_TEST_QUESTIONS_PATH))
    print(describe_path(MISTRAL_MODEL_PATH))
    print(describe_path(QWEN_MODEL_PATH))
    print()
    print("Submission root:")
    print(SUBMISSION_ROOT)

    out_path = OUTPUT_ROOT / "local_environment_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Wrote local summary to: {out_path}")

if __name__ == "__main__":
    main()
