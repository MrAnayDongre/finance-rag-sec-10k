import os
import re
import gc
import json
import random
import shutil
import warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# ============================================================
# Configuration
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

RUNNING_ON_KAGGLE = os.path.exists("/kaggle")
BID_NAME = "BID"

if RUNNING_ON_KAGGLE:
    WORK_DIR = Path("/kaggle/working")
    STAFF_TEST_QUESTIONS_PATH = "/kaggle/input/staff-test/questions.txt"
    MISTRAL_MODEL_PATH = "/kaggle/input/models/mistral-ai/mistral/pytorch/7b-instruct-v0.1-hf/1"
    QWEN_MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1"
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
    WORK_DIR = PROJECT_ROOT / "outputs"
    STAFF_TEST_QUESTIONS_PATH = str(PROJECT_ROOT / "data" / "staff-test" / "questions.txt")
    MISTRAL_MODEL_PATH = str(PROJECT_ROOT / "models" / "mistral")
    QWEN_MODEL_PATH = str(PROJECT_ROOT / "models" / "qwen")

SUBMISSION_ROOT = WORK_DIR / BID_NAME
ARTIFACTS_DIR = SUBMISSION_ROOT / "artifacts"
TRAIN_DIR = SUBMISSION_ROOT / "data" / "train"
TEST_DIR = SUBMISSION_ROOT / "data" / "test"
SYSTEM_OUTPUTS_DIR = SUBMISSION_ROOT / "system_outputs"

for path in [SUBMISSION_ROOT, ARTIFACTS_DIR, TRAIN_DIR, TEST_DIR, SYSTEM_OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

YEARS = [2024, 2023, 2022]
PARQUET_URLS = [
    f"https://huggingface.co/datasets/PleIAs/SEC/resolve/main/{year}.parquet"
    for year in YEARS
]

CONVFINQA_DATASET_ID = "AdaptLLM/ConvFinQA"
FINQA_MIRROR_ID = "embedding-benchmark/FinQA"

LOCAL_MODEL_TYPE = "mistral"

GITHUB_URL_VALUE = "REPLACE_WITH_YOUR_GITHUB_REPO_URL"
CONTRIBUTIONS_MD_VALUE = """# Contributions

## Team members
- Anay Dongre
- Jay Kurade

## Data annotation contributions
- Anay Dongre: Designed and implemented the SEC QA generation pipeline, including metadata QA generation, regex-based extraction, sentence-template QA creation, filtering, and deduplication.
- Jay Kurade: Reviewed generated QA examples, checked quality and consistency of the dataset, and helped validate whether the generated questions and answers were suitable for development and evaluation.

## Data collection, preprocessing, modeling, and evaluation
- Anay Dongre: Selected the finance domain and public SEC 10-K filings as the main source. Implemented the full end-to-end pipeline, including SEC section extraction, ConvFinQA integration, FinQA mirror corpus integration, augmented knowledge base construction, closed-book TF-IDF baseline, dense retrieval with E5-base-v2, FAISS indexing, local Mistral reader integration, hybrid answer selection logic, evaluation pipeline, and submission package generation.
- Jay Kurade: Helped with project planning, experiment discussion, result interpretation, error analysis, and documentation of the system design and findings in the final report.

## Report writing contributions
- Jay Kurade: Primary author of the final report.pdf.
- Anay Dongre: Contributed technical details, implementation notes, experiment results, and system outputs used in the report.

## Final note
Both members discussed the project design, reviewed intermediate outputs, and contributed to finalizing the submission.
"""

MAX_FILINGS = 40
MIN_FILING_WORDS = 2500
MAX_STREAM_BUFFER = 5000

TARGET_SECTIONS = ["ITEM_1", "ITEM_1A", "ITEM_7", "ITEM_7A", "ITEM_8"]
MIN_SECTION_WORDS = 120

CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
MAX_CHUNKS_PER_SECTION = 2
MAX_CHUNKS_TOTAL = 300

DEV_RATIO = 0.10
MAX_ANSWER_WORDS = 18

EXTERNAL_CONTEXT_CHUNK_WORDS = 160
EXTERNAL_CONTEXT_OVERLAP = 30
MAX_CONVFINQA_ROWS = 600
MAX_FINQA_CORPUS_ROWS = 1500
MAX_EXTERNAL_CHUNKS = 5000
MIN_KB_CHUNK_WORDS = 8

DENSE_MODEL_NAME = "intfloat/e5-base-v2"
DENSE_TOPK = 10
FINAL_CONTEXT_TOPK = 5
MAX_CONTEXT_CHARS = 2600
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.0
USE_4BIT = True

CLOSED_BOOK_SIM_THRESHOLD = 0.72

SECTION_PRIORITY = {
    "ITEM_1": 0,
    "ITEM_1A": 1,
    "ITEM_7": 2,
    "ITEM_7A": 3,
    "ITEM_8": 4,
}

BOUNDARY_REGEX = re.compile(
    r"(?i)\bitem\s+(1a|1b|1|2|3|4|5|6|7a|7|8|9a|9b|9|10|11|12|13|14|15)\b[\s\.\-:]*"
)

LABEL_MAP = {
    "1": "ITEM_1",
    "1A": "ITEM_1A",
    "1B": "ITEM_1B",
    "2": "ITEM_2",
    "3": "ITEM_3",
    "4": "ITEM_4",
    "5": "ITEM_5",
    "6": "ITEM_6",
    "7": "ITEM_7",
    "7A": "ITEM_7A",
    "8": "ITEM_8",
    "9": "ITEM_9",
    "9A": "ITEM_9A",
    "9B": "ITEM_9B",
    "10": "ITEM_10",
    "11": "ITEM_11",
    "12": "ITEM_12",
    "13": "ITEM_13",
    "14": "ITEM_14",
    "15": "ITEM_15",
}

EXPECTED_QA_COLUMNS = [
    "question", "answer", "evidence", "context", "doc_id",
    "year", "cik", "filename", "section", "source_type"
]

FINAL_COLUMNS = [
    "question", "answer", "context", "source_dataset", "source_split",
    "source_type", "doc_id", "year", "cik", "filename", "section"
]

MONTHS = {
    "january","february","march","april","may","june",
    "july","august","september","october","november","december"
}

US_STATES = {
    "delaware","california","minnesota","texas","new york","florida","arizona",
    "georgia","illinois","massachusetts","nevada","new jersey","maryland",
    "virginia","washington","colorado","ohio","indiana","michigan","pennsylvania",
    "north carolina","south carolina","tennessee","utah","wisconsin","oregon",
    "missouri","connecticut","alabama","kansas","oklahoma","louisiana","kentucky",
    "maine","iowa","idaho","montana","nebraska","new mexico","arkansas","wyoming",
    "district of columbia"
}

COUNTRY_HINTS = {
    "china","canada","bermuda","cayman islands","british virgin islands",
    "england","ireland","israel","singapore","egypt","japan","india","france",
    "germany","switzerland","netherlands","australia","mexico","luxembourg",
    "hong kong"
}

# ============================================================
# Helpers
# ============================================================
def clean_text(x):
    if x is None:
        return ""
    x = str(x)
    x = re.sub(r"<[^>]+>", " ", x)
    x = x.replace("\\xa0", " ").replace("\xa0", " ")
    x = x.replace("&nbsp;", " ").replace("&amp;", "&")
    x = x.replace("\\n", " ").replace("\n", " ")
    x = x.replace("\\t", " ").replace("\t", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def word_count(text):
    return len(clean_text(text).split())

def normalize_question(text):
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_text(text):
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def contains_span(context, span):
    return normalize_text(span) in normalize_text(context)

def ensure_columns(df, cols):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def ensure_dataframe(df, cols):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    return ensure_columns(df, cols)[cols].copy()

def write_txt(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(clean_text(line) + "\n")

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def split_into_chunks(text, chunk_words=220, overlap_words=40):
    words = clean_text(text).split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks

def sentence_split(text):
    text = clean_text(text)
    if not text:
        return []
    sents = re.split(r"(?<=[\.\?\!;])\s+(?=[A-Z0-9])", text)
    sents = [clean_text(s) for s in sents]
    return [s for s in sents if 6 <= word_count(s) <= 40]

def safe_display(df, n=10):
    print(df.head(n).to_string(index=False))

# ============================================================
# Answer filtering
# ============================================================
def bad_answer(answer):
    answer = clean_text(answer)
    if not answer:
        return True
    if word_count(answer) > MAX_ANSWER_WORDS:
        return True

    lower = answer.lower()

    if lower in {
        "our business", "facilities", "investments", "operations",
        "competition", "customers", "products", "services", "markets",
        "yes", "no", "unknown", "none", "n/a", "na"
    }:
        return True

    if lower in MONTHS:
        return True

    if lower.startswith((
        "and ", "or ", "with ", "for ", "to ", "of ", "in ",
        "from ", "that ", "which ", "including ", "during "
    )):
        return True

    if re.search(r"\b(is|are|was|were|be|been)\b", lower):
        return True

    if len(answer) <= 1:
        return True

    return False

def add_qa(qas, question, answer, evidence, context, row, section, source_type):
    question = clean_text(question)
    answer = clean_text(answer).strip(" ,.;:-")
    evidence = clean_text(evidence).strip(" ,.;:-")
    context = clean_text(context)

    if not question or not answer or not context:
        return
    if not question.endswith("?"):
        question += "?"
    if bad_answer(answer):
        return
    if not contains_span(context, answer):
        return
    if evidence and not contains_span(context, evidence):
        return

    qas.append({
        "question": question,
        "answer": answer,
        "evidence": evidence if evidence else answer,
        "context": context,
        "doc_id": clean_text(row.get("doc_id")),
        "year": clean_text(row.get("year")),
        "cik": clean_text(row.get("cik")),
        "filename": clean_text(row.get("filename")),
        "section": clean_text(section),
        "source_type": clean_text(source_type),
    })

def deduplicate_qas(df):
    df = df.copy()
    df["q_norm"] = df["question"].apply(normalize_question)
    df["a_norm"] = df["answer"].apply(normalize_text)
    df = df.drop_duplicates(subset=["q_norm", "a_norm"]).drop(columns=["q_norm", "a_norm"])
    return df.reset_index(drop=True)

def clean_final_df(df):
    df = ensure_columns(df, FINAL_COLUMNS).copy()

    for c in FINAL_COLUMNS:
        df[c] = df[c].apply(clean_text)

    df["question"] = df["question"].apply(lambda x: x if x.endswith("?") else (x + "?" if x else ""))

    df = df[
        (df["question"].apply(lambda x: 3 <= word_count(x) <= 40)) &
        (df["answer"].apply(lambda x: 1 <= word_count(x) <= 25)) &
        (df["context"].apply(lambda x: word_count(x) >= 5))
    ].copy()

    df = df[~df["answer"].apply(bad_answer)].copy()

    df["q_norm"] = df["question"].apply(normalize_question)
    df["a_norm"] = df["answer"].apply(normalize_text)
    df["c_norm"] = df["context"].apply(lambda x: clean_text(x)[:300].lower())

    df = df.drop_duplicates(subset=["q_norm", "a_norm", "c_norm"]).reset_index(drop=True)
    return df.drop(columns=["q_norm", "a_norm", "c_norm"])

# ============================================================
# SEC parsing and QA generation
# ============================================================
def extract_sections_from_filing(text):
    text = clean_text(text)
    if not text:
        return {}

    matches = []
    for m in BOUNDARY_REGEX.finditer(text):
        raw_label = m.group(1).upper()
        norm_label = LABEL_MAP.get(raw_label)
        if norm_label:
            matches.append((m.start(), norm_label))

    if not matches:
        return {}

    matches = sorted(matches, key=lambda x: x[0])
    segments = []

    for i, (start, label) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        seg = clean_text(text[start:next_start])
        wc = word_count(seg)
        segments.append({"section": label, "text": seg, "word_count": wc})

    best = {}
    for target in TARGET_SECTIONS:
        candidates = [s for s in segments if s["section"] == target and s["word_count"] >= MIN_SECTION_WORDS]
        if candidates:
            best[target] = max(candidates, key=lambda x: x["word_count"])["text"]
    return best

def filing_has_useful_sections(text):
    return len(extract_sections_from_filing(text)) >= 2

def load_sec_stream():
    ds = load_dataset(
        "parquet",
        data_files={"train": PARQUET_URLS},
        split="train",
        streaming=True,
    )
    return ds.shuffle(seed=SEED, buffer_size=MAX_STREAM_BUFFER)

def sample_filings(stream, max_filings=40):
    selected = []
    seen_ids = set()

    for row in tqdm(stream, desc="Sampling filings"):
        text = clean_text(row.get("text"))
        doc_id = clean_text(row.get("id"))
        if not doc_id or doc_id in seen_ids:
            continue
        if word_count(text) < MIN_FILING_WORDS:
            continue
        if not filing_has_useful_sections(text):
            continue

        selected.append({
            "doc_id": doc_id,
            "year": clean_text(row.get("year")),
            "cik": clean_text(row.get("cik")),
            "filename": clean_text(row.get("filename")),
            "text": text,
        })
        seen_ids.add(doc_id)

        if len(selected) >= max_filings:
            break

    if not selected:
        raise RuntimeError("No usable filings found.")
    return pd.DataFrame(selected)

def generate_metadata_qas(row):
    qas = []
    doc_id = clean_text(row.get("doc_id"))
    year = clean_text(row.get("year"))
    cik = clean_text(row.get("cik"))
    filename = clean_text(row.get("filename"))
    context = f"Filing id: {doc_id}. Filing year: {year}. CIK: {cik}. Filename: {filename}."

    if doc_id and year:
        add_qa(qas, f"What filing year is associated with filing {doc_id}", year, year, context, row, "METADATA", "template_metadata")
    if doc_id and cik:
        add_qa(qas, f"What is the SEC CIK for filing {doc_id}", cik, cik, context, row, "METADATA", "template_metadata")

    return qas

def clean_location_answer(ans):
    ans = clean_text(ans)
    return re.sub(r"\b(in|on|at)\s+$", "", ans, flags=re.I).strip(" ,.;:-")

def is_valid_incorporation_answer(ans):
    lower = clean_text(ans).lower()
    if lower in MONTHS or len(lower) < 3:
        return False
    if "which " in lower or "manufactures" in lower or "business" in lower:
        return False
    if lower.endswith(" in"):
        return False
    return lower in US_STATES or lower in COUNTRY_HINTS or len(lower.split()) <= 4

def is_valid_address_answer(ans):
    ans = clean_text(ans)
    return len(ans) >= 12 and not re.search(r",\s*[A-Z]$", ans)

def regex_extract_qas_from_section(row, section_name, section_text):
    qas = []
    context = clean_text(section_text)

    for m in re.finditer(r"(?i)\bheadquartered in ([A-Z][A-Za-z0-9,\-\s]{2,60}?)(?:\.|,| and\b)", context):
        ans = clean_location_answer(m.group(1))
        if not bad_answer(ans):
            add_qa(qas, "Where is the company headquartered", ans, ans, context, row, section_name, "regex_template")

    for m in re.finditer(r"(?i)\bour principal executive offices are located at ([^.]{12,140})\.", context):
        ans = clean_text(m.group(1))
        if is_valid_address_answer(ans):
            add_qa(qas, "Where are the principal executive offices located", ans, ans, context, row, section_name, "regex_template")

    incorp_patterns = [
        r"(?i)\bincorporated under the laws of ([A-Z][A-Za-z\s]{2,30}?)(?:,|\.|\s+on\s|\s+in\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b)",
        r"(?i)\bincorporated in the state of ([A-Z][A-Za-z\s]{2,30}?)(?:,|\.|\s+on\s|\s+in\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b)",
        r"(?i)\bincorporated in ([A-Z][A-Za-z\s]{2,30}?)(?:,|\.|\s+on\s|\s+in\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b)"
    ]
    for pat in incorp_patterns:
        for m in re.finditer(pat, context):
            ans = clean_text(m.group(1))
            if is_valid_incorporation_answer(ans):
                add_qa(qas, "Where is the company incorporated", ans, ans, context, row, section_name, "regex_template")

    for m in re.finditer(r"(?i)\b(?:we had|the company had|had)\s+(?:approximately\s+)?([\d,]+)\s+employees\b", context):
        ans = clean_text(m.group(1))
        add_qa(qas, "How many employees did the company have", ans, ans, context, row, section_name, "regex_template")

    if section_name == "ITEM_1A":
        for m in re.finditer(r"(?i)\brisks related to ([^.]{6,80})\.", context):
            ans = clean_text(m.group(1))
            if not bad_answer(ans):
                add_qa(qas, "What risk does the filing mention", ans, ans, context, row, section_name, "regex_template")

        for m in re.finditer(r"(?i)\bsubject to risks associated with ([^.]{6,80})\.", context):
            ans = clean_text(m.group(1))
            if not bad_answer(ans):
                add_qa(qas, "What risk is the company subject to", ans, ans, context, row, section_name, "regex_template")

    auditor_patterns = [
        r"(?i)\b([A-Z][A-Za-z&,\.\s]{3,60}LLP)\s+(?:has|have)\s+audited",
        r"(?i)\b([A-Z][A-Za-z&,\.\s]{3,60}LLP)\s+served as our independent registered public accounting firm",
        r"(?i)\bindependent registered public accounting firm(?: was| is)?\s+([A-Z][A-Za-z&,\.\s]{3,60}LLP)",
    ]
    for pat in auditor_patterns:
        for m in re.finditer(pat, context):
            ans = clean_text(m.group(1))
            if not bad_answer(ans):
                add_qa(qas, "Who is the independent registered public accounting firm", ans, ans, context, row, section_name, "regex_template")

    return qas

def sentence_template_qas(row, section_name, section_text, max_qas_per_section=4):
    qas = []
    added = 0

    for sent in sentence_split(section_text):
        if added >= max_qas_per_section:
            break

        m = re.search(r"(?i)\b(?:approximately\s+)?([\d,]+)\s+employees\b", sent)
        if m:
            add_qa(qas, "How many employees are mentioned", clean_text(m.group(1)), clean_text(m.group(1)), sent, row, section_name, "sentence_template")
            added += 1
            continue

        m = re.search(r"(?i)\bheadquartered in ([A-Z][A-Za-z0-9,\-\s]{2,60})", sent)
        if m:
            ans = clean_location_answer(m.group(1))
            if not bad_answer(ans):
                add_qa(qas, "Where is the company headquartered", ans, ans, sent, row, section_name, "sentence_template")
                added += 1
                continue

        m = re.search(r"(?i)\bprincipal executive offices are located at ([^.]{12,140})\.", sent)
        if m:
            ans = clean_text(m.group(1))
            if is_valid_address_answer(ans):
                add_qa(qas, "Where are the principal executive offices located", ans, ans, sent, row, section_name, "sentence_template")
                added += 1
                continue

        if section_name == "ITEM_1A":
            m = re.search(r"(?i)\b(?:could|may)\s+adversely affect\s+([^.]{6,80})\.", sent)
            if m:
                ans = clean_text(m.group(1))
                if not bad_answer(ans):
                    add_qa(qas, "What could be adversely affected", ans, ans, sent, row, section_name, "sentence_template")
                    added += 1

    return qas

# ============================================================
# External datasets
# ============================================================
def take_first_nonempty(ex, keys):
    for key in keys:
        if key in ex and ex[key] not in [None, "", [], {}]:
            return ex[key]
    return ""

def stringify_table_like(x, limit_items=80):
    parts = []

    def walk(obj):
        if len(parts) >= limit_items or obj is None:
            return
        if isinstance(obj, (str, int, float, np.integer, np.floating)):
            txt = clean_text(obj)
            if txt:
                parts.append(txt)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
                if len(parts) >= limit_items:
                    break
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
                if len(parts) >= limit_items:
                    break

    walk(x)
    return " ".join(parts[:limit_items]).strip()

def load_json_any(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ["data", "train", "examples", "records"]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
            return [obj]
    except Exception:
        pass

    return [json.loads(line) for line in raw.splitlines() if line.strip()]

def flatten_dialogue_history(history):
    if history is None:
        return ""

    parts = []
    if isinstance(history, list):
        for item in history:
            if isinstance(item, str):
                parts.append(clean_text(item))
            elif isinstance(item, dict):
                for key in ["user", "assistant", "question", "answer", "text"]:
                    if key in item and item[key] not in [None, ""]:
                        parts.append(clean_text(item[key]))
    elif isinstance(history, dict):
        for v in history.values():
            if v not in [None, "", [], {}]:
                parts.append(clean_text(v))
    elif isinstance(history, str):
        parts.append(clean_text(history))

    return " ".join([p for p in parts if p])

def extract_convfinqa_qa(ex):
    question = clean_text(take_first_nonempty(ex, ["question", "query"]))
    answer = clean_text(take_first_nonempty(ex, ["answer", "gold_answer", "exe_ans"]))

    qa = ex.get("qa")
    if (not question or not answer) and qa not in [None, ""]:
        if isinstance(qa, dict):
            if not question:
                question = clean_text(take_first_nonempty(qa, ["question", "query", "user_question"]))
            if not answer:
                answer = clean_text(take_first_nonempty(qa, ["answer", "gold_answer", "exe_ans"]))
        elif isinstance(qa, list) and len(qa) > 0:
            first = qa[0]
            if isinstance(first, dict):
                if not question:
                    question = clean_text(take_first_nonempty(first, ["question", "query", "user_question"]))
                if not answer:
                    answer = clean_text(take_first_nonempty(first, ["answer", "gold_answer", "exe_ans"]))
            elif isinstance(first, str) and first.strip().endswith("?") and not question:
                question = clean_text(first)
        elif isinstance(qa, str) and qa.strip().endswith("?") and not question:
            question = clean_text(qa)

    return question, answer

def normalize_convfinqa_example(ex):
    question, answer = extract_convfinqa_qa(ex)
    context_parts = []

    for key in ["pre_text", "post_text"]:
        if isinstance(ex.get(key), list):
            context_parts.extend([clean_text(x) for x in ex[key] if clean_text(x)])

    for key in ["annotation", "text", "context", "report"]:
        if ex.get(key):
            context_parts.append(clean_text(ex[key]))

    for key in ["conversation", "dialogue", "history"]:
        if ex.get(key):
            hist = flatten_dialogue_history(ex[key])
            if hist:
                context_parts.append(hist)

    if ex.get("table") not in [None, ""]:
        context_parts.append(stringify_table_like(ex["table"], limit_items=60))

    doc_id = ""
    for key in ["id", "uid", "doc_id", "filename"]:
        if ex.get(key) not in [None, ""]:
            doc_id = clean_text(ex[key])
            break

    return {
        "question": clean_text(question),
        "answer": clean_text(answer),
        "context": clean_text(" ".join(context_parts)),
        "source_dataset": "convfinqa",
        "source_split": "train",
        "source_type": "external_convfinqa",
        "doc_id": doc_id,
        "year": "",
        "cik": "",
        "filename": clean_text(ex.get("filename", "")),
        "section": "",
    }

def load_convfinqa_raw_records():
    for fname in ["train_turn.json", "data/train_turn.json", "train.json", "data/train.json"]:
        try:
            local_path = hf_hub_download(repo_id=CONVFINQA_DATASET_ID, repo_type="dataset", filename=fname)
            rows = load_json_any(local_path)
            if rows:
                print(f"Loaded ConvFinQA from: {fname}")
                return rows
        except Exception:
            continue
    raise RuntimeError("Could not locate ConvFinQA raw JSON.")

def load_and_normalize_convfinqa(max_rows=600):
    rows = load_convfinqa_raw_records()
    conv = pd.DataFrame([normalize_convfinqa_example(ex) for ex in rows if isinstance(ex, dict)])
    conv = clean_final_df(conv)
    if len(conv) > max_rows:
        conv = conv.sample(max_rows, random_state=SEED).reset_index(drop=True)
    return conv

def load_finqa_corpus_contexts(max_rows=1500):
    corpus_df = pd.DataFrame(load_dataset(FINQA_MIRROR_ID, "corpus", split="corpus"))
    print("FinQA corpus columns:", list(corpus_df.columns))

    text_col = next((c for c in ["text", "contents", "content", "document", "corpus", "passage"] if c in corpus_df.columns), None)
    if text_col is None:
        raise RuntimeError(f"Could not detect FinQA text column. Columns: {list(corpus_df.columns)}")

    id_col = next((c for c in ["corpus-id", "corpus_id", "_id", "id", "doc_id", "document_id", "pid", "passage_id"] if c in corpus_df.columns), None)
    if id_col is None:
        corpus_df = corpus_df.reset_index().rename(columns={"index": "doc_id_generated"})
        id_col = "doc_id_generated"

    corpus_df[id_col] = corpus_df[id_col].astype(str)
    corpus_df[text_col] = corpus_df[text_col].apply(clean_text)
    corpus_df = corpus_df[corpus_df[text_col].apply(lambda x: word_count(x) >= 10)].copy()

    if len(corpus_df) > max_rows:
        corpus_df = corpus_df.sample(max_rows, random_state=SEED).reset_index(drop=True)

    rows = [{
        "doc_id": clean_text(r[id_col]),
        "context": clean_text(r[text_col]),
        "source_dataset": "finqa_corpus",
        "source_split": "corpus",
        "source_type": "external_finqa_corpus",
        "question": "",
        "answer": "",
        "year": "",
        "cik": "",
        "filename": "",
        "section": "",
    } for _, r in corpus_df.iterrows()]

    out_df = pd.DataFrame(rows)
    if len(out_df) == 0:
        raise RuntimeError("FinQA corpus loader returned 0 rows.")
    return out_df

def load_and_normalize_sec_train(sec_train_path):
    sec = pd.read_csv(sec_train_path)
    if "context" not in sec.columns:
        sec["context"] = ""
    sec["source_dataset"] = "sec_generated"
    sec["source_split"] = "train"
    if "source_type" not in sec.columns:
        sec["source_type"] = "sec_generated"
    return clean_final_df(ensure_columns(sec, FINAL_COLUMNS))

# ============================================================
# Model helpers
# ============================================================
def choose_local_model_path():
    if LOCAL_MODEL_TYPE.lower() == "mistral":
        if os.path.exists(MISTRAL_MODEL_PATH):
            return "mistral", MISTRAL_MODEL_PATH
        return "qwen", QWEN_MODEL_PATH
    if os.path.exists(QWEN_MODEL_PATH):
        return "qwen", QWEN_MODEL_PATH
    return "mistral", MISTRAL_MODEL_PATH

def load_local_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quant_config = None
    if USE_4BIT and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    if quant_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )

    model.eval()
    return tokenizer, model

def get_model_input_device(model):
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for dev in model.hf_device_map.values():
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    if hasattr(model, "device") and model.device is not None:
        return model.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")

def build_prompt(question, contexts, tokenizer, model_kind="mistral"):
    merged = ""
    for c in contexts[:FINAL_CONTEXT_TOPK]:
        c = clean_text(c)
        if not c:
            continue
        if len(merged) + len(c) + 1 > MAX_CONTEXT_CHARS:
            break
        merged += ("\n\n" + c) if merged else c

    system_text = (
        "You are a financial question answering assistant. "
        "Use only the retrieved context. "
        "Return the shortest exact answer phrase when possible. "
        "Do not explain."
    )
    user_text = (
        f"Question: {clean_text(question)}\n\n"
        f"Retrieved Context:\n{merged}\n\n"
        "Return only the final answer phrase."
    )

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system_text},
             {"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    if model_kind == "mistral":
        return f"<s>[INST] {system_text}\n\n{user_text} [/INST]"
    return f"{system_text}\n\n{user_text}\n\nAnswer:"

def extract_supported_short_answer(candidate, contexts):
    cand = clean_text(candidate)
    cand = re.sub(r"^(answer\s*:\s*)", "", cand, flags=re.I).strip()
    cand = cand.strip(" \"'`.,;:-")
    if not cand:
        return ""

    norm_context = normalize_text(" ".join([clean_text(c) for c in contexts if clean_text(c)]))
    norm_cand = normalize_text(cand)

    if norm_cand and norm_cand in norm_context and not bad_answer(cand):
        return cand

    words = cand.split()
    for span_len in range(min(8, len(words)), 0, -1):
        for start in range(0, len(words) - span_len + 1):
            span = " ".join(words[start:start + span_len]).strip(" ,.;:-")
            if normalize_text(span) in norm_context and not bad_answer(span):
                return span
    return ""

def sentence_overlap_fallback(question, contexts):
    q_terms = set(normalize_question(question).split())
    sentences = re.split(r"(?<=[\.\!\?;])\s+", " ".join([clean_text(c) for c in contexts if clean_text(c)]))

    best_sent, best_score = "", -1
    for sent in sentences:
        sent_clean = clean_text(sent)
        if not sent_clean:
            continue
        overlap = len(q_terms & set(normalize_text(sent_clean).split()))
        if overlap > best_score:
            best_score = overlap
            best_sent = sent_clean

    if not best_sent:
        return ""
    candidate = " ".join(best_sent.split()[: min(8, len(best_sent.split()))]).strip(" ,.;:")
    return "" if bad_answer(candidate) else candidate

def is_metadata_question(question):
    q = normalize_question(question)
    return ("what filing year is associated with filing" in q) or ("what is the sec cik for filing" in q)

def heuristic_extract_answer(question, contexts):
    q = normalize_question(question)
    full_context = " ".join([clean_text(c) for c in contexts if clean_text(c)])

    if "sec cik" in q:
        m = re.search(r"\bCIK:\s*([\d]+)\b", full_context, flags=re.I) or re.search(r"\b([\d]{5,10})\b", full_context)
        if m:
            return clean_text(m.group(1))

    if "filing year" in q:
        m = re.search(r"\bFiling year:\s*(20\d{2})\b", full_context, flags=re.I) or re.search(r"\b(20\d{2})\b", full_context)
        if m:
            return clean_text(m.group(1))

    if "employees" in q:
        m = re.search(r"(?i)\b(?:approximately\s+)?([\d,]+)\s+employees\b", full_context)
        if m:
            return clean_text(m.group(1))

    if "headquartered" in q:
        m = re.search(r"(?i)\bheadquartered in ([A-Z][A-Za-z0-9,\-\s]{2,60}?)(?:\.|,| and\b)", full_context)
        if m:
            ans = clean_location_answer(m.group(1))
            if not bad_answer(ans):
                return ans

    if "principal executive offices" in q:
        m = re.search(r"(?i)\bour principal executive offices are located at ([^.]{12,140})\.", full_context)
        if m:
            ans = clean_text(m.group(1))
            if is_valid_address_answer(ans):
                return ans

    if "incorporated" in q:
        for pat in [
            r"(?i)\bincorporated under the laws of ([A-Z][A-Za-z\s]{2,30}?)(?:,|\.|\s+on\s|\s+in\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b)",
            r"(?i)\bincorporated in the state of ([A-Z][A-Za-z\s]{2,30}?)(?:,|\.|\s+on\s|\s+in\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b)",
            r"(?i)\bincorporated in ([A-Z][A-Za-z\s]{2,30}?)(?:,|\.|\s+on\s|\s+in\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b)"
        ]:
            m = re.search(pat, full_context)
            if m:
                ans = clean_text(m.group(1))
                if is_valid_incorporation_answer(ans):
                    return ans

    if "independent registered public accounting firm" in q:
        for pat in [
            r"(?i)\b([A-Z][A-Za-z&,\.\s]{3,60}LLP)\s+(?:has|have)\s+audited",
            r"(?i)\b([A-Z][A-Za-z&,\.\s]{3,60}LLP)\s+served as our independent registered public accounting firm",
            r"(?i)\bindependent registered public accounting firm(?: was| is)?\s+([A-Z][A-Za-z&,\.\s]{3,60}LLP)",
        ]:
            m = re.search(pat, full_context)
            if m:
                ans = clean_text(m.group(1))
                if not bad_answer(ans):
                    return ans

    if "adversely affected" in q:
        m = re.search(r"(?i)\b(?:could|may)\s+adversely affect\s+([^.]{6,80})\.", full_context)
        if m:
            ans = clean_text(m.group(1))
            if not bad_answer(ans):
                return ans

    if "risk" in q:
        m = re.search(r"(?i)\brisks related to ([^.]{6,80})\.", full_context)
        if m:
            ans = clean_text(m.group(1))
            if not bad_answer(ans):
                return ans

    return ""

def generate_local_answer(question, contexts, tokenizer, model, model_kind):
    prompt = build_prompt(question, contexts, tokenizer, model_kind=model_kind)
    device = get_model_input_device(model)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False if TEMPERATURE == 0.0 else True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if TEMPERATURE > 0:
        gen_kwargs["temperature"] = TEMPERATURE

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    raw_text = clean_text(tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True))
    supported = extract_supported_short_answer(raw_text, contexts)
    if supported:
        return supported

    fallback = sentence_overlap_fallback(question, contexts)
    if fallback:
        return fallback

    return raw_text

def final_answer_decision(question, contexts, closed_book_pred, closed_book_score, tokenizer, model, model_kind):
    heuristic = heuristic_extract_answer(question, contexts)
    if heuristic:
        return heuristic
    if is_metadata_question(question):
        return clean_text(closed_book_pred)
    if closed_book_score >= CLOSED_BOOK_SIM_THRESHOLD:
        return clean_text(closed_book_pred)

    rag_pred = generate_local_answer(question, contexts, tokenizer, model, model_kind)
    rag_pred = extract_supported_short_answer(rag_pred, contexts) or rag_pred
    if rag_pred and not bad_answer(rag_pred):
        return clean_text(rag_pred)

    return clean_text(closed_book_pred)

# ============================================================
# Evaluation
# ============================================================
def normalize_answer_for_eval(s):
    s = clean_text(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer_for_eval(prediction) == normalize_answer_for_eval(ground_truth))

def f1_score_tokens(prediction, ground_truth):
    pred_tokens = normalize_answer_for_eval(prediction).split()
    gold_tokens = normalize_answer_for_eval(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def recall_score_token_overlap(prediction, ground_truth):
    pred_tokens = normalize_answer_for_eval(prediction).split()
    gold_tokens = normalize_answer_for_eval(ground_truth).split()
    if len(gold_tokens) == 0:
        return 1.0 if len(pred_tokens) == 0 else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    return sum(common.values()) / len(gold_tokens)

def evaluate_predictions(df, pred_col="prediction", answer_col="answer"):
    ems, f1s, recs = [], [], []
    for _, row in df.iterrows():
        pred = clean_text(row[pred_col])
        gold = clean_text(row[answer_col])
        ems.append(exact_match_score(pred, gold))
        f1s.append(f1_score_tokens(pred, gold))
        recs.append(recall_score_token_overlap(pred, gold))
    return {
        "exact_match": float(np.mean(ems)) if ems else 0.0,
        "f1": float(np.mean(f1s)) if ems else 0.0,
        "answer_recall": float(np.mean(recs)) if ems else 0.0,
        "count": int(len(df)),
    }

def question_type(question):
    q = normalize_question(question)
    if "sec cik" in q:
        return "metadata_cik"
    if "filing year" in q:
        return "metadata_year"
    if "headquartered" in q:
        return "headquarters"
    if "principal executive offices" in q:
        return "address"
    if "incorporated" in q:
        return "incorporation"
    if "employees" in q:
        return "employees"
    if "independent registered public accounting firm" in q:
        return "auditor"
    if "risk" in q:
        return "risk"
    if "adversely affected" in q:
        return "impact"
    return "other"

def add_eval_columns(df, pred_col="prediction", answer_col="answer"):
    out = df.copy()
    out["em"] = out.apply(lambda r: exact_match_score(r[pred_col], r[answer_col]), axis=1)
    out["f1"] = out.apply(lambda r: f1_score_tokens(r[pred_col], r[answer_col]), axis=1)
    out["answer_recall_metric"] = out.apply(lambda r: recall_score_token_overlap(r[pred_col], r[answer_col]), axis=1)
    out["question_type"] = out["question"].apply(question_type)
    return out

def summarize_by_question_type(df, pred_col="prediction", answer_col="answer"):
    tmp = add_eval_columns(df, pred_col=pred_col, answer_col=answer_col)
    grouped = tmp.groupby("question_type").agg(
        count=("question", "count"),
        exact_match=("em", "mean"),
        f1=("f1", "mean"),
        answer_recall=("answer_recall_metric", "mean"),
    ).reset_index()
    return grouped.sort_values(["count", "question_type"], ascending=[False, True]).reset_index(drop=True)

def pick_examples(result_df, n_best=4, n_worst=4):
    if result_df is None or len(result_df) == 0:
        return pd.DataFrame()

    tmp = add_eval_columns(result_df)
    best = tmp.sort_values(["f1", "em"], ascending=[False, False]).head(n_best)
    worst = tmp.sort_values(["f1", "em"], ascending=[True, True]).head(n_worst)

    keep_cols = [c for c in [
        "question", "answer", "prediction", "section", "context",
        "retrieved_contexts", "retrieved_source_datasets", "retrieved_train_question",
        "decision_source", "model_name", "question_type", "em", "f1", "answer_recall_metric"
    ] if c in tmp.columns]

    out = pd.concat([best[keep_cols], worst[keep_cols]], axis=0)
    return out.drop_duplicates(subset=[c for c in ["question", "answer", "prediction", "model_name"] if c in out.columns]).reset_index(drop=True)

# ============================================================
# Main pipeline
# ============================================================
def main():
    print("Step 1: Load SEC parquet stream")
    sec_stream = load_sec_stream()

    print("\nStep 2: Sample usable filings")
    filings_df = sample_filings(sec_stream, max_filings=MAX_FILINGS)
    print(f"Selected filings: {len(filings_df)}")
    safe_display(filings_df[["doc_id", "year", "cik", "filename"]], n=5)

    print("\nStep 3: Extract sections and KB chunks")
    section_rows, chunk_rows = [], []

    for _, row in tqdm(filings_df.iterrows(), total=len(filings_df), desc="Extracting sections"):
        sections = extract_sections_from_filing(row["text"])
        for section_name, section_text in sections.items():
            if word_count(section_text) < MIN_SECTION_WORDS:
                continue

            section_rows.append({
                "doc_id": row["doc_id"],
                "year": row["year"],
                "cik": row["cik"],
                "filename": row["filename"],
                "section": section_name,
                "section_text": section_text,
            })

            chunks = split_into_chunks(section_text, chunk_words=CHUNK_WORDS, overlap_words=CHUNK_OVERLAP)
            for chunk_idx, chunk in enumerate(chunks[:MAX_CHUNKS_PER_SECTION]):
                chunk_rows.append({
                    "doc_id": row["doc_id"],
                    "year": row["year"],
                    "cik": row["cik"],
                    "filename": row["filename"],
                    "section": section_name,
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk,
                })

    sections_df = pd.DataFrame(section_rows)
    chunks_df = pd.DataFrame(chunk_rows)

    if len(sections_df) == 0:
        raise RuntimeError("No usable sections extracted.")

    chunks_df["section_priority"] = chunks_df["section"].map(SECTION_PRIORITY).fillna(999)
    chunks_df = chunks_df.sort_values(["section_priority", "year", "doc_id", "chunk_idx"]).head(MAX_CHUNKS_TOTAL).reset_index(drop=True)
    chunks_df.drop(columns=["section_priority"], errors="ignore").to_csv(ARTIFACTS_DIR / "knowledge_base_chunks.csv", index=False)

    print(f"Usable sections: {len(sections_df)}")
    print(f"Saved KB chunks: {len(chunks_df)}")
    safe_display(sections_df[["doc_id", "year", "section"]], n=10)

    print("\nStep 4: Generate SEC metadata QA")
    meta_qas = []
    for _, row in tqdm(filings_df.iterrows(), total=len(filings_df), desc="Metadata QA"):
        meta_qas.extend(generate_metadata_qas(row.to_dict()))
    meta_qa_df = ensure_dataframe(pd.DataFrame(meta_qas), EXPECTED_QA_COLUMNS)
    print(f"Metadata QA pairs: {len(meta_qa_df)}")

    print("\nStep 5: Generate deterministic section QA")
    det_qas = []
    for _, row in tqdm(sections_df.iterrows(), total=len(sections_df), desc="Section QA"):
        row_dict = row.to_dict()
        det_qas.extend(regex_extract_qas_from_section(row_dict, row_dict["section"], row_dict["section_text"]))
        det_qas.extend(sentence_template_qas(row_dict, row_dict["section"], row_dict["section_text"], max_qas_per_section=4))

    det_qa_df = ensure_dataframe(pd.DataFrame(det_qas), EXPECTED_QA_COLUMNS)
    print(f"Deterministic QA pairs before dedup: {len(det_qa_df)}")

    print("\nStep 6: Merge and clean SEC QA")
    all_qas = pd.concat([meta_qa_df, det_qa_df], ignore_index=True)
    all_qas = ensure_dataframe(all_qas, EXPECTED_QA_COLUMNS)

    for col in EXPECTED_QA_COLUMNS:
        all_qas[col] = all_qas[col].apply(clean_text)

    all_qas = all_qas[
        (all_qas["question"].str.len() > 0) &
        (all_qas["answer"].str.len() > 0) &
        (all_qas["question"].str.endswith("?"))
    ].copy()

    all_qas = deduplicate_qas(all_qas.drop_duplicates().reset_index(drop=True))
    all_qas = all_qas.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    if len(all_qas) == 0:
        raise RuntimeError("No QA pairs survived filtering.")

    print(f"\nFinal SEC QA pairs: {len(all_qas)}")
    safe_display(all_qas[["question", "answer", "section", "source_type"]], n=30)

    dev_size = max(1, int(len(all_qas) * DEV_RATIO))
    dev_df = all_qas.iloc[:dev_size].reset_index(drop=True)
    train_df = all_qas.iloc[dev_size:].reset_index(drop=True)

    all_qas.to_csv(ARTIFACTS_DIR / "finance_all_qa.csv", index=False)
    train_df.to_csv(ARTIFACTS_DIR / "finance_train_qa.csv", index=False)
    dev_df.to_csv(ARTIFACTS_DIR / "finance_dev_qa.csv", index=False)

    write_txt(train_df["question"].tolist(), TRAIN_DIR / "questions.txt")
    write_txt(train_df["answer"].tolist(), TRAIN_DIR / "reference_answers.txt")
    write_txt(dev_df["question"].tolist(), TEST_DIR / "questions.txt")
    write_txt(dev_df["answer"].tolist(), TEST_DIR / "reference_answers.txt")

    with open(ARTIFACTS_DIR / "finance_qa_with_context.jsonl", "w", encoding="utf-8") as f:
        for rec in all_qas.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    save_json({
        "years_used": YEARS,
        "num_selected_filings": int(len(filings_df)),
        "num_sections_extracted": int(len(sections_df)),
        "num_chunks_saved": int(len(chunks_df)),
        "num_metadata_qas": int(len(meta_qa_df)),
        "num_deterministic_qas": int(len(det_qa_df)),
        "num_final_qas": int(len(all_qas)),
        "num_train_qas": int(len(train_df)),
        "num_dev_qas": int(len(dev_df)),
    }, ARTIFACTS_DIR / "run_summary_sec_generation.json")

    print("\nStep 7: Load SEC-generated train data")
    sec_df = load_and_normalize_sec_train(str(ARTIFACTS_DIR / "finance_train_qa.csv"))
    print(f"SEC normalized rows: {len(sec_df)}")

    print("\nStep 8: Load ConvFinQA QA")
    conv_df = load_and_normalize_convfinqa(MAX_CONVFINQA_ROWS)
    print(f"ConvFinQA normalized rows: {len(conv_df)}")

    print("\nStep 9: Load FinQA corpus contexts for KB")
    finqa_corpus_df = load_finqa_corpus_contexts(MAX_FINQA_CORPUS_ROWS)
    print(f"FinQA corpus contexts: {len(finqa_corpus_df)}")
    safe_display(finqa_corpus_df.head(5), n=5)

    print("\nStep 10: Merge SEC QA + ConvFinQA QA")
    target_ext = max(len(sec_df), 300)
    conv_kept = conv_df.sample(target_ext, random_state=SEED).reset_index(drop=True) if len(conv_df) > target_ext else conv_df.copy()

    combined_df = pd.concat([sec_df, conv_kept], ignore_index=True)
    combined_df = clean_final_df(combined_df).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    combined_df.to_csv(ARTIFACTS_DIR / "combined_train.csv", index=False)
    with open(ARTIFACTS_DIR / "combined_train.jsonl", "w", encoding="utf-8") as f:
        for rec in combined_df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    save_json({
        "sec_rows": int(len(sec_df)),
        "convfinqa_rows_after_cleaning_and_sampling": int(len(conv_kept)),
        "combined_rows": int(len(combined_df)),
        "source_distribution": {k: int(v) for k, v in combined_df["source_dataset"].value_counts().to_dict().items()}
    }, ARTIFACTS_DIR / "source_stats.json")

    print("\nSample combined rows:")
    safe_display(combined_df.head(20), n=20)

    print("\nStep 11: Load existing files for final RAG")
    sec_train_df = pd.read_csv(ARTIFACTS_DIR / "finance_train_qa.csv")
    sec_dev_df = pd.read_csv(ARTIFACTS_DIR / "finance_dev_qa.csv")
    kb_chunks_df = pd.read_csv(ARTIFACTS_DIR / "knowledge_base_chunks.csv")
    combined_train_df = pd.read_csv(ARTIFACTS_DIR / "combined_train.csv")

    print("Cleaning datasets...")
    sec_train_df = clean_final_df(sec_train_df.assign(source_dataset="sec_generated", source_split="train"))
    sec_dev_df = clean_final_df(sec_dev_df.assign(source_dataset="sec_generated", source_split="dev"))
    combined_train_df = clean_final_df(combined_train_df)

    kb_chunks_df = ensure_columns(kb_chunks_df, ["doc_id", "year", "cik", "filename", "section", "chunk_text"])
    for c in kb_chunks_df.columns:
        kb_chunks_df[c] = kb_chunks_df[c].apply(clean_text)
    kb_chunks_df = kb_chunks_df[kb_chunks_df["chunk_text"].apply(lambda x: word_count(x) >= MIN_KB_CHUNK_WORDS)].reset_index(drop=True)

    sec_train_df.to_csv(ARTIFACTS_DIR / "cleaned_sec_train.csv", index=False)
    sec_dev_df.to_csv(ARTIFACTS_DIR / "cleaned_sec_dev.csv", index=False)
    combined_train_df.to_csv(ARTIFACTS_DIR / "cleaned_combined_train.csv", index=False)

    print("\nCleaned dataset sizes:")
    print("SEC train:", len(sec_train_df))
    print("SEC dev:", len(sec_dev_df))
    print("Combined train:", len(combined_train_df))
    print("Original KB chunks:", len(kb_chunks_df))
    safe_display(sec_dev_df[["question", "answer", "section"]], n=10)

    print("\nStep 12: Build augmented knowledge base")
    def build_augmented_kb(sec_chunks_df, sec_train, combined_train, finqa_corpus):
        rows = []

        for idx, row in sec_chunks_df.iterrows():
            chunk = clean_text(row.get("chunk_text", ""))
            if word_count(chunk) >= MIN_KB_CHUNK_WORDS:
                rows.append({
                    "kb_id": f"sec_chunk_{idx}",
                    "kb_source": "sec_chunks",
                    "source_dataset": "sec_generated",
                    "doc_id": clean_text(row.get("doc_id", "")),
                    "year": clean_text(row.get("year", "")),
                    "cik": clean_text(row.get("cik", "")),
                    "filename": clean_text(row.get("filename", "")),
                    "section": clean_text(row.get("section", "")),
                    "chunk_text": chunk,
                })

        for idx, row in sec_train.iterrows():
            ctx = clean_text(row.get("context", ""))
            if word_count(ctx) < MIN_KB_CHUNK_WORDS:
                continue
            for j, ch in enumerate(split_into_chunks(ctx, EXTERNAL_CONTEXT_CHUNK_WORDS, EXTERNAL_CONTEXT_OVERLAP)):
                if word_count(ch) >= MIN_KB_CHUNK_WORDS:
                    rows.append({
                        "kb_id": f"sec_ctx_{idx}_{j}",
                        "kb_source": "sec_train_context",
                        "source_dataset": clean_text(row.get("source_dataset", "sec_generated")),
                        "doc_id": clean_text(row.get("doc_id", "")),
                        "year": clean_text(row.get("year", "")),
                        "cik": clean_text(row.get("cik", "")),
                        "filename": clean_text(row.get("filename", "")),
                        "section": clean_text(row.get("section", "")),
                        "chunk_text": ch,
                    })

        ext_count = 0
        for idx, row in combined_train.iterrows():
            ctx = clean_text(row.get("context", ""))
            if word_count(ctx) < MIN_KB_CHUNK_WORDS:
                continue
            for j, ch in enumerate(split_into_chunks(ctx, EXTERNAL_CONTEXT_CHUNK_WORDS, EXTERNAL_CONTEXT_OVERLAP)):
                if ext_count >= MAX_EXTERNAL_CHUNKS:
                    break
                if word_count(ch) >= MIN_KB_CHUNK_WORDS:
                    rows.append({
                        "kb_id": f"ext_ctx_{idx}_{j}",
                        "kb_source": "external_train_context",
                        "source_dataset": clean_text(row.get("source_dataset", "")),
                        "doc_id": clean_text(row.get("doc_id", "")),
                        "year": clean_text(row.get("year", "")),
                        "cik": clean_text(row.get("cik", "")),
                        "filename": clean_text(row.get("filename", "")),
                        "section": clean_text(row.get("section", "")),
                        "chunk_text": ch,
                    })
                    ext_count += 1
            if ext_count >= MAX_EXTERNAL_CHUNKS:
                break

        for idx, row in finqa_corpus.iterrows():
            ctx = clean_text(row.get("context", ""))
            if word_count(ctx) < MIN_KB_CHUNK_WORDS:
                continue
            for j, ch in enumerate(split_into_chunks(ctx, EXTERNAL_CONTEXT_CHUNK_WORDS, EXTERNAL_CONTEXT_OVERLAP)):
                if word_count(ch) >= MIN_KB_CHUNK_WORDS:
                    rows.append({
                        "kb_id": f"finqa_ctx_{idx}_{j}",
                        "kb_source": "finqa_corpus_context",
                        "source_dataset": "finqa_corpus",
                        "doc_id": clean_text(row.get("doc_id", "")),
                        "year": "",
                        "cik": "",
                        "filename": "",
                        "section": "",
                        "chunk_text": ch,
                    })

        kb = pd.DataFrame(rows)
        if len(kb) == 0:
            raise RuntimeError("Augmented KB is empty.")

        kb["chunk_text"] = kb["chunk_text"].apply(clean_text)
        kb = kb[kb["chunk_text"].apply(lambda x: word_count(x) >= MIN_KB_CHUNK_WORDS)].copy()
        kb["chunk_norm"] = kb["chunk_text"].apply(normalize_text)
        kb = kb.drop_duplicates(subset=["chunk_norm"]).drop(columns=["chunk_norm"]).reset_index(drop=True)
        return kb

    aug_kb_df = build_augmented_kb(kb_chunks_df, sec_train_df, combined_train_df, finqa_corpus_df)
    aug_kb_df.to_csv(ARTIFACTS_DIR / "augmented_kb_chunks.csv", index=False)

    print("Augmented KB size:", len(aug_kb_df))
    print("\nBy kb_source:")
    print(aug_kb_df["kb_source"].value_counts(dropna=False).to_string())
    print("\nBy source_dataset:")
    print(aug_kb_df["source_dataset"].value_counts(dropna=False).head(10).to_string())

    print("\nStep 13: Build closed-book baseline")
    closed_book_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=50000)
    closed_book_train_questions = combined_train_df["question"].fillna("").tolist()
    closed_book_train_answers = combined_train_df["answer"].fillna("").tolist()

    closed_book_X = closed_book_vectorizer.fit_transform(closed_book_train_questions)
    closed_book_dev_X = closed_book_vectorizer.transform(sec_dev_df["question"].fillna("").tolist())
    closed_book_sim = cosine_similarity(closed_book_dev_X, closed_book_X)
    closed_book_best_idx = np.argmax(closed_book_sim, axis=1)

    closed_book_preds = [clean_text(closed_book_train_answers[idx]) for idx in closed_book_best_idx]
    closed_book_retrieved_q = [clean_text(closed_book_train_questions[idx]) for idx in closed_book_best_idx]
    closed_book_scores = [float(closed_book_sim[i, idx]) for i, idx in enumerate(closed_book_best_idx)]

    closed_book_results = sec_dev_df.copy()
    closed_book_results["prediction"] = closed_book_preds
    closed_book_results["retrieved_train_question"] = closed_book_retrieved_q
    closed_book_results["retrieval_score"] = closed_book_scores
    closed_book_results["model_name"] = "closed_book_tfidf_nn"

    closed_book_results.to_csv(ARTIFACTS_DIR / "closed_book_predictions.csv", index=False)
    closed_book_metrics = evaluate_predictions(closed_book_results)
    print("Closed-book metrics:", json.dumps(closed_book_metrics, indent=2))

    print("\nStep 14: Prepare dense retriever")
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)

    kb_embeddings = dense_model.encode(
        ["passage: " + clean_text(x) for x in aug_kb_df["chunk_text"].tolist()],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dev_embeddings = dense_model.encode(
        ["query: " + clean_text(x) for x in sec_dev_df["question"].tolist()],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    faiss_index = faiss.IndexFlatIP(kb_embeddings.shape[1])
    faiss_index.add(kb_embeddings.astype("float32"))
    dense_scores, dense_neighbors = faiss_index.search(dev_embeddings.astype("float32"), DENSE_TOPK)

    del dense_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nStep 15: Load local reader")
    MODEL_KIND, LOCAL_MODEL_PATH = choose_local_model_path()
    print("Using local model kind:", MODEL_KIND)
    print("Using local model path:", LOCAL_MODEL_PATH)

    reader_tokenizer, reader_model = load_local_model(LOCAL_MODEL_PATH)
    print("Local reader loaded successfully.")

    print("\nStep 16: Run final hybrid RAG system")
    final_predictions = []
    retrieved_indices = []
    retrieved_contexts = []
    retrieval_scores = []
    retrieved_source_datasets = []
    decision_sources = []

    for i in tqdm(range(len(sec_dev_df)), desc="Hybrid Dense Local RAG inference"):
        top_idx = dense_neighbors[i].tolist()
        retrieved_rows = aug_kb_df.iloc[top_idx]
        contexts = retrieved_rows["chunk_text"].tolist()

        pred = final_answer_decision(
            question=sec_dev_df.iloc[i]["question"],
            contexts=contexts,
            closed_book_pred=closed_book_preds[i],
            closed_book_score=closed_book_scores[i],
            tokenizer=reader_tokenizer,
            model=reader_model,
            model_kind=MODEL_KIND,
        )

        heuristic = heuristic_extract_answer(sec_dev_df.iloc[i]["question"], contexts)
        if heuristic:
            source = "heuristic_extract"
        elif is_metadata_question(sec_dev_df.iloc[i]["question"]):
            source = "closed_book_metadata"
        elif closed_book_scores[i] >= CLOSED_BOOK_SIM_THRESHOLD:
            source = "closed_book_similarity"
        else:
            source = "rag_or_fallback"

        final_predictions.append(clean_text(pred))
        retrieved_indices.append(json.dumps([int(x) for x in top_idx]))
        retrieved_contexts.append(" ||| ".join([clean_text(x) for x in contexts]))
        retrieval_scores.append(json.dumps([float(x) for x in dense_scores[i].tolist()]))
        retrieved_source_datasets.append(json.dumps(retrieved_rows["source_dataset"].fillna("").astype(str).tolist()))
        decision_sources.append(source)

    final_results = sec_dev_df.copy()
    final_results["prediction"] = final_predictions
    final_results["retrieved_indices"] = retrieved_indices
    final_results["retrieved_contexts"] = retrieved_contexts
    final_results["retrieval_scores"] = retrieval_scores
    final_results["retrieved_source_datasets"] = retrieved_source_datasets
    final_results["decision_source"] = decision_sources
    final_results["model_name"] = f"hybrid_dense_local_{MODEL_KIND}_rag"

    final_results.to_csv(ARTIFACTS_DIR / "dense_augmented_rag_predictions.csv", index=False)
    final_metrics = evaluate_predictions(final_results)
    print("Final hybrid metrics:", json.dumps(final_metrics, indent=2))

    print("\nStep 17: Create analysis artifacts")
    qual_rows = []
    for name, df in [("closed_book", closed_book_results), ("final_hybrid_rag", final_results)]:
        ex = pick_examples(df)
        if len(ex) > 0:
            ex = ex.copy()
            ex["system"] = name
            qual_rows.append(ex)

    qualitative_examples = pd.concat(qual_rows, ignore_index=True) if qual_rows else pd.DataFrame()
    qualitative_examples.to_csv(ARTIFACTS_DIR / "qualitative_examples.csv", index=False)

    summarize_by_question_type(closed_book_results).to_csv(ARTIFACTS_DIR / "closed_book_metrics_by_question_type.csv", index=False)
    summarize_by_question_type(final_results).to_csv(ARTIFACTS_DIR / "final_hybrid_metrics_by_question_type.csv", index=False)

    save_json({
        "closed_book_tfidf_nn": closed_book_metrics,
        "final_hybrid_dense_local_rag": final_metrics,
        "delta_exact_match": float(final_metrics["exact_match"] - closed_book_metrics["exact_match"]),
        "delta_f1": float(final_metrics["f1"] - closed_book_metrics["f1"]),
        "delta_answer_recall": float(final_metrics["answer_recall"] - closed_book_metrics["answer_recall"]),
        "hybrid_beats_closed_book_on_em": bool(final_metrics["exact_match"] > closed_book_metrics["exact_match"]),
        "hybrid_beats_closed_book_on_f1": bool(final_metrics["f1"] > closed_book_metrics["f1"]),
        "hybrid_beats_closed_book_on_answer_recall": bool(final_metrics["answer_recall"] > closed_book_metrics["answer_recall"]),
    }, ARTIFACTS_DIR / "model_comparison_summary.json")

    print("\nStep 18: Generate submission outputs")

    if os.path.exists(STAFF_TEST_QUESTIONS_PATH):
        with open(STAFF_TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = [clean_text(line) for line in f if clean_text(line)]
        official_test_used = True
        print(f"Using official staff questions from: {STAFF_TEST_QUESTIONS_PATH}")
    else:
        test_questions = sec_dev_df["question"].tolist()
        official_test_used = False
        print("Official staff questions not found; using SEC dev questions as a smoke test.")

    closed_book_test_X = closed_book_vectorizer.transform(test_questions)
    closed_book_test_sim = cosine_similarity(closed_book_test_X, closed_book_X)
    closed_book_test_best_idx = np.argmax(closed_book_test_sim, axis=1)
    closed_book_test_preds = [clean_text(closed_book_train_answers[idx]) for idx in closed_book_test_best_idx]
    closed_book_test_scores = [float(closed_book_test_sim[i, idx]) for i, idx in enumerate(closed_book_test_best_idx)]

    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    test_query_embeddings = dense_model.encode(
        ["query: " + clean_text(q) for q in test_questions],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    test_scores, test_neighbors = faiss_index.search(test_query_embeddings.astype("float32"), DENSE_TOPK)

    system_outputs = []
    submission_trace_rows = []

    for i, q in tqdm(list(enumerate(test_questions)), desc="Submission inference"):
        top_idx = test_neighbors[i].tolist()
        retrieved_rows = aug_kb_df.iloc[top_idx]
        contexts = retrieved_rows["chunk_text"].tolist()

        pred = final_answer_decision(
            question=q,
            contexts=contexts,
            closed_book_pred=closed_book_test_preds[i],
            closed_book_score=closed_book_test_scores[i],
            tokenizer=reader_tokenizer,
            model=reader_model,
            model_kind=MODEL_KIND,
        )
        pred = clean_text(pred)
        system_outputs.append(pred)

        submission_trace_rows.append({
            "question": clean_text(q),
            "prediction": pred,
            "closed_book_prediction": clean_text(closed_book_test_preds[i]),
            "closed_book_score": float(closed_book_test_scores[i]),
            "retrieved_indices": json.dumps([int(x) for x in top_idx]),
            "retrieved_source_datasets": json.dumps(retrieved_rows["source_dataset"].fillna("").astype(str).tolist()),
            "retrieved_contexts": " ||| ".join([clean_text(x) for x in contexts]),
        })

    write_txt(system_outputs, SYSTEM_OUTPUTS_DIR / "system_output_1.txt")
    pd.DataFrame(submission_trace_rows).to_csv(ARTIFACTS_DIR / "submission_inference_trace.csv", index=False)

    save_json({
        "dataset_summary": {
            "sec_all_rows": int(len(all_qas)),
            "sec_train_rows": int(len(train_df)),
            "sec_dev_rows": int(len(dev_df)),
            "combined_train_rows": int(len(combined_df)),
            "original_kb_chunk_rows": int(len(kb_chunks_df)),
            "augmented_kb_chunk_rows": int(len(aug_kb_df)),
            "finqa_corpus_context_rows": int(len(finqa_corpus_df)),
        },
        "models": {
            "closed_book_tfidf_nn": closed_book_metrics,
            "final_hybrid_dense_local_rag": final_metrics,
        },
        "settings": {
            "local_model_kind": MODEL_KIND,
            "local_model_path": LOCAL_MODEL_PATH,
            "dense_model_name": DENSE_MODEL_NAME,
            "dense_topk": DENSE_TOPK,
            "final_context_topk": FINAL_CONTEXT_TOPK,
            "max_context_chars": MAX_CONTEXT_CHARS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "use_4bit": USE_4BIT,
            "closed_book_sim_threshold": CLOSED_BOOK_SIM_THRESHOLD,
            "external_context_chunk_words": EXTERNAL_CONTEXT_CHUNK_WORDS,
            "external_context_overlap": EXTERNAL_CONTEXT_OVERLAP,
            "max_external_chunks": MAX_EXTERNAL_CHUNKS,
            "staff_test_questions_used": bool(official_test_used),
        },
        "report_notes": {
            "retrieve_and_augment_vs_closed_book": "Included through a closed-book TF-IDF nearest-neighbor baseline and the final hybrid RAG system.",
            "qualitative_examples_available": True,
            "per_question_type_analysis_available": True,
            "official_submission_output_uses_staff_test": bool(official_test_used),
        }
    }, ARTIFACTS_DIR / "metrics_summary.json")

    print("\nStep 19: Write package files")

    README_MD = f"""# Finance RAG Submission

This project builds a retrieval-augmented question answering system for the finance domain using public SEC filings and public finance QA resources.

## Assignment
CS5170 Assignment 2: End-to-end-NLP-System-Building

## Domain
Public U.S. SEC 10-K filings, with supporting public finance corpora.

## Main idea
The system answers factual finance questions by combining conservative QA generation from SEC filings, dense retrieval over an augmented knowledge base, lightweight extractive rules, and a local generative reader.

## Data sources
- SEC filings from PleIAs SEC parquet data for years: {YEARS}
- ConvFinQA used as external QA training data
- FinQA mirror corpus used as KB-only external context

## Pipeline
1. Stream public SEC filings.
2. Extract core 10-K sections (Item 1, 1A, 7, 7A, 8).
3. Generate conservative SEC QA pairs.
4. Save train and dev files.
5. Load ConvFinQA and FinQA mirror corpus.
6. Build a merged training set and an augmented retrieval knowledge base.
7. Run:
   - a closed-book TF-IDF baseline
   - a dense-retrieval hybrid RAG system
8. Generate final outputs in `system_outputs/system_output_1.txt`.

## System design
- Train data:
  - SEC-generated QA pairs
  - ConvFinQA QA pairs
- Retrieval KB:
  - SEC filing chunks
  - SEC training contexts
  - ConvFinQA contexts
  - FinQA corpus contexts
- Retriever:
  - {DENSE_MODEL_NAME} with FAISS
- Reader:
  - local {MODEL_KIND} model
- Answer policy:
  - metadata heuristics first
  - extractive heuristics next
  - closed-book fallback for confident cases
  - dense RAG + reader otherwise
  - short supported-span postprocessing at the end

## Environment note
The full end-to-end system was developed and tested primarily on Kaggle because the final setup used dense retrieval together with a local Mistral 7B reader, which required GPU resources we did not have available for reliable local testing.

## Run summary
- Selected filings: {len(filings_df)}
- Usable sections: {len(sections_df)}
- Saved KB chunks: {len(chunks_df)}
- Metadata QA pairs: {len(meta_qa_df)}
- Deterministic QA pairs before deduplication: {len(det_qa_df)}
- Final SEC QA pairs: {len(all_qas)}
- SEC normalized train rows: {len(sec_df)}
- ConvFinQA normalized rows: {len(conv_df)}
- FinQA corpus contexts: {len(finqa_corpus_df)}
- Combined train rows: {len(combined_df)}
- Augmented KB size: {len(aug_kb_df)}

## Dev metrics
### Closed-book TF-IDF baseline
- exact_match: {closed_book_metrics["exact_match"]}
- f1: {closed_book_metrics["f1"]}
- answer_recall: {closed_book_metrics["answer_recall"]}
- count: {closed_book_metrics["count"]}

### Final hybrid RAG
- exact_match: {final_metrics["exact_match"]}
- f1: {final_metrics["f1"]}
- answer_recall: {final_metrics["answer_recall"]}
- count: {final_metrics["count"]}

## Official test handling
- If `{STAFF_TEST_QUESTIONS_PATH}` exists, the final output file is generated on that staff test set.
- Otherwise, SEC dev questions are used only as a smoke test.
- `staff_test_questions_used`: {official_test_used}

## Required submission files
- report.pdf
- github_url.txt
- contributions.md
- data/train/questions.txt
- data/train/reference_answers.txt
- data/test/questions.txt
- data/test/reference_answers.txt
- system_outputs/system_output_1.txt
- README.md

## Helpful analysis files
Additional analysis files are saved under `artifacts/` for reporting and inspection.
"""

    with open(SUBMISSION_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write(README_MD)

    with open(SUBMISSION_ROOT / "github_url.txt", "w", encoding="utf-8") as f:
        f.write(GITHUB_URL_VALUE.strip() + "\n")

    with open(SUBMISSION_ROOT / "contributions.md", "w", encoding="utf-8") as f:
        f.write(CONTRIBUTIONS_MD_VALUE)

    with open(SUBMISSION_ROOT / "REPORT_INSTRUCTIONS.txt", "w", encoding="utf-8") as f:
        f.write("Place your final report here as report.pdf before submission.\n")

    print("\nStep 20: Save file manifest and zip submission folder")

    file_manifest = []
    for p in sorted(SUBMISSION_ROOT.rglob("*")):
        if p.is_file():
            file_manifest.append(str(p.relative_to(SUBMISSION_ROOT)))

    save_json(file_manifest, ARTIFACTS_DIR / "file_manifest.json")

    print("\nSaved files:")
    for p in file_manifest:
        print("-", p)

    zip_base = str(WORK_DIR / BID_NAME)
    zip_file = f"{zip_base}.zip"
    shutil.make_archive(zip_base, "zip", root_dir=WORK_DIR, base_dir=BID_NAME)

    print("\nDone.")
    print(f"\nSubmission folder: {SUBMISSION_ROOT}")
    print(f"Submission zip: {zip_file}")

if __name__ == "__main__":
    main()