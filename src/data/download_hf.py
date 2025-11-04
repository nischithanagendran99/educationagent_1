import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from datasets import load_dataset

from src.utils.config import RAW_DIR
from src.schemas.columns import UNIFIED_CODE_COLUMNS


# ---------- helpers ----------
def _enforce_unified(df: pd.DataFrame) -> pd.DataFrame:
    for col in UNIFIED_CODE_COLUMNS.keys():
        if col not in df.columns:
            df[col] = ""
    return df[list(UNIFIED_CODE_COLUMNS.keys())]


def _save_parquet(df: pd.DataFrame, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"✅ saved -> {out_path}")


# ---------- mappers ----------
def _first(rec: Dict[str, Any], keys) -> str:
    for k in keys:
        v = rec.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() != "none":
            return s
    return ""

def map_leetcode(rec: Dict[str, Any]) -> Dict[str, Any]:
    def _first(rec, keys):
        for k in keys:
            v = rec.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() != "none":
                return s
        return ""

    qid = _first(rec, ["question_id", "id"])
    title = _first(rec, ["title", "question_title"]) or (f"leetcode_{qid}" if qid else "leetcode_problem")

    return {
        "source": "LeetCodeDataset",
        "dataset_id": qid,
        "title": title,
        "prompt": _first(rec, ["content", "translatedContent", "description", "question", "body", "prompt"]),
        "solution": _first(rec, ["solution", "accepted_answer", "reference_solution"]),
        "language": "",
        "difficulty": _first(rec, ["difficulty", "level"]),
        "tags": ",".join(rec.get("tags", []) or []),
        "split": ""  # set later
    }

def map_apps(rec: Dict[str, Any]) -> Dict[str, Any]:
    # codeparrot/apps
    sols = rec.get("solutions")
    if isinstance(sols, list):
        sols = "\n\n---\n\n".join([str(s) for s in sols])
    return {
        "source": "APPS",
        "dataset_id": str(rec.get("problem_id") or rec.get("id") or ""),
        "title": rec.get("title") or "",
        "prompt": rec.get("question") or rec.get("prompt") or "",
        "solution": sols or rec.get("solution") or "",
        "language": "python",  # APPS is mostly Python
        "difficulty": str(rec.get("difficulty") or ""),
        "tags": "",
        "split": ""  # set later
    }


def map_codeforces(rec: Dict[str, Any]) -> Dict[str, Any]:
    # DenCT/codeforces-problems-7k
    return {
        "source": "Codeforces",
        "dataset_id": str(rec.get("id") or rec.get("problem_id") or ""),
        "title": rec.get("name") or rec.get("title") or "",
        "prompt": rec.get("statement") or rec.get("prompt") or rec.get("description") or "",
        "solution": "",
        "language": "",
        "difficulty": str(rec.get("rating") or rec.get("difficulty") or ""),
        "tags": ",".join(rec.get("tags", []) or []),
        "split": ""  # set later
    }


def map_codesearchnet(rec: Dict[str, Any]) -> Dict[str, Any]:
    # sentence-transformers/codesearchnet or Nan-Do/code-search-net-python
    return {
        "source": "CodeSearchNet",
        "dataset_id": str(rec.get("func_id") or rec.get("id") or ""),
        "title": rec.get("func_name") or rec.get("path") or "",
        "prompt": rec.get("docstring") or rec.get("original_string") or "",
        "solution": rec.get("code") or "",
        "language": rec.get("language") or rec.get("programming_language") or "",
        "difficulty": "",
        "tags": "",
        "split": ""  # set later
    }


# ---------- registry ----------
REGISTRY = {
    # key                 HF path                           mapper            default_split(s)
    "leetcode":     ("newfacade/LeetCodeDataset",           map_leetcode,     ["train"]),
    "apps":         ("codeparrot/apps",                     map_apps,         ["train"]),
    "codeforces":   ("DenCT/codeforces-problems-7k",        map_codeforces,   ["train"]),
    "codesearchnet":("sentence-transformers/codesearchnet", map_codesearchnet,["train"]),
    # more will be added later…
}


def load_and_map(hf_path: str, mapper, split: str) -> pd.DataFrame:
    print(f"⏬ loading: {hf_path}  split={split}")

    # Some datasets (e.g., codeparrot/apps) need trust_remote_code
    load_kwargs = {}
    if hf_path == "codeparrot/apps":
        load_kwargs["trust_remote_code"] = True

    ds = load_dataset(hf_path, split=split, **load_kwargs)
    rows: List[Dict[str, Any]] = [mapper(rec) for rec in ds]
    df = pd.DataFrame(rows)
    df = _enforce_unified(df)

    # strip strings column-wise (avoid deprecated applymap)
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        s = df[c].astype(str).str.strip()
        df[c] = s.where(s != "None", "")

    df["split"] = split
    return df


def main():
    ap = argparse.ArgumentParser(description="Download & normalize HF datasets into unified parquet.")
    ap.add_argument("--dataset", required=True, choices=list(REGISTRY.keys()),
                    help="Which dataset key to download.")
    ap.add_argument("--split", action="append",
                    help="Optionally override split(s). Can repeat, e.g. --split train --split validation")
    args = ap.parse_args()

    hf_path, mapper, default_splits = REGISTRY[args.dataset]
    splits = args.split or default_splits

    all_parts = []
    for sp in splits:
        df = load_and_map(hf_path, mapper, sp)
        all_parts.append((sp, df))

    # Save each split separately
    for sp, df in all_parts:
        _save_parquet(df, RAW_DIR / args.dataset, sp)

    print("✅ done.")


if __name__ == "__main__":
    main()
