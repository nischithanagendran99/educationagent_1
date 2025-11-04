import argparse
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from src.utils.config import RAW_DIR, CLEAN_DIR
from src.schemas.columns import UNIFIED_CODE_COLUMNS

# Required text columns for a usable row
REQ_TEXT_COLS = ["prompt"]


def _read_parquet(in_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(in_path, engine="pyarrow")
    # Ensure unified columns exist & are ordered
    for col in UNIFIED_CODE_COLUMNS.keys():
        if col not in df.columns:
            df[col] = ""
    return df[list(UNIFIED_CODE_COLUMNS.keys())]


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        s = df[c].astype(str)
        # collapse whitespace to single spaces and strip
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        # remove literal "None"
        df[c] = s.where(s.ne("None"), "")
    # normalize tags
    if "tags" in df.columns:
        df["tags"] = (
            df["tags"]
            .astype(str)
            .str.replace(r"\s*,\s*", ",", regex=True)
            .str.strip(", ")
            .str.lower()
        )
    # normalize difficulty
    if "difficulty" in df.columns:
        df["difficulty"] = df["difficulty"].astype(str).str.lower()
    return df


def _drop_empties(df: pd.DataFrame, require_solution: bool = False) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for c in REQ_TEXT_COLS:
        if c in df.columns:
            mask &= df[c].astype(str).str.len() > 0
    if require_solution and "solution" in df.columns:
        mask &= df["solution"].astype(str).str.len() > 0
    return df.loc[mask].copy()


def _dedupe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    # Stable hash on (source, title, prompt)
    def _row_key(row) -> str:
        h = hashlib.sha256()
        for k in ("source", "title", "prompt"):
            h.update(str(row.get(k, "")).encode("utf-8"))
        return h.hexdigest()

    keys = df.apply(_row_key, axis=1)
    before = len(df)
    kept = ~keys.duplicated()
    df = df.loc[kept].copy()
    return df, (before - len(df))


def _summary(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {"rows": f"{len(df):,}"}
    for c in ["prompt", "solution", "language", "difficulty", "tags"]:
        if c in df.columns:
            out[f"null_{c}"] = str(int(df[c].isna().sum()))
            out[f"empty_{c}"] = str(int((df[c].astype(str).str.len() == 0).sum()))
    if "title" in df.columns:
        out["unique_titles"] = str(df["title"].nunique())
    return out


def clean_one(dataset_key: str, split: str, require_solution: bool = False) -> None:
    in_path = RAW_DIR / dataset_key / f"{split}.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    print(f"🔎 Loading {in_path}")
    df = _read_parquet(in_path)

    print("🧹 Normalizing strings/tags/difficulty…")
    df = _normalize_strings(df)

    print("🚫 Dropping empty required fields…")
    df = _drop_empties(df, require_solution=require_solution)

    print("🧬 Deduplicating…")
    df, removed = _dedupe(df)
    print(f"   dedup removed: {removed}")

    stats = _summary(df)
    print("📊 Summary:", stats)

    out_dir = CLEAN_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"✅ Clean saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Validate + clean unified parquet into data/clean")
    ap.add_argument("--dataset", required=True,
                    help="Folder name under data/raw (e.g., leetcode, apps, codeforces, codesearchnet)")
    ap.add_argument("--split", default="train",
                    help="Split name (default: train)")
    ap.add_argument("--require-solution", action="store_true",
                    help="Drop rows without solution text (useful for APPS)")
    args = ap.parse_args()

    clean_one(args.dataset, args.split, require_solution=args.require_solution)


if __name__ == "__main__":
    main()
