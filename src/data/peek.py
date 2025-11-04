import argparse
import pandas as pd
from src.utils.config import RAW_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=3, help="rows to show")
    args = ap.parse_args()

    path = RAW_DIR / args.dataset / f"{args.split}.parquet"
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"📦 {args.dataset}/{args.split}: {len(df):,} rows, {len(df.columns)} cols")
    print("🧾 columns:", list(df.columns))
    for col in ["title", "prompt", "solution", "difficulty", "tags"]:
        if col in df.columns:
            nonempty = (df[col].astype(str).str.len() > 0).sum()
            print(f"   non-empty {col}: {nonempty:,}")
    print("\n🔍 sample:")
    print(df[["title","prompt","difficulty","tags"]].head(args.n).to_string(index=False))

if __name__ == "__main__":
    main()
