from __future__ import annotations
import re
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T
from src.utils.config import CLEAN_DIR

UNIFIED_COLS = ["source","dataset_id","title","prompt","solution","language","difficulty","tags","split"]

# --- regex helpers ---
RE_MULTI_WS = re.compile(r"\s+")
RE_IMPORT_BLOCK = re.compile(
    r"^(?:\s*(?:from\s+\w+(?:\.\w+)*\s+import\s+.*|import\s+[\w\.,\s\*]+)\s*\n)+",
    re.MULTILINE,
)
RE_CODE_FENCE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)

def normalize_prompt(txt: str) -> str:
    if not txt:
        return ""
    txt2 = RE_IMPORT_BLOCK.sub("", txt)
    txt2 = RE_CODE_FENCE.sub("", txt2)
    txt2 = RE_MULTI_WS.sub(" ", txt2).strip()
    return txt2

normalize_prompt_udf = F.udf(normalize_prompt, T.StringType())

def main():
    # simpler one-liner to avoid linter false-positives
    spark = SparkSession.builder.appName("spark_clean_unified").getOrCreate()  # type: ignore[attr-defined]


    # 1) Read all cleaned parquet files
    glob_path = str((CLEAN_DIR / "*" / "*.parquet").resolve())
    # quick guard if nothing to read
    if not list((CLEAN_DIR).glob("*/*.parquet")):
        print(f"[spark] no cleaned parquet files at {glob_path}")
        spark.stop()
        return

    df = spark.read.parquet(glob_path)

    # 2) Ensure consistent schema/cols
    for c in UNIFIED_COLS:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(""))
    df = df.select(*UNIFIED_COLS)

    # 3) Heavier prompt normalization
    df = df.withColumn("prompt", normalize_prompt_udf(F.col("prompt"))) \
           .where(F.length("prompt") >= 20) \
           .where(~F.col("prompt").rlike(r"^\s*$"))

    # 4) Global dedupe across datasets (hash of source|title|prompt)
    df = df.withColumn(
        "hash_key",
        F.sha2(F.concat_ws("||", "source", "title", "prompt"), 256)
    ).dropDuplicates(["hash_key"]).drop("hash_key")

    # 5) Stats
    stats = df.agg(
        F.count("*").alias("rows"),
        F.countDistinct("source").alias("sources"),
        F.countDistinct("language").alias("languages")
    ).collect()[0]
    print(f"[spark] rows={stats['rows']}, sources={stats['sources']}, languages={stats['languages']}")

    # 6) Write unioned dataset partitioned by source
    out_dir = (CLEAN_DIR / "_union").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.repartition(1, "source").write.mode("overwrite").partitionBy("source").parquet(str(out_dir))
    print(f"✅ Spark union saved -> {out_dir}")

    # 7) Tiny sample for eyeballing
    sample_csv = out_dir / "sample_50.csv"
    df.limit(50).toPandas().to_csv(sample_csv, index=False)
    print(f"👀 sample -> {sample_csv}")

    spark.stop()

if __name__ == "__main__":
    main()
