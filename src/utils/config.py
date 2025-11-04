from pathlib import Path
from dotenv import load_dotenv
import os

# Project root = two levels up from this file (src/utils → src → project root)
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
LOG_DIR = ROOT / "logs"

for d in (RAW_DIR, CLEAN_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
