import argparse
import os
import hashlib
from pathlib import Path
from typing import Optional, Iterable, Tuple

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from src.utils.config import ROOT

# load env from project root
load_dotenv(ROOT / ".env")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "").strip("/")


def _client():
    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET):
        raise RuntimeError(
            "Missing AWS env vars. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET in .env"
        )
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def _key_for(local_path: Path, base_prefix: Optional[str] = None) -> str:
    rel = local_path.resolve().relative_to(ROOT)
    prefix = (base_prefix.strip("/") if base_prefix is not None else S3_PREFIX)
    return (f"{prefix}/{rel}".strip("/") if prefix else str(rel)).replace("\\", "/")


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- single-file ops ----
def upload_file(local_path: Path, key: Optional[str] = None) -> str:
    s3 = _client()
    local_path = local_path.resolve()
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    obj_key = key or _key_for(local_path)
    print(f"⬆️  Upload {local_path} → s3://{S3_BUCKET}/{obj_key}")
    s3.upload_file(str(local_path), S3_BUCKET, obj_key)
    return obj_key


def download_file(key: str, dest: Path):
    s3 = _client()
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Download s3://{S3_BUCKET}/{key} → {dest}")
    s3.download_file(S3_BUCKET, key, str(dest))


def list_prefix(prefix: str) -> Iterable[Tuple[str, int]]:
    s3 = _client()
    token = None
    while True:
        kwargs = {"Bucket": S3_BUCKET, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            yield obj["Key"], obj["Size"]
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break


# ---- folder sync (up) ----
def sync_dir(local_dir: Path, prefix: Optional[str] = None, dry_run: bool = False, only_changed: bool = True):
    """
    Upload a local directory tree to S3.
    - prefix: override S3_PREFIX for this sync (appended with the local relative paths).
    - only_changed: skip upload if remote size matches.
    """
    _client()  # validate creds early
    local_dir = local_dir.resolve()
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    base_prefix = prefix.strip("/") if prefix else S3_PREFIX
    print(f"🔁 Sync {local_dir} → s3://{S3_BUCKET}/{base_prefix or ''}")

    remote_index = {k: sz for k, sz in list_prefix(base_prefix)} if only_changed else {}

    s3 = _client()
    uploaded = 0
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        key = _key_for(p, base_prefix)
        size = p.stat().st_size
        if only_changed and remote_index.get(key) == size:
            print(f"↪︎ skip (same size) {key}")
            continue
        msg = f"⬆︎ {p} → s3://{S3_BUCKET}/{key}"
        print(("DRY-RUN " if dry_run else "") + msg)
        if not dry_run:
            s3.upload_file(str(p), S3_BUCKET, key)
            uploaded += 1

    print(f"✅ Sync complete. Uploaded {uploaded} file(s).")


# ---- folder sync (down) ----
def sync_down(prefix: str, local_dir: Path, dry_run: bool = False):
    """
    Download all objects under S3 'prefix' into local_dir, recreating folders.
    Example:
      sync_down("education_agent/data/clean/_union", Path("downloads/_union"))
    """
    _client()  # validate creds early
    local_dir = local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    prefix_clean = prefix.strip("/")
    print(f"🔁 Sync-down s3://{S3_BUCKET}/{prefix_clean} → {local_dir}")

    s3 = _client()
    count = 0
    for key, _size in list_prefix(prefix_clean):
        # keep structure after the given prefix
        rel = key[len(prefix_clean):].lstrip("/")
        dest = (local_dir / rel).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        msg = f"⬇︎ s3://{S3_BUCKET}/{key} → {dest}"
        print(("DRY-RUN " if dry_run else "") + msg)
        if not dry_run:
            s3.download_file(S3_BUCKET, key, str(dest))
            count += 1

    print(f"✅ Sync-down complete. Downloaded {count} file(s).")


# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="S3 upload/download/list/sync helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload-file")
    up.add_argument("local_path")
    up.add_argument("--key")

    down = sub.add_parser("download-file")
    down.add_argument("key")
    down.add_argument("dest")

    ls = sub.add_parser("list")
    ls.add_argument("--prefix", default=S3_PREFIX)

    sync = sub.add_parser("sync-dir")
    sync.add_argument("local_dir")
    sync.add_argument("--prefix", default=None)
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--no-only-changed", action="store_true")

    downsync = sub.add_parser("sync-down")
    downsync.add_argument("prefix", help="S3 prefix to download (e.g., education_agent/data/clean/_union)")
    downsync.add_argument("local_dir", help="Local destination directory")
    downsync.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    if args.cmd == "upload-file":
        key = upload_file(Path(args.local_path), args.key)
        print(f"✅ Uploaded to s3://{S3_BUCKET}/{key}")

    elif args.cmd == "download-file":
        download_file(args.key, Path(args.dest))
        print("✅ Downloaded")

    elif args.cmd == "list":
        for k, sz in list_prefix(args.prefix.strip("/")):
            print(f"{k}\t{sz}")

    elif args.cmd == "sync-dir":
        sync_dir(
            Path(args.local_dir),
            prefix=args.prefix,
            dry_run=args.dry_run,
            only_changed=not args.no_only_changed,
        )

    elif args.cmd == "sync-down":
        sync_down(args.prefix, Path(args.local_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
