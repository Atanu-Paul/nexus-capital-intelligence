"""S3 storage helpers and synthetic data sync CLI."""

import argparse
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from error_handling.retry_utils import retry_transient
from rag.paths import synthetic_data_dir

load_dotenv()

logger = logging.getLogger(__name__)


class S3Storage:
    """S3-backed object storage using the default boto3 credential chain."""

    def __init__(self) -> None:
        self._bucket = os.getenv("S3_BUCKET_NAME", "").strip()
        self._region = os.getenv("AWS_REGION", "").strip() or None
        self._client = None
        if self._bucket:
            self._client = boto3.client("s3", region_name=self._region)

    @property
    def bucket(self) -> str:
        return self._bucket

    def available(self) -> bool:
        """Return True when the bucket is configured and reachable."""
        if not self._client or not self._bucket:
            return False
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return True
        except ClientError:
            return False

    @retry_transient
    def upload_json(self, key: str, obj: dict) -> None:
        if not self._client or not self._bucket:
            raise RuntimeError("S3 is not configured")
        body = json.dumps(obj, indent=2).encode("utf-8")
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )

    @retry_transient
    def upload_file(self, path: str | Path, key: str) -> None:
        if not self._client or not self._bucket:
            raise RuntimeError("S3 is not configured")
        file_path = Path(path)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        extra_args = {"ContentType": content_type}
        self._client.upload_file(
            str(file_path),
            self._bucket,
            key,
            ExtraArgs=extra_args,
        )

    @retry_transient
    def download_json(self, key: str) -> dict:
        if not self._client or not self._bucket:
            raise RuntimeError("S3 is not configured")
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        payload = response["Body"].read().decode("utf-8")
        return json.loads(payload)

    @retry_transient
    def list_keys(self, prefix: str = "") -> list[str]:
        if not self._client or not self._bucket:
            raise RuntimeError("S3 is not configured")
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                keys.append(item["Key"])
        return keys

    @retry_transient
    def sync_dir(self, local_dir: str | Path, prefix: str) -> int:
        if not self._client or not self._bucket:
            raise RuntimeError("S3 is not configured")
        root = Path(local_dir)
        if not root.exists():
            raise FileNotFoundError(f"Local directory not found: {root}")

        uploaded = 0
        for file_path in root.rglob("*"):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            relative = file_path.relative_to(root).as_posix()
            key = f"{prefix.rstrip('/')}/{relative}"
            self.upload_file(file_path, key)
            uploaded += 1
        return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description="S3 utilities for Nexus Capital Intelligence")
    parser.add_argument(
        "--sync-synthetic",
        action="store_true",
        help="Upload synthetic_data/** to the synthetic_data/ S3 prefix",
    )
    args = parser.parse_args()

    storage = S3Storage()
    if not storage.available():
        raise SystemExit("S3 is not configured or bucket is unreachable")

    if args.sync_synthetic:
        count = storage.sync_dir(synthetic_data_dir(), "synthetic_data")
        print(f"Uploaded {count} files to s3://{storage.bucket}/synthetic_data/")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
