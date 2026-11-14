"""Verify source/results and optional original artifact hashes."""
from pathlib import Path
import hashlib
import json
import argparse

ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--artifacts",action="store_true");args=ap.parse_args()
    name="original_artifacts" if args.artifacts else "published_sources"
    rows=json.loads((ROOT/"manifests"/(name+".json")).read_text(encoding="utf-8"))
    for row in rows:
        path=ROOT/("experiments/" if args.artifacts else "")/row["path"]
        assert path.stat().st_size==row["size"],path
        with path.open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest()==row["sha256"],path
    print("Verified",len(rows),"files")
