"""Download and validate the immutable raw experiment archive, then safely extract it."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
META=json.loads((ROOT/"manifests"/"release.json").read_text())
URL=f"https://github.com/fengjianliuli-xsw/TIDE/releases/download/{META['tag']}/{META['asset']}"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",type=Path,help="Use an already downloaded archive")
    args=ap.parse_args()
    path=args.archive or ROOT/META["asset"]
    if not path.exists():
        if args.archive:raise FileNotFoundError(path)
        print("Downloading",URL,flush=True)
        urllib.request.urlretrieve(URL,path)
    with path.open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    if digest!=META["sha256"]:raise ValueError("Archive SHA256 differs from release manifest")
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            dest=(ROOT/member.filename).resolve()
            if not dest.is_relative_to(ROOT) or not member.filename.startswith("experiments/"):
                raise ValueError("Unsafe archive entry")
        archive.extractall(ROOT)
    print("Validated and restored experiment artifacts")


if __name__=="__main__":main()
