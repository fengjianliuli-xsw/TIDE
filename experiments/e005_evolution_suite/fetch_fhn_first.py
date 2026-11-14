"""Retrieve the lowest numbered archived PDEBench trajectory using validated byte ranges."""
import functools
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://darus.uni-stuttgart.de/api/access/datafile/133017"
OUT = Path(__file__).resolve().parent / "data"


class H5RangeReader(io.RawIOBase):
    def __init__(self):
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=0.4)))
        self.size = 13243560688
        self.block_size = 256 * 1024
        self.pos = 0

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    @functools.lru_cache(maxsize=160)
    def block(self, i):
        lo = i * self.block_size
        hi = min(self.size, lo + self.block_size) - 1
        for path in OUT.glob("fhn_head_*.bin"):
            if path.stat().st_size > hi:
                with path.open("rb") as file:
                    file.seek(lo)
                    return file.read(hi - lo + 1)
        r = self.session.get(URL, params={"block": i}, headers={"Range": f"bytes={lo}-{hi}"}, timeout=60)
        r.raise_for_status()
        if r.headers.get("Content-Range") != f"bytes {lo}-{hi}/{self.size}" or len(r.content) != hi-lo+1:
            raise IOError(f"Invalid range for block {i}: {r.headers.get('Content-Range')}")
        return r.content

    def readinto(self, buffer):
        remaining = min(len(buffer), self.size - self.pos)
        out = memoryview(buffer).cast("B")
        written = 0
        while written < remaining:
            block, start = divmod(self.pos, self.block_size)
            src = self.block(block)
            count = min(remaining - written, len(src) - start)
            out[written:written+count] = src[start:start+count]
            written += count
            self.pos += count
        return written


if __name__ == "__main__":
    with h5py.File(H5RangeReader(), "r") as f:
        first = "0000"
        root = f[first]
        with h5py.File(OUT / "fhn_first.h5", "w") as dest:
            group = dest.create_group(first)
            def copy(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(name, obj.shape, flush=True)
                    group.create_dataset(name, data=obj[()])
            root.visititems(copy)
            for key, value in root.attrs.items():
                group.attrs[key] = value
    path = OUT / "fhn_first.h5"
    with h5py.File(path) as f:
        times = np.ravel(f[first]["grid/t"][()])
        print(first, times[[0,-1]], flush=True)
    (OUT / "fhn_source.json").write_text(json.dumps({
        "url": URL, "group": first, "source_file_bytes": 13243560688,
        "sha256_extracted_file": hashlib.sha256(path.read_bytes()).hexdigest()
    }, indent=2))
