"""Download official CIFAR-10 with range requests and verify its published MD5."""
import concurrent.futures
import hashlib
from pathlib import Path
import tarfile
import urllib.request

URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
SIZE = 170498071
MD5 = "c58f30108f718f92721af3b95e74349a"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "experiments" / "downloads"
SEGMENTS = 8


def fetch(index):
    start, stop = index * SIZE // SEGMENTS, (index + 1) * SIZE // SEGMENTS - 1
    path = DEST / f"cifar10.part{index}"
    if path.exists() and path.stat().st_size == stop - start + 1:
        return path
    existing = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start+existing}-{stop}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206:
            raise RuntimeError("Official server did not honor range request")
        with path.open("ab") as output:
            while block := response.read(64 * 1024):
                output.write(block)
                output.flush()
    if path.stat().st_size != stop - start + 1:
        raise RuntimeError("Incomplete range response")
    print(f"Downloaded segment {index + 1}/{SEGMENTS}", flush=True)
    return path


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=SEGMENTS) as pool:
        parts = list(pool.map(fetch, range(SEGMENTS)))
    archive = DEST / "cifar-10-python.tar.gz"
    digest = hashlib.md5()
    with archive.open("wb") as output:
        for path in parts:
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
    if digest.hexdigest() != MD5:
        raise RuntimeError(f"CIFAR archive checksum failed: {digest.hexdigest()}")
    with tarfile.open(archive) as source:
        source.extractall(ROOT / "code" / "data", filter="data")
    print(f"Verified official CIFAR-10 archive: MD5={MD5}", flush=True)


if __name__ == "__main__":
    main()
