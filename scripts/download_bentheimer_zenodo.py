from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "core1_subvol1_18um": {
        "url": "https://zenodo.org/records/5542624/files/Core1_Subvol1_18micron_75cube_16bit_LE.raw?download=1",
        "path": ROOT / "data" / "raw" / "Core1_Subvol1_18micron_75cube_16bit_LE.raw",
        "md5": "a10d9807dbb234f90a1d5b3ff85dd3f4",
    },
    "core1_subvol1_6um": {
        "url": "https://zenodo.org/records/5542624/files/Core1_Subvol1_6micron_225cube_16bit_LE.raw?download=1",
        "path": ROOT / "data" / "raw" / "Core1_Subvol1_6micron_225cube_16bit_LE.raw",
        "md5": "80c1995a8ee9a14a97cdc752c64661b7",
    },
    "core1_subvol2_6um": {
        "url": "https://zenodo.org/records/5542624/files/Core1_Subvol2_6micron_225cube_16bit_LE.raw?download=1",
        "path": ROOT / "data" / "raw" / "Core1_Subvol2_6micron_225cube_16bit_LE.raw",
        "md5": "53ec6e1425e5d45e4d9726c1c503b575",
    },
    "core2_subvol1_6um": {
        "url": "https://zenodo.org/records/5542624/files/Core2_Subvol1_6micron_225cube_16bit_LE.raw?download=1",
        "path": ROOT / "data" / "raw" / "Core2_Subvol1_6micron_225cube_16bit_LE.raw",
        "md5": "d5e3659459e612a0c17ed5b328fffa04",
    },
    "core2_subvol2_6um": {
        "url": "https://zenodo.org/records/5542624/files/Core2_Subvol2_6micron_225cube_16bit_LE.raw?download=1",
        "path": ROOT / "data" / "raw" / "Core2_Subvol2_6micron_225cube_16bit_LE.raw",
        "md5": "902317cea1b7c3c4ad818e46b94c1824",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small Bentheimer sandstone CT volume from Zenodo.")
    parser.add_argument("--file", choices=FILES, default="core1_subvol1_18um")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = FILES[args.file]
    path = info["path"]
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not args.overwrite:
        print(f"Already exists: {path}")
    else:
        print(f"Downloading {info['url']}")
        urllib.request.urlretrieve(info["url"], path)

    digest = md5(path)
    if digest != info["md5"]:
        raise RuntimeError(f"MD5 mismatch for {path}: expected {info['md5']}, got {digest}")
    print(f"Verified: {path}")


def md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    main()
