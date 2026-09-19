"""mini-tokyo-3d 리포지토리에서 수도권 철도 시각표 원본을 내려받는다.

데이터 출처: https://github.com/nagix/mini-tokyo-3d (MIT License)
원 출처   : 公共交通オープンデータセンター (ODPT)
"""
from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

REPO = "https://raw.githubusercontent.com/nagix/mini-tokyo-3d/master/data"
INDEX = "https://api.github.com/repos/nagix/mini-tokyo-3d/contents/data/train-timetables"

REGION = os.environ.get("REGION", "kanto")
RAW = Path(__file__).resolve().parent.parent / "data" / "regions" / REGION / "raw"
TIMETABLES = RAW / "train-timetables"

# 노선/역/운영사 등 참조 데이터
REFERENCE_FILES = [
    "stations.json",
    "railways.json",
    "operators.json",
    "station-groups.json",
    "rail-directions.json",
    "train-types.json",
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tokyo-isochrone/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _download(url: str, dest: Path) -> tuple[str, int]:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.name, dest.stat().st_size
    data = _get(url)
    dest.write_bytes(data)
    return dest.name, len(data)


def main() -> None:
    TIMETABLES.mkdir(parents=True, exist_ok=True)

    for name in REFERENCE_FILES:
        n, size = _download(f"{REPO}/{name}", RAW / name)
        print(f"ref  {n:24} {size:>10,} B", flush=True)

    listing = json.loads(_get(INDEX))
    names = [x["name"] for x in listing if x["name"].endswith(".json")]
    print(f"\n시각표 {len(names)}개 수집 시작", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(_download, f"{REPO}/train-timetables/{n}", TIMETABLES / n)
            for n in names
        ]
        for fut in futures:
            name, size = fut.result()
            done += 1
            print(f"  [{done:3}/{len(names)}] {name:36} {size:>10,} B", flush=True)

    total = sum(f.stat().st_size for f in TIMETABLES.glob("*.json"))
    print(f"\n완료: {done}개 파일, {total / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
