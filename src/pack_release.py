"""빌드해 둔 권역 데이터를 GitHub Releases 에 올릴 zip 으로 묶는다.

서버가 읽는 파일만 담는다. 빌드에만 쓰는 캐시(OSM 추출, 행정경계, 해안선
원본, 위키 표 원본)는 뺀다. 권역 하나가 zip 하나다.

    python src/pack_release.py              # 시각표 없는 권역 전부 -> dist/
    python src/pack_release.py kyushu tokai # 고른 권역만

간토 시각표 권역(kanto)은 담지 않는다. ODPT 시각표의 재배포 조건을 확인하지
못했다(README 의 라이선스 절). 같은 범위를 OSM 으로 만든 kanto_osm 을 쓴다.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ROOT / "data" / "regions"
DIST = ROOT / "dist"

# region.json 은 저장소에 있으므로 담지 않는다. 받는 쪽은 코드와 짝이 맞아야 한다.
RUNTIME = [
    "stops.json",
    "graph-Weekday.npz",
    "graph-SaturdayHoliday.npz",
    "prefectures.json",
    "raw/railways.json",
    "raw/stations.json",
    "raw/station-groups.json",
    "raw/coordinates.json",
    "raw/coordinates-osm.json",
    "raw/track-segments.json",
    "raw/prefecture-rings.json",
    "raw/prefecture-outline.json",
    "raw/line-names.json",
    "walk/graph.npz",
    "walk/sheds.npz",
    "walk/land.npz",
    "walk/fine_grid.npy",
    "walk/fine_pt.npy",
    "walk/fine_seg.npy",
    "walk/fine_ptr.npy",
]


def packable() -> list[str]:
    out = []
    for d in sorted(REGIONS.iterdir()):
        meta = d / "region.json"
        if not meta.exists() or not (d / "stops.json").exists():
            continue
        if json.loads(meta.read_text(encoding="utf-8")).get("model") != "naive":
            continue
        out.append(d.name)
    return out


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def pack(region: str) -> Path:
    base = REGIONS / region
    DIST.mkdir(exist_ok=True)
    out = DIST / f"{region}.zip"
    manifest = {"region": region, "built": date.today().isoformat(),
                "code": commit(), "files": []}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        for rel in RUNTIME:
            p = base / rel
            if p.exists():
                z.write(p, rel)
                manifest["files"].append(rel)
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    return out


def main() -> None:
    want = sys.argv[1:] or packable()
    bad = [r for r in want if r not in packable()]
    if bad:
        sys.exit(f"묶을 수 없는 권역: {bad} (시각표 없는 권역 중 빌드를 마친 것만)")
    for region in want:
        out = pack(region)
        print(f"{region:12} {out.stat().st_size / 1e6:7.1f} MB  -> {out}", flush=True)


if __name__ == "__main__":
    main()
