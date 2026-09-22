"""미리 빌드해 둔 권역 데이터를 GitHub Releases 에서 받아 푼다.

OSM 추출본을 받거나 빌드를 돌리지 않아도 서버를 띄울 수 있다.

    python src/fetch_release.py                # 받을 수 있는 권역 전부
    python src/fetch_release.py kyushu tokai   # 고른 권역만
    python src/fetch_release.py --list         # 무엇이 있는지만

태그가 "data-" 로 시작하는 릴리스 중 가장 최근 것을 쓴다. 데이터는 그걸 만든
코드와 짝이 맞아야 하므로, 오래된 체크아웃이면 먼저 코드를 최신으로 받는다.
**서버를 멈춘 뒤에 돌린다.** 서버가 보행망 파일을 열어 두면 윈도우가 덮어쓰기를
막는다.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ROOT / "data" / "regions"
REPO = "HuijuKim/rail-isochrone"
API = f"https://api.github.com/repos/{REPO}/releases?per_page=30"


def _get(url: str, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rail-isochrone",
                                               "Accept": accept})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read()


def latest_data_release() -> dict:
    releases = json.loads(_get(API))
    for rel in releases:
        if rel.get("tag_name", "").startswith("data-") and not rel.get("draft"):
            return rel
    sys.exit("데이터 릴리스를 찾지 못했습니다(태그가 data- 로 시작하는 것).")


def main() -> None:
    rel = latest_data_release()
    assets = {a["name"][:-4]: a for a in rel.get("assets", [])
              if a["name"].endswith(".zip")}
    if "--list" in sys.argv:
        print(f"{rel['tag_name']}  ({rel.get('published_at', '')[:10]})")
        for name, a in sorted(assets.items()):
            print(f"  {name:12} {a['size'] / 1e6:7.1f} MB")
        return
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or sorted(assets)
    missing = [r for r in want if r not in assets]
    if missing:
        sys.exit(f"릴리스에 없는 권역: {missing}. 있는 것: {sorted(assets)}")
    print(f"{rel['tag_name']} 에서 받는다", flush=True)
    for region in want:
        base = REGIONS / region
        if not (base / "region.json").exists():
            print(f"  {region}: region.json 이 없어 건너뛴다(코드를 최신으로 받으세요)")
            continue
        a = assets[region]
        print(f"  {region}: {a['size'] / 1e6:.0f} MB 받는 중...", flush=True)
        blob = _get(a["browser_download_url"], accept="application/octet-stream")
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = [n for n in z.namelist() if n != "manifest.json"]
            # 압축 안의 경로가 권역 폴더를 벗어나지 못하게 한다.
            for n in names:
                target = (base / n).resolve()
                if base.resolve() not in target.parents:
                    sys.exit(f"이상한 경로가 들어 있습니다: {n}")
            z.extractall(base, members=names)
        print(f"  {region}: {len(names)}개 파일을 풀었다", flush=True)
    print("끝. python src/server.py 로 띄운다.")


if __name__ == "__main__":
    main()
