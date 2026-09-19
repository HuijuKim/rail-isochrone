"""역마다 어느 도도부현에 있는지 붙인다.

mini-tokyo-3d 의 역 데이터에는 현 정보가 없다. OSM 추출본에는 행정경계가
들어 있으므로(admin_level=4 가 도도부현) 거기서 다각형을 받아 역 좌표를
넣어 본다.

다각형은 실행 중에 쓸 일이 없다. 역 -> 현 이름만 남기고 버린다.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REGION = os.environ.get("REGION", "kanto")
BASE = DATA / "regions" / REGION
# 읽을 추출본은 build_walk 와 같은 목록을 쓴다. 야마나시·시즈오카 경계는
# 주부 추출본에만 있다.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_walk import PBFS  # noqa: E402

# 도도부현
ADMIN_LEVEL = "4"


def prefecture_polygons() -> dict[str, list[np.ndarray]]:
    """현 이름 -> 닫힌 고리들.

    osmium 의 면 조립(with_areas)은 관계의 멤버가 하나라도 빠지면 면을 만들지
    않는다. 도쿄도 경계에는 이즈 제도와 오가사와라가 들어 있는데 그 way 들이
    간토 추출본 밖이라, 도쿄도가 통째로 빠진다. 시즈오카·야마나시처럼 추출본
    가장자리에서 잘린 현도 마찬가지다.

    그래서 멤버 way 를 직접 받아 이어 붙인다. 섬 쪽 조각이 없어도 본토 고리는
    제 힘으로 닫히므로, 닫힌 고리만 남기면 된다.
    """
    import osmium

    t0 = time.time()

    # 1) 어느 way 가 어느 현에 속하는지
    # way id -> 그 way 를 경계로 쓰는 현들. 같은 관계가 두 추출본에
    # 들어 있으면 이름이 겹치므로 집합으로 받는다.
    want: dict[int, set[str]] = {}
    names: set[str] = set()
    labels: dict[str, dict[str, str]] = {}
    for path, rel in ((p, r) for p in PBFS
                      for r in osmium.FileProcessor(str(p)).with_filter(
                          osmium.filter.EntityFilter(osmium.osm.RELATION))):
        tags = rel.tags
        if tags.get("boundary") != "administrative":
            continue
        if tags.get("admin_level") != ADMIN_LEVEL:
            continue
        name = tags.get("name") or tags.get("name:ja")
        if not name:
            continue
        names.add(name)
        labels[name] = {
            "ja": name,
            "ko": tags.get("name:ko") or name,
            "en": tags.get("name:en") or name,
        }
        for member in rel.members:
            if member.type == "w":
                want.setdefault(member.ref, set()).add(name)
    print(f"  현 {len(names)}개, 경계 way {len(want):,}개 ({time.time() - t0:.0f}s)", flush=True)

    # 2) 그 way 들의 좌표.
    #
    # way id 는 전역이라 두 추출본이 겹치는 영역의 way 가 두 번 들어온다.
    # 같은 조각이 둘이면 조각과 그 복제본이 서로 맞물려 2개짜리 가짜 고리를
    # 만들고, 정작 본토 고리는 못 닫힌다. id 로 한 번만 받는다.
    pieces: dict[str, list[np.ndarray]] = {n: [] for n in names}
    taken: set[int] = set()
    for path, way in ((p, w) for p in PBFS
                      for w in osmium.FileProcessor(str(p)).with_locations().with_filter(
                          osmium.filter.EntityFilter(osmium.osm.WAY))):
        owners = want.get(way.id)
        if not owners or way.id in taken:
            continue
        pts = [(n.location.lon, n.location.lat) for n in way.nodes if n.location.valid()]
        if len(pts) < 2:
            continue
        taken.add(way.id)
        arr = np.asarray(pts, dtype=np.float64)
        for name in owners:
            pieces[name].append(arr)

    # 3) 조각을 이어 고리로
    out: dict[str, list[np.ndarray]] = {}
    for name, parts in pieces.items():
        rings = _stitch(parts)
        if rings:
            out[name] = rings
    print(f"  닫힌 고리 {sum(len(v) for v in out.values())}개 "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out, {k: v for k, v in labels.items() if k in out}


def _stitch(parts: list[np.ndarray], tol: float = 1e-7) -> list[np.ndarray]:
    """끝점이 맞는 조각들을 이어 닫힌 고리를 만든다. 못 닫으면 버린다."""
    from collections import defaultdict

    ends = defaultdict(list)

    def key(p):
        return (round(float(p[0]) / tol), round(float(p[1]) / tol))

    for i, part in enumerate(parts):
        ends[key(part[0])].append(i)
        ends[key(part[-1])].append(i)

    used = [False] * len(parts)
    rings: list[np.ndarray] = []
    for start in range(len(parts)):
        if used[start]:
            continue
        used[start] = True
        chain = [parts[start]]
        head, tail = key(parts[start][0]), key(parts[start][-1])
        # 꼬리에 붙일 조각을 계속 찾는다
        while tail != head:
            nxt = None
            for j in ends.get(tail, ()):
                if used[j]:
                    continue
                nxt = j
                break
            if nxt is None:
                break
            used[nxt] = True
            piece = parts[nxt]
            if key(piece[0]) != tail:
                piece = piece[::-1]
            chain.append(piece[1:])
            tail = key(piece[-1])
        if tail == head:
            ring = np.concatenate(chain)
            if len(ring) >= 4:
                rings.append(ring)
    return rings


def assign(polygons: dict[str, list[np.ndarray]], coords: np.ndarray) -> list[str | None]:
    """역 좌표를 현 다각형에 넣어 본다.

    현 하나에 고리가 수백 개씩(섬) 나오므로, 먼저 경계상자로 거르고 남은
    것만 포함 검사를 한다. 그러지 않으면 역 2,610개 x 고리 수천 개가 된다.
    """
    from matplotlib.path import Path as MplPath

    entries = []
    for name, rings in polygons.items():
        for ring in rings:
            entries.append((name, ring[:, 0].min(), ring[:, 0].max(),
                            ring[:, 1].min(), ring[:, 1].max(), MplPath(ring)))
    # 큰 고리부터 본다. 본토가 먼저 걸리면 섬 수백 개를 건너뛴다.
    entries.sort(key=lambda e: -((e[2] - e[1]) * (e[4] - e[3])))

    out: list[str | None] = []
    for lon, lat in coords:
        hit = None
        if np.isfinite(lon):
            for name, x0, x1, y0, y1, path in entries:
                if not (x0 <= lon <= x1 and y0 <= lat <= y1):
                    continue
                if path.contains_point((lon, lat)):
                    hit = name
                    break
        out.append(hit)
    return out


def main() -> None:
    missing = [p for p in PBFS if not p.exists()]
    if missing:
        sys.exit("OSM 추출본이 없습니다: " + ", ".join(str(p) for p in missing))
    stops_path = BASE / "stops.json"
    if not stops_path.exists():
        sys.exit(f"역 목록이 없습니다: {stops_path}")

    print("1) 행정경계 추출", flush=True)
    polygons, labels = prefecture_polygons()
    if not polygons:
        sys.exit("admin_level=4 경계를 찾지 못했습니다.")

    print("2) 역에 현 붙이기", flush=True)
    stops = json.loads(stops_path.read_text(encoding="utf-8"))
    coords = np.array(stops["coords"], dtype=np.float64)
    names = assign(polygons, coords)

    found = sum(1 for n in names if n)
    payload = {
        "names": labels,
        "stations": {sid: n for sid, n in zip(stops["ids"], names) if n},
    }
    (BASE / "prefectures.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8"
    )

    from collections import Counter
    tally = Counter(n for n in names if n)
    print(f"  {found:,} / {len(names):,} 건에 현을 붙였다", flush=True)
    for name, count in tally.most_common():
        print(f"    {name:<8} {count:>5}", flush=True)


if __name__ == "__main__":
    main()
