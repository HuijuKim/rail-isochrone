"""권역 하나에 필요한 데이터 일습.

지금까지는 경로가 모듈 여섯 군데에 흩어져 있어서 권역을 늘릴 수가 없었다.
한 권역이 쓰는 것들 — 시각표 그래프, 역 목록, 보행망, 노선 선형, 범위 —
을 이 클래스가 한꺼번에 들고 있는다.

데이터는 data/regions/<id>/ 아래에 모인다. OSM 추출본만 빌드 입력이라
data/osm/ 에 공유로 둔다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REGIONS_DIR = ROOT / "data" / "regions"


def available() -> list[str]:
    """준비된 권역 id 목록."""
    if not REGIONS_DIR.exists():
        return []
    return sorted(
        d.name for d in REGIONS_DIR.iterdir()
        if (d / "region.json").exists() and (d / "stops.json").exists()
    )


@dataclass
class Region:
    id: str
    meta: dict
    dir: Path
    stops: dict
    coords: np.ndarray
    graphs: dict          # calendar -> router.Graph
    railways: dict        # railway id -> 제목/색
    walk: object | None   # walknet.WalkNet
    geometry: object      # geometry.Geometry
    coverage: object | None
    coverage_geojson: dict | None
    search_index: list[dict]
    # 역 목록은 노선별로 쪼개져 있다. 신주쿠 하나가 11개 항목이다.
    # 세어 보여줄 때는 사람이 아는 "역" 단위여야 하므로 묶음 번호를 들고 있는다.
    station_group: np.ndarray   # 역 인덱스 -> 묶음 번호 (-1 이면 좌표 없음)
    n_groups: int

    @property
    def name(self) -> str:
        return self.meta.get("name", self.id)


def load(region_id: str) -> Region:
    """권역 하나를 통째로 올린다."""
    import walknet
    from coverage import Coverage
    from geometry import Geometry
    from router import load_graph

    base = REGIONS_DIR / region_id
    meta = json.loads((base / "region.json").read_text(encoding="utf-8"))
    stops = json.loads((base / "stops.json").read_text(encoding="utf-8"))
    coords = np.array(stops["coords"], dtype=np.float64)

    graphs = {
        calendar: load_graph(base / f"graph-{calendar}.npz")
        for calendar in ("Weekday", "SaturdayHoliday")
        if (base / f"graph-{calendar}.npz").exists()
    }

    railways = {
        r["id"]: r
        for r in json.loads((base / "raw" / "railways.json").read_text(encoding="utf-8"))
    }

    walk = walknet.load(base / "walk")
    if walk is not None:
        # 지점 스냅용 KD 트리는 노드가 수백만이라 세우는 데 시간이 걸린다.
        # 첫 요청이 그 값을 물지 않도록 여기서 미리 세운다.
        lon, lat = meta.get("center", [139.7, 35.69])
        walk.nearest_node(lon, lat)

    geometry = Geometry(base / "raw" / "coordinates.json", stops["railway"], coords)
    # 범위는 역 위치에서 잡는다. 보행망 전체로 잡으면 우리가 다루지 않는
    # 노선의 역들까지 선 안에 들어온다.
    cover = (
        Coverage(walk, base / "walk" / "land.npz", stations=coords)
        if walk is not None else None
    )

    index, groups, n_groups = build_search_index(base, stops, coords)

    return Region(
        id=region_id,
        meta=meta,
        dir=base,
        stops=stops,
        coords=coords,
        graphs=graphs,
        railways=railways,
        walk=walk,
        geometry=geometry,
        coverage=cover,
        coverage_geojson=cover.boundary_geojson() if cover is not None else None,
        search_index=index,
        station_group=groups,
        n_groups=n_groups,
    )


def build_search_index(base: Path, stops: dict, coords: np.ndarray):
    """검색창에 쓸 역 목록. 같은 역은 한 줄로 합친다.

    신주쿠처럼 여러 철도사가 들어오는 역은 노선 수만큼 항목이 생기는데,
    이용자 입장에서는 전부 같은 "신주쿠역" 이다. station-groups.json 의
    묶음을 기준으로 합치고, 어느 회사들이 지나는지를 함께 보여준다.
    """
    from collections import Counter

    from operators import operator_of, sort_key, title as operator_title

    groups_path = base / "raw" / "station-groups.json"
    groups = (
        json.loads(groups_path.read_text(encoding="utf-8"))
        if groups_path.exists() else []
    )

    group_of: dict[str, int] = {}
    for gi, group in enumerate(groups):
        for sub in group:
            for sid in sub:
                group_of[sid] = gi

    buckets: dict[object, list[int]] = {}
    for i, sid in enumerate(stops["ids"]):
        if not np.isfinite(coords[i, 0]):
            continue
        # 그룹에 없는 역은 역명 + 대략적인 위치로 묶는다 (100 m 안팎)
        key = group_of.get(sid)
        if key is None:
            key = (stops["ja"][i], round(coords[i, 0], 3), round(coords[i, 1], 3))
        buckets.setdefault(key, []).append(i)

    out = []
    group_of_station = np.full(len(stops["ids"]), -1, dtype=np.int32)
    for gi, members in enumerate(buckets.values()):
        group_of_station[members] = gi
        # 한 그룹 안에 이름이 섞여 있으면 (신주쿠 / 신주쿠니시구치) 다수를 따른다
        members = list(members)
        names = Counter(stops["ja"][i] for i in members)
        ja = names.most_common(1)[0][0]
        lead = next(i for i in members if stops["ja"][i] == ja)

        prefixes = sorted({operator_of(stops["ids"][i]) for i in members}, key=sort_key)
        out.append(
            {
                "id": stops["ids"][lead],
                "lon": float(np.mean(coords[members, 0])),
                "lat": float(np.mean(coords[members, 1])),
                "ja": ja,
                "en": stops["en"][lead],
                "ko": stops["ko"][lead],
                "operators": [operator_title(p, "ko") for p in prefixes],
            }
        )

    out.sort(key=lambda s: s["ja"])
    return out, group_of_station, len(buckets)
