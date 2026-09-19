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

# 화면이 고를 수 있는 언어. 역 이름을 이만큼 내보낸다.
LANGS = ("ja", "en", "ko", "zh-Hans", "zh-Hant")


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
    fine: object | None   # finegeom.FineGeometry (선을 도로에 붙일 때만)
    supported: np.ndarray  # 묶음 번호 -> 도보권이 있어 실제로 쓸 수 있는가
    railway_shapes: list   # 지도에 겹쳐 그릴 노선 선형 (단순화된 것)
    pref_names: dict       # 도도부현 일본어 이름 -> 언어별 표기
    geometry: object      # geometry.Geometry
    coverage: object | None
    coverage_geojson: dict | None
    search_index: list[dict]
    # 역 목록은 노선별로 쪼개져 있다. 신주쿠 하나가 11개 항목이다.
    # 세어 보여줄 때는 사람이 아는 "역" 단위여야 하므로 묶음 번호를 들고 있는다.
    station_group: np.ndarray   # 역 인덱스 -> 묶음 번호 (-1 이면 좌표 없음)
    n_groups: int

    @property
    def names(self) -> dict:
        """권역 이름을 언어별로. 없는 언어는 id 로 때운다."""
        listed = self.meta.get("names") or {}
        return {lang: listed.get(lang) or listed.get("ja") or self.id for lang in LANGS}

    @property
    def name(self) -> str:
        """로그나 오류 메시지용. 화면은 names 를 쓴다."""
        return self.names.get("ja", self.id)


def load(region_id: str) -> Region:
    """권역 하나를 통째로 올린다."""
    import finegeom
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
    fine = finegeom.load(base / "walk")
    geometry = Geometry(base / "raw" / "coordinates.json", stops["railway"], coords)

    index, groups, n_groups = build_search_index(base, stops, coords)
    supported = supported_mask(walk, coords, groups, n_groups)

    # 역 -> 도도부현. build_admin.py 가 OSM 행정경계로 만들어 둔다.
    pref_path = base / "prefectures.json"
    pref_data = (
        json.loads(pref_path.read_text(encoding="utf-8")) if pref_path.exists() else {}
    )
    prefectures = pref_data.get("stations", {})
    pref_names = pref_data.get("names", {})
    row_of = {sid: i for i, sid in enumerate(stops["ids"])}
    for entry in index:
        i = row_of.get(entry["id"], -1)
        g = int(groups[i]) if i >= 0 else -1
        entry["supported"] = bool(supported[g]) if g >= 0 else False
        entry["pref"] = prefectures.get(entry["id"])

    # 범위는 역 위치에서 잡는다. 보행망 전체로 잡으면 우리가 다루지 않는
    # 노선의 역들까지 선 안에 들어온다. 실제로 쓸 수 있는 역만, 노선별 중복을
    # 합쳐서 넘긴다. 중복이 섞이면 이웃 거리가 0 이 되어 반경이 무너진다.
    cover = None
    if walk is not None:
        pts, ends = coverage_seeds(base, stops, coords, groups, n_groups, supported)
        cover = Coverage(walk, base / "walk" / "land.npz",
                         stations=pts, line_ends=ends)

    return Region(
        id=region_id,
        meta=meta,
        dir=base,
        stops=stops,
        coords=coords,
        graphs=graphs,
        railways=railways,
        walk=walk,
        fine=fine,
        geometry=geometry,
        coverage=cover,
        coverage_geojson=cover.boundary_geojson() if cover is not None else None,
        search_index=index,
        station_group=groups,
        n_groups=n_groups,
        supported=supported,
        railway_shapes=railway_shapes(geometry, railways, stops, coords, groups,
                                      supported, prefectures),
        pref_names=pref_names,
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
                **{lang: (stops[lang][lead] if lang != "ja" else ja)
                   for lang in LANGS if lang in stops},
                "operators": [operator_title(p, "ko") for p in prefixes],
            }
        )

    out.sort(key=lambda s: s["ja"])
    return out, group_of_station, len(buckets)


def supported_mask(walk, coords: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """역 묶음별로 "실제로 쓸 수 있는가".

    OSM 추출본이 간토뿐이라 야마나시·이즈·시즈오카의 역은 보행망에 붙지
    못한다. 그런 역은 도보권이 없어 권역에 아무것도 칠하지 못하고, 출발지로
    찍어도 계산이 안 된다. 지도에 "지원 역" 으로 점을 찍으면 거짓말이 된다.
    """
    ok = np.zeros(max(n_groups, 1), dtype=bool)
    if walk is None:
        return ok
    for i in range(len(coords)):
        g = int(groups[i])
        if g < 0 or ok[g]:
            continue
        if int(walk.station_node[i]) >= 0 and len(walk.shed(i)[0]) > 0:
            ok[g] = True
    return ok


def coverage_seeds(base: Path, stops: dict, coords: np.ndarray, groups: np.ndarray,
                   n_groups: int, supported: np.ndarray):
    """권역을 칠할 씨앗 점과, 각 점이 노선의 끝인지.

    끝인지는 노선의 정차역 목록에서 본다. 어떤 노선의 첫 역이거나 마지막
    역이면서, 다른 노선의 중간역이 아닌 역이 끝이다. 예전에는 이웃 역의
    방향이 한쪽으로 쏠렸는지로 짐작했는데, 그러면 망이 성겨지는 외곽의
    중간역(하스다, 가모노미야, 네부카와)까지 끝으로 잡혔다.
    """
    railways = json.loads((base / "raw" / "railways.json").read_text(encoding="utf-8"))
    index = {sid: i for i, sid in enumerate(stops["ids"])}

    ends: set[int] = set()
    middles: set[int] = set()
    for railway in railways:
        order = railway.get("stations") or []
        if len(order) < 2:
            continue
        for sid, is_end in [(order[0], True), (order[-1], True)]:
            i = index.get(sid)
            if i is not None and groups[i] >= 0:
                ends.add(int(groups[i]))
        for sid in order[1:-1]:
            i = index.get(sid)
            if i is not None and groups[i] >= 0:
                middles.add(int(groups[i]))
    line_ends = ends - middles

    sums = np.zeros((n_groups, 2))
    counts = np.zeros(n_groups)
    for i in range(len(coords)):
        g = int(groups[i])
        if g < 0 or not supported[g]:
            continue
        sums[g] += coords[i]
        counts[g] += 1

    keep = counts > 0
    pts = sums[keep] / counts[keep, None]
    flags = np.array([g in line_ends for g in np.flatnonzero(keep)], dtype=bool)
    return pts, flags


# 지도에 그릴 때 허용할 선형 오차. 25 m 면 7만 6천 점이 7천 점으로 준다.
SHAPE_TOLERANCE_M = 25.0


def _simplify(line: np.ndarray, tol_m: float) -> np.ndarray:
    """더글러스-포이커. 선 모양을 지키면서 점 수를 줄인다."""
    if len(line) < 3:
        return line
    scale = np.cos(np.radians(float(line[:, 1].mean())))
    keep = np.zeros(len(line), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(line) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = line[a], line[b]
        seg = line[a + 1:b]
        dx = (q[0] - p[0]) * scale * 111_320.0
        dy = (q[1] - p[1]) * 111_132.0
        length = np.hypot(dx, dy)
        sx = (seg[:, 0] - p[0]) * scale * 111_320.0
        sy = (seg[:, 1] - p[1]) * 111_132.0
        if length < 1e-9:
            dist = np.hypot(sx, sy)
        else:
            dist = np.abs(sx * dy - sy * dx) / length
        i = int(np.argmax(dist))
        if dist[i] > tol_m:
            keep[a + 1 + i] = True
            stack.append((a, a + 1 + i))
            stack.append((a + 1 + i, b))
    return line[keep]


def railway_shapes(geometry, railways: dict, stops: dict, coords: np.ndarray,
                   groups: np.ndarray, supported: np.ndarray,
                   prefectures: dict) -> list[dict]:
    """지도에 겹쳐 그릴 노선 선형.

    노선이 가진 선형을 그대로 쓰면 안 된다. 쇼난신주쿠라인이나 호쿠소선처럼
    남의 선로를 빌려 쓰는 계통은 자체 선형이 몇 점짜리 껍데기여서, 그리면
    제 역에서 15 km 떨어진 직선이 지도를 가로지른다. 대신 정차역을 차례로
    이어 구간마다 촘촘한 선형을 고르는 ride_path 를 쓴다. 이 길은 경로를
    그릴 때 이미 쓰고 있어 검증돼 있다.

    쓸 수 없는 역만 있는 노선(야마나시의 후지큐, 이즈의 이즈큐)은 뺀다.
    지원 범위를 보여주는 것이 목적인데 넣으면 거짓말이 된다.
    """
    row_of = {sid: i for i, sid in enumerate(stops["ids"])}

    out = []
    for rid, railway in railways.items():
        order = railway.get("stations") or []
        rows = [row_of[s] for s in order
                if s in row_of and np.isfinite(coords[row_of[s], 0])]
        if len(rows) < 2:
            continue
        if not any(groups[i] >= 0 and supported[int(groups[i])] for i in rows):
            continue

        path = geometry.ride_path(rows)
        if len(path) < 2:
            continue

        title = railway.get("title", {})
        seen = {prefectures.get(order[k]) for k in range(len(order))}
        out.append(
            {
                "id": rid,
                **{lang: title.get(lang, "") or title.get("ja", "")
                   for lang in LANGS},
                "color": railway.get("color", "#888888"),
                "prefs": sorted(p for p in seen if p),
                "path": _simplify(np.asarray(path, dtype=np.float64),
                                  SHAPE_TOLERANCE_M).round(5).tolist(),
            }
        )
    out.sort(key=lambda r: r["ja"])
    return out
