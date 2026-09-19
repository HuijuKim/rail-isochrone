"""OSM 간토 추출본에서 보행 네트워크를 뽑아 역별 도보권을 미리 계산한다.

직선거리 x 1.3 근사로는 강이나 철도 부지처럼 건널 수 없는 곳을 그냥 가로질러
버린다. 실제 보행로를 따라 걷게 하려면 매 질의마다 수백만 노드 위에서
다익스트라를 돌려야 하는데 그러면 10초가 넘는다.

보행망은 출발 시각에 따라 변하지 않는다는 점을 쓴다. 역마다 도보권을 한 번만
계산해 격자 셀 단위로 저장해두면, 질의할 때는 도달한 역들의 도보권을 겹쳐
최솟값만 취하면 된다.

출력 (data/walk/):
  graph.npz    보행 그래프 (CSR 형식) + 노드 좌표
  sheds.npz    역별 도보권 (격자 셀 번호와 도보 시간)
"""
from __future__ import annotations

import sys
import time
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REGION = os.environ.get("REGION", "kanto")
BASE = DATA / "regions" / REGION
# 어느 OSM 추출본을 쓸지는 권역 설정에 적는다. 여럿을 겹쳐 읽는다.
# 야마나시와 이즈는 Geofabrik 이 주부로 분류해서 간토 추출본에 없다.
# 격자 밖의 점은 어차피 버리므로 나고야까지 읽어도 결과는 같다.
def _pbf_list() -> list[Path]:
    names = os.environ.get("PBF")
    if names:
        return [DATA / "osm" / n.strip() for n in names.split(",") if n.strip()]
    meta_path = BASE / "region.json"
    if meta_path.exists():
        import json as _json
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        listed = meta.get("osm_files")
        if listed:
            return [DATA / "osm" / n for n in listed]
    return [DATA / "osm" / "kanto-latest.osm.pbf"]


PBFS = _pbf_list()
OUT = BASE / "walk"

# 보행 가능한 도로 종류. 자동차 전용도로는 뺀다.
WALKABLE = {
    "footway", "path", "pedestrian", "steps", "corridor", "living_street",
    "residential", "unclassified", "service", "track", "road",
    "tertiary", "tertiary_link", "secondary", "secondary_link",
    "primary", "primary_link", "trunk", "trunk_link", "cycleway",
}
# 계단은 같은 거리라도 더 오래 걸린다
STEP_PENALTY = 2.2
# 이 크기 미만의 연결 성분은 주변과 끊긴 파편으로 보고 버린다
MIN_COMPONENT_NODES = 50

# 도보 속도는 router.py 가 기준이다. 여기서 따로 정의하면 사전 계산된 도보권과
# 실행 중 계산이 서로 다른 속도로 돌게 된다.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from router import WALK_SPEED  # noqa: E402

MAX_SHED_SEC = 40 * 60            # 화면 슬라이더의 최대치와 맞춘다

# 그래프용 격자는 촘촘해야 한다. 100 m 로 묶으면 폭 20-40 m 짜리 철도 부지나
# 좁은 하천의 양안 도로가 같은 노드로 합쳐져, 있지도 않은 건널목이 생긴다.
CELL_M = 40.0
# 저장·렌더링용 격자는 성겨도 된다. 도보권을 이 해상도로 내려 담아야 용량과
# 질의 속도가 감당된다.
SHED_CELL_M = 100.0

# 격자 원점 (간토 남서쪽 귀퉁이). 모든 도보권이 같은 격자를 공유해야
# 질의할 때 셀 번호로 바로 겹칠 수 있다.
# 이즈 반도 남단(34.679)과 고후·오마에(138.53) 까지 담아야 한다
GRID_LON0, GRID_LAT0 = 138.38, 34.53
GRID_LAT_REF = 35.7
M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(GRID_LAT_REF))
GRID_SPAN_X = 440_000.0    # 동서
GRID_SPAN_Y = 330_000.0    # 남북
GRID_W = int(np.ceil(GRID_SPAN_X / CELL_M))
GRID_H = int(np.ceil(GRID_SPAN_Y / CELL_M))
SHED_W = int(np.ceil(GRID_SPAN_X / SHED_CELL_M))
SHED_H = int(np.ceil(GRID_SPAN_Y / SHED_CELL_M))
# 격자가 덮는 위경도 범위. 추출본을 읽을 때 밖은 버린다.
GRID_LON1 = GRID_LON0 + GRID_SPAN_X / M_PER_DEG_LON
GRID_LAT1 = GRID_LAT0 + GRID_SPAN_Y / M_PER_DEG_LAT


# 셀 번호와 CSR 포인터를 int32 로 담는다. 611만 노드에서 90 MB 가 줄어든다.
# 다만 넘치면 조용히 뒤틀리므로 빌드 때 한 번 확인한다.
INT32_MAX = 2**31 - 1


def check_grid_fits() -> None:
    """격자를 키웠을 때 int32 를 넘지 않는지. 넘으면 그 자리에서 멈춘다."""
    limits = {
        "격자 셀 번호 (node_cell)": GRID_W * GRID_H,
        "저장 격자 셀 번호 (node_shed)": SHED_W * SHED_H,
        "그리기 타일 번호": FINE_W * FINE_H,
    }
    over = {k: v for k, v in limits.items() if v > INT32_MAX}
    if over:
        detail = ", ".join(f"{k} {v:,} > {INT32_MAX:,}" for k, v in over.items())
        sys.exit("격자가 int32 를 넘습니다. 칸을 키우거나 권역을 나누세요: " + detail)


def cell_index(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """위경도를 전역 격자의 셀 번호로. 범위를 벗어나면 -1."""
    gx = np.floor((np.asarray(lon) - GRID_LON0) * M_PER_DEG_LON / CELL_M).astype(np.int64)
    gy = np.floor((np.asarray(lat) - GRID_LAT0) * M_PER_DEG_LAT / CELL_M).astype(np.int64)
    ok = (gx >= 0) & (gx < GRID_W) & (gy >= 0) & (gy < GRID_H)
    return np.where(ok, gy * GRID_W + gx, -1)


def shed_cell_index(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """저장용(성긴) 격자의 셀 번호."""
    gx = np.floor((np.asarray(lon) - GRID_LON0) * M_PER_DEG_LON / SHED_CELL_M).astype(np.int64)
    gy = np.floor((np.asarray(lat) - GRID_LAT0) * M_PER_DEG_LAT / SHED_CELL_M).astype(np.int64)
    ok = (gx >= 0) & (gx < SHED_W) & (gy >= 0) & (gy < SHED_H)
    return np.where(ok, gy * SHED_W + gx, -1)


# 그리기 전용 원본 기하를 담는 타일 한 변. 경로 하나가 걸치는 타일만 읽으면
# 되므로, 작으면 읽는 타일이 늘고 크면 한 번에 읽는 양이 는다. 2 km 면
# 1-2 km 짜리 도보 구간이 보통 네 타일 안에 들어온다.
FINE_TILE_M = 2000.0
FINE_W = int(np.ceil(GRID_SPAN_X / FINE_TILE_M))
FINE_H = int(np.ceil(GRID_SPAN_Y / FINE_TILE_M))


def fine_tile_index(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """그리기 전용 타일 번호. 범위를 벗어나면 -1."""
    gx = np.floor((np.asarray(lon) - GRID_LON0) * M_PER_DEG_LON / FINE_TILE_M).astype(np.int64)
    gy = np.floor((np.asarray(lat) - GRID_LAT0) * M_PER_DEG_LAT / FINE_TILE_M).astype(np.int64)
    ok = (gx >= 0) & (gx < FINE_W) & (gy >= 0) & (gy < FINE_H)
    return np.where(ok, gy * FINE_W + gx, -1)


def cell_center(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.divmod(np.asarray(cells), GRID_W)
    lon = GRID_LON0 + (gx + 0.5) * CELL_M / M_PER_DEG_LON
    lat = GRID_LAT0 + (gy + 0.5) * CELL_M / M_PER_DEG_LAT
    return lon, lat


# --------------------------------------------------------------------------
# 1단계: PBF 에서 보행로 뽑기
# --------------------------------------------------------------------------

def extract_ways() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """보행 가능한 way 들을 (노드 좌표열, 구간 경계, 계단 여부) 로 돌려준다."""
    import osmium

    class WalkHandler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.lon: list[float] = []
            self.lat: list[float] = []
            self.bounds: list[int] = [0]
            self.steps: list[bool] = []
            self.seen = 0

        def way(self, w) -> None:
            self.seen += 1
            tags = w.tags
            highway = tags.get("highway")
            if highway not in WALKABLE:
                return
            if tags.get("foot") in ("no", "private"):
                return
            if tags.get("access") in ("no", "private") and tags.get("foot") is None:
                return

            n = 0
            for node in w.nodes:
                if not node.location.valid():
                    continue
                lon, lat = node.location.lon, node.location.lat
                # 격자 밖은 어차피 버려진다. 여기서 걸러야 주부 추출본의
                # 나고야까지 메모리에 쌓지 않는다.
                if not (GRID_LON0 <= lon <= GRID_LON1 and GRID_LAT0 <= lat <= GRID_LAT1):
                    continue
                self.lon.append(lon)
                self.lat.append(lat)
                n += 1
            if n < 2:
                del self.lon[len(self.lon) - n:]
                del self.lat[len(self.lat) - n:]
                return
            self.bounds.append(self.bounds[-1] + n)
            self.steps.append(highway == "steps")

    handler = WalkHandler()
    t0 = time.time()
    for path in PBFS:
        before = len(handler.steps)
        handler.apply_file(str(path), locations=True, idx="flex_mem")
        print(f"    {path.name}: 보행로 +{len(handler.steps) - before:,}개 "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(
        f"  way {handler.seen:,}개 훑어 보행로 {len(handler.steps):,}개, "
        f"점 {len(handler.lon):,}개 ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return (
        np.array(handler.lon, dtype=np.float64),
        np.array(handler.lat, dtype=np.float64),
        np.array(handler.bounds, dtype=np.int64),
    ), np.array(handler.steps, dtype=bool)


# --------------------------------------------------------------------------
# 2단계: 격자에 스냅해 그래프 만들기
# --------------------------------------------------------------------------

def build_graph(points, steps) -> dict:
    """way 의 점들을 격자 셀로 스냅해 노드를 합치고 간선을 만든다.

    OSM 원본 노드를 그대로 쓰면 수천만 개라 감당이 안 된다. 100 m 격자로
    합치면 보행 시간 오차는 한 칸 남짓이면서 노드 수가 수십만으로 떨어진다.
    """
    lon, lat, bounds = points
    print("  격자 스냅 중...", flush=True)
    cells = cell_index(lon, lat)

    # 유효한 셀만 노드로 승격
    used = np.unique(cells[cells >= 0])
    print(f"  노드 {len(used):,}개", flush=True)

    # 노드의 실제 위치. 칸 중심을 쓰면 경로를 그릴 때 직각 계단이 된다.
    # 그 칸에 들어온 원본 점들의 평균을 대표 위치로 삼는다.
    order = np.argsort(cells, kind="stable")
    sc = cells[order]
    first = np.searchsorted(sc, used, "left")
    last = np.searchsorted(sc, used, "right")
    counts = last - first
    sums_lon = np.add.reduceat(lon[order], first)
    sums_lat = np.add.reduceat(lat[order], first)
    node_lon = (sums_lon / counts).astype(np.float32)
    node_lat = (sums_lat / counts).astype(np.float32)
    node_of = np.full(int(used.max()) + 1, -1, dtype=np.int32)
    node_of[used] = np.arange(len(used), dtype=np.int32)

    node_id = np.where(cells >= 0, node_of[np.clip(cells, 0, None)], -1)

    # way 안의 연속한 두 점을 잇는다 (way 경계를 넘지 않도록 마스크)
    a = node_id[:-1]
    b = node_id[1:]
    boundary = np.zeros(len(node_id) - 1, dtype=bool)
    boundary[bounds[1:-1] - 1] = True  # 이전 way 의 마지막 -> 다음 way 의 첫 점
    valid = (a >= 0) & (b >= 0) & (a != b) & ~boundary

    # 계단 가중치: 각 점이 어느 way 에 속하는지 되짚는다
    way_of_point = np.repeat(np.arange(len(steps)), np.diff(bounds))
    is_step = steps[way_of_point[:-1]]

    a, b = a[valid], b[valid]

    # 거리는 원본 점이 아니라 셀 중심 사이로 잰다. 점 좌표를 그대로 쓰면
    # 셀 경계를 사이에 두고 몇 m 떨어진 두 점이 실제로는 100 m 떨어진 두 셀을
    # 잇는 간선을 헐값에 깔아버려서, 그런 간선이 연쇄되면 격자를 거의 공짜로
    # 가로지르게 된다.
    ay, ax = np.divmod(used[a], GRID_W)
    by, bx = np.divmod(used[b], GRID_W)
    dist = np.hypot((bx - ax) * CELL_M, (by - ay) * CELL_M)

    cost = dist / WALK_SPEED
    cost = np.where(is_step[valid], cost * STEP_PENALTY, cost)

    # 양방향
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    w = np.concatenate([cost, cost]).astype(np.float32)
    print(f"  간선 {len(src):,}개 (양방향)", flush=True)

    return {
        "n_nodes": len(used),
        "node_cell": used.astype(np.int32),
        "node_lon": node_lon,
        "node_lat": node_lat,
        "src": src.astype(np.int32),
        "dst": dst.astype(np.int32),
        "cost": w,
    }


def prune_small_components(graph: dict) -> dict:
    """어디로도 이어지지 않는 파편을 걷어낸다.

    OSM 에는 주변 도로와 끊긴 짧은 통로가 곳곳에 있다. 역을 가장 가까운
    노드에 그냥 붙이면 이런 파편에 걸리는 경우가 생기고, 그 역은 도보권이
    한 칸뿐이라 도달 가능한데도 지도에 아무것도 그려지지 않는다. 오후나나
    공항제2빌딩처럼 큰 역도 걸린다.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = graph["n_nodes"]
    A = csr_matrix(
        (graph["cost"], (graph["src"], graph["dst"])), shape=(n, n)
    )
    _, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    keep = sizes[labels] >= MIN_COMPONENT_NODES

    remap = np.full(n, -1, dtype=np.int32)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int32)
    alive = keep[graph["src"]] & keep[graph["dst"]]

    print(
        f"  파편 제거: 노드 {n:,} -> {int(keep.sum()):,} "
        f"({n - int(keep.sum()):,}개 걷어냄)",
        flush=True,
    )
    return {
        "n_nodes": int(keep.sum()),
        "node_cell": graph["node_cell"][keep],
        "node_lon": graph["node_lon"][keep],
        "node_lat": graph["node_lat"][keep],
        "src": remap[graph["src"][alive]],
        "dst": remap[graph["dst"][alive]],
        "cost": graph["cost"][alive],
    }


def to_csr(graph: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """간선 목록을 CSR 로 정렬한다. 이웃 조회가 슬라이스 한 번으로 끝난다."""
    n = graph["n_nodes"]
    src, dst, cost = graph["src"], graph["dst"], graph["cost"]
    order = np.argsort(src, kind="stable")
    src, dst, cost = src[order], dst[order], cost[order]
    indptr = np.searchsorted(src, np.arange(n + 1)).astype(np.int32)
    return indptr, dst, cost


# --------------------------------------------------------------------------
# 3단계: 역별 도보권
# --------------------------------------------------------------------------

def bounded_dijkstra(indptr, indices, data, source: int, limit: float,
                     dist: np.ndarray, touched: list[int]) -> None:
    """source 에서 limit 초 안에 닿는 노드의 도보 시간을 dist 에 채운다.

    dist 는 호출 간에 재사용한다. 방문한 자리만 touched 에 적어두고 끝나면
    그 자리만 되돌리므로, 노드가 수백만 개여도 역 하나당 비용이 실제 도보권
    크기에만 비례한다.
    """
    import heapq

    dist[source] = 0.0
    touched.append(source)
    heap = [(0.0, source)]
    push, pop = heapq.heappush, heapq.heappop

    while heap:
        d, u = pop(heap)
        if d > dist[u]:
            continue
        lo, hi = indptr[u], indptr[u + 1]
        for k in range(lo, hi):
            v = indices[k]
            nd = d + data[k]
            if nd < dist[v] and nd <= limit:
                dist[v] = nd
                touched.append(v)
                push(heap, (nd, v))


def snap_stations(node_cell: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """각 역을 가장 가까운 보행망 노드에 붙인다. 못 붙이면 -1."""
    from scipy.spatial import cKDTree

    nlon, nlat = cell_center(node_cell)
    # 위경도를 미터로 펴서 거리 비교를 맞춘다
    tree = cKDTree(np.stack([nlon * M_PER_DEG_LON, nlat * M_PER_DEG_LAT], axis=1))
    ok = np.isfinite(coords[:, 0])
    query = np.stack([coords[:, 0] * M_PER_DEG_LON, coords[:, 1] * M_PER_DEG_LAT], axis=1)
    query[~ok] = 0.0

    # 역에서 300 m 안에 보행로가 없으면 붙이지 않는다
    d, idx = tree.query(query, distance_upper_bound=300.0)
    snapped = np.where(np.isfinite(d) & ok, idx, -1).astype(np.int32)
    print(f"  역 {int((snapped >= 0).sum()):,}/{len(snapped):,}개를 보행망에 연결", flush=True)
    return snapped


def build_sheds(indptr, indices, data, n_nodes: int, node_shed: np.ndarray,
                station_nodes: np.ndarray) -> dict:
    """역마다 MAX_SHED_SEC 안의 도보권을 계산해 이어 붙인다.

    계산은 촘촘한 그래프 위에서 하고, 결과는 성긴 저장용 격자로 내려 담는다.
    한 저장 칸에 여러 노드가 들어가면 그중 가장 빠른 시간만 남긴다.
    """
    dist = np.full(n_nodes, np.inf, dtype=np.float64)
    shed_cell: list[np.ndarray] = []
    shed_sec: list[np.ndarray] = []
    ptr = [0]

    t0 = time.time()
    for i, source in enumerate(station_nodes):
        if source < 0:
            ptr.append(ptr[-1])
            continue
        touched: list[int] = []
        bounded_dijkstra(indptr, indices, data, int(source), MAX_SHED_SEC, dist, touched)

        nodes = np.fromiter(set(touched), dtype=np.int32, count=-1)
        secs = dist[nodes]
        dist[nodes] = np.inf  # 다음 역을 위해 되돌린다

        cells = node_shed[nodes]
        ok = cells >= 0
        cells, secs = cells[ok], secs[ok]
        if len(cells) == 0:
            ptr.append(ptr[-1])
            continue

        # 저장 칸별 최솟값
        order = np.argsort(cells, kind="stable")
        cells, secs = cells[order], secs[order]
        starts = np.flatnonzero(np.concatenate(([True], cells[1:] != cells[:-1])))
        cells = cells[starts]
        secs = np.minimum.reduceat(secs, starts)

        shed_cell.append(cells.astype(np.int32))
        shed_sec.append(secs.astype(np.float32))
        ptr.append(ptr[-1] + len(cells))

        if (i + 1) % 500 == 0:
            print(
                f"    {i + 1:>4}/{len(station_nodes)}역, 누적 칸 {ptr[-1]:,}개, "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )

    return {
        "shed_cell": np.concatenate(shed_cell) if shed_cell else np.zeros(0, np.int32),
        "shed_sec": np.concatenate(shed_sec) if shed_sec else np.zeros(0, np.float32),
        "shed_ptr": np.array(ptr, dtype=np.int32),
    }


def save_fine_geometry(points) -> None:
    """원본 좌표를 그대로 저장한다. 경로를 그릴 때만 쓴다.

    계산은 40 m 격자 그래프가 그대로 한다. 거기서 나온 선은 칸마다 점이
    하나뿐이라 도로에서 최대 40 m 벗어나는데, 지도에 겹쳐 놓으면 그게
    그대로 보인다. 그릴 때만 원본 선형 위로 다시 얹으려고 따로 둔다.

    압축하지 않고 .npy 로 쓴다. npz 는 통째로 풀어야 해서 구간 하나를 그리려
    190 MB 를 다 올리게 된다. 압축을 안 하면 mmap 으로 걸쳐서 필요한 조각만
    읽을 수 있다.
    """
    lon, lat, bounds = points

    # 1e-7 도 = 약 1 cm. float32 는 이 위도에서 이미 1.5 m 라 도로 폭보다 굵다.
    pts = np.stack([np.rint(lon * 1e7), np.rint(lat * 1e7)], axis=1).astype(np.int32)

    # 선분은 way 안에서 이웃한 두 점. way 와 way 사이는 잇지 않는다.
    seg = np.arange(len(lon) - 1, dtype=np.int64)
    boundary = np.zeros(len(lon) - 1, dtype=bool)
    boundary[bounds[1:-1] - 1] = True
    seg = seg[~boundary]

    # 선분을 첫 점이 놓인 타일에 넣고, 타일 순으로 모아 CSR 로 만든다
    tile = fine_tile_index(lon[seg], lat[seg])
    keep = tile >= 0
    seg, tile = seg[keep], tile[keep]
    order = np.argsort(tile, kind="stable")
    seg, tile = seg[order].astype(np.int32), tile[order]
    ptr = np.searchsorted(tile, np.arange(FINE_W * FINE_H + 1, dtype=np.int64)).astype(np.int32)

    np.save(OUT / "fine_pt.npy", pts)
    np.save(OUT / "fine_seg.npy", seg)
    np.save(OUT / "fine_ptr.npy", ptr)
    np.save(OUT / "fine_grid.npy",
            np.array([GRID_LON0, GRID_LAT0, FINE_TILE_M, FINE_W, FINE_H,
                      M_PER_DEG_LON, M_PER_DEG_LAT]))
    total = sum((OUT / f).stat().st_size for f in
                ("fine_pt.npy", "fine_seg.npy", "fine_ptr.npy", "fine_grid.npy"))
    print(f"  점 {len(pts):,}개, 선분 {len(seg):,}개, {total / 1e6:.0f} MB", flush=True)


def main() -> None:
    check_grid_fits()

    missing = [p for p in PBFS if not p.exists()]
    if missing:
        sys.exit("OSM 추출본이 없습니다:\n  " + "\n  ".join(str(p) for p in missing))

    # 서버가 돌고 있으면 원본 기하 파일을 mmap 으로 물고 있어서 윈도우가
    # 덮어쓰기를 막는다. 마지막 단계에서 5분 쓰고 실패하는 것보다 지금
    # 걸리는 편이 낫다.
    locked = []
    for name in ("fine_pt.npy", "fine_seg.npy", "fine_ptr.npy"):
        target = OUT / name
        if not target.exists():
            continue
        try:
            with open(target, "r+b"):
                pass
        except OSError:
            locked.append(name)
    if locked:
        sys.exit(
            "다른 프로세스가 이 파일들을 쓰고 있습니다: " + ", ".join(locked)
            + "\n  서버(src/server.py)를 멈춘 뒤 다시 돌려주세요."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    import json

    print("1) 보행로 추출", flush=True)
    points, steps = extract_ways()

    print("2) 그래프 구축", flush=True)
    graph = build_graph(points, steps)
    graph = prune_small_components(graph)
    indptr, indices, data = to_csr(graph)

    # 노드마다 저장용 격자의 어느 칸에 떨어지는지 미리 구해 둔다
    nlon, nlat = cell_center(graph["node_cell"])
    node_shed = shed_cell_index(nlon, nlat).astype(np.int32)

    np.savez_compressed(
        OUT / "graph.npz",
        node_cell=graph["node_cell"],
        node_lon=graph["node_lon"],
        node_lat=graph["node_lat"],
        node_shed=node_shed,
        indptr=indptr,
        indices=indices,
        data=data,
        grid=np.array([GRID_LON0, GRID_LAT0, CELL_M, GRID_W, GRID_H,
                       M_PER_DEG_LON, M_PER_DEG_LAT]),
        shed_grid=np.array([GRID_LON0, GRID_LAT0, SHED_CELL_M, SHED_W, SHED_H,
                            M_PER_DEG_LON, M_PER_DEG_LAT]),
    )
    print(f"  graph.npz {(OUT / 'graph.npz').stat().st_size / 1e6:.0f} MB", flush=True)

    print("2-1) 그리기 전용 원본 기하", flush=True)
    save_fine_geometry(points)

    print("3) 역별 도보권", flush=True)
    stops = json.loads((BASE / "stops.json").read_text(encoding="utf-8"))
    coords = np.array(stops["coords"], dtype=np.float64)
    station_nodes = snap_stations(graph["node_cell"], coords)

    sheds = build_sheds(indptr, indices, data, graph["n_nodes"], node_shed, station_nodes)
    np.savez_compressed(
        OUT / "sheds.npz",
        station_node=station_nodes,
        shed_cell=sheds["shed_cell"],
        shed_sec=sheds["shed_sec"],
        shed_ptr=sheds["shed_ptr"],
    )
    print("4) 육지 마스크", flush=True)
    land = build_land_mask()
    np.savez_compressed(OUT / "land.npz", land=land["land"], grid=land["grid"])
    print(f"  land.npz {(OUT / 'land.npz').stat().st_size / 1e6:.1f} MB", flush=True)

    size = (OUT / "sheds.npz").stat().st_size / 1e6
    total = len(sheds["shed_cell"])
    print(f"  sheds.npz {size:.0f} MB, 도보권 셀 {total:,}개 "
          f"(역당 평균 {total / max((station_nodes >= 0).sum(), 1):.0f}개)", flush=True)




# --------------------------------------------------------------------------
# 4단계: 육지 / 바다 마스크
# --------------------------------------------------------------------------
# 지점을 클릭했을 때 "바다"와 "권역 밖"을 갈라 말해주려면 육지인지 알아야
# 한다. 거리만으로는 안 된다. 앞바다 20 km 와 권역 경계 밖 20 km 가 같은
# 거리인데 하나는 바다, 하나는 다른 지방이다.
# 해안선은 지도에서 그대로 눈에 띄므로 촘촘해야 한다. 1 km 로 구우면
# 만을 가로질러 잘리고 반도 끝이 뭉개진다.
LAND_CELL_M = 250.0


def build_land_mask() -> dict:
    """OSM 해안선으로 육지 마스크를 만든다.

    해안선 way 는 육지를 왼쪽에 두고 그려진다는 규약이 있지만, 조각조각
    끊겨 있어 그대로 이어 붙이기는 까다롭다. 대신 해안선을 격자에 굽고
    바깥에서 물을 채워 들어가, 닿지 않은 곳을 육지로 본다.
    """
    import osmium
    from scipy import ndimage

    class Coast(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.lon: list[float] = []
            self.lat: list[float] = []
            self.bounds: list[int] = [0]

        def way(self, w) -> None:
            if w.tags.get("natural") != "coastline":
                return
            n = 0
            for nd in w.nodes:
                if nd.location.valid():
                    self.lon.append(nd.location.lon)
                    self.lat.append(nd.location.lat)
                    n += 1
            if n < 2:
                del self.lon[len(self.lon) - n:]
                del self.lat[len(self.lat) - n:]
                return
            self.bounds.append(self.bounds[-1] + n)

    print("  해안선 추출 중...", flush=True)
    h = Coast()
    for path in PBFS:
        h.apply_file(str(path), locations=True, idx="flex_mem")
    lon = np.array(h.lon)
    lat = np.array(h.lat)
    print(f"  해안선 점 {len(lon):,}개", flush=True)

    m_lon = 111_320.0 * np.cos(np.radians(GRID_LAT_REF))
    lon0, lat0 = float(lon.min()) - 0.3, float(lat.min()) - 0.3
    w = int(np.ceil((float(lon.max()) + 0.3 - lon0) * m_lon / LAND_CELL_M)) + 1
    hgt = int(np.ceil((float(lat.max()) + 0.3 - lat0) * M_PER_DEG_LAT / LAND_CELL_M)) + 1

    def to_cell(x, y):
        return (
            np.floor((x - lon0) * m_lon / LAND_CELL_M).astype(np.int64),
            np.floor((y - lat0) * M_PER_DEG_LAT / LAND_CELL_M).astype(np.int64),
        )

    coast = np.zeros((hgt, w), dtype=bool)

    def burn(ax, ay, bx, by) -> None:
        """격자에 선분을 굽는다. 칸 사이가 벌어지지 않도록 촘촘히 찍는다."""
        if len(ax) == 0:
            return
        steps = np.maximum(np.abs(bx - ax), np.abs(by - ay)) + 1
        for t in range(int(steps.max())):
            f = np.minimum(t / np.maximum(steps - 1, 1), 1.0)
            cx = np.rint(ax + (bx - ax) * f).astype(np.int64)
            cy = np.rint(ay + (by - ay) * f).astype(np.int64)
            ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < hgt) & (t < steps)
            coast[cy[ok], cx[ok]] = True

    # 해안선 본체
    a, b = np.arange(len(lon) - 1), np.arange(1, len(lon))
    seam = np.zeros(len(lon) - 1, dtype=bool)
    seam[np.array(h.bounds[1:-1]) - 1] = True
    a, b = a[~seam], b[~seam]
    ax, ay = to_cell(lon[a], lat[a])
    bx, by = to_cell(lon[b], lat[b])
    burn(ax, ay, bx, by)

    # 추출본 경계를 넘어가며 끊긴 자리를 격자 가장자리까지 이어 막는다.
    # 본토 해안선은 열린 호라서, 그대로 두면 양 끝을 돌아 물이 육지로 샌다.
    ends = []
    for k in range(len(h.bounds) - 1):
        i0, i1 = h.bounds[k], h.bounds[k + 1] - 1
        ends.append((lon[i0], lat[i0]))
        ends.append((lon[i1], lat[i1]))
    key = np.round(np.array(ends) * 1e6).astype(np.int64)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    lone = uniq[counts == 1] / 1e6
    print(f"  해안선이 끊긴 자리 {len(lone)}곳을 격자 끝까지 막는다", flush=True)
    for x, y in lone:
        cx, cy = to_cell(np.array([x]), np.array([y]))
        cx, cy = int(cx[0]), int(cy[0])
        # 가장 가까운 가장자리로 곧장 뺀다
        to_left, to_right, to_bot, to_top = cx, w - 1 - cx, cy, hgt - 1 - cy
        nearest = min(to_left, to_right, to_bot, to_top)
        if nearest == to_left:
            tx, ty = 0, cy
        elif nearest == to_right:
            tx, ty = w - 1, cy
        elif nearest == to_bot:
            tx, ty = cx, 0
        else:
            tx, ty = cx, hgt - 1
        burn(np.array([cx]), np.array([cy]), np.array([tx]), np.array([ty]))

    # 확실한 바다 지점에서 물을 흘려 넣는다. 해안선에 막혀 못 들어간 곳이
    # 육지다. 가장자리 전체에서 채우면 안 된다. 추출본의 북쪽·서쪽 경계는
    # 육지를 가로지르므로 거기서 채우면 육지까지 잠겨버린다.
    free = ~coast
    labels, _ = ndimage.label(free)

    OCEAN_SEEDS = [
        (142.0, 35.5),   # 지바 동쪽 먼바다
        (140.0, 33.0),   # 남쪽 먼바다
        (139.0, 34.3),   # 사가미나다
        (141.5, 30.0),   # 이즈 제도 앞바다
    ]
    sea_labels = set()
    for slon, slat in OCEAN_SEEDS:
        cx, cy = to_cell(np.array([slon]), np.array([slat]))
        if 0 <= cx[0] < w and 0 <= cy[0] < hgt:
            lab = int(labels[cy[0], cx[0]])
            if lab > 0:
                sea_labels.add(lab)

    sea = np.isin(labels, sorted(sea_labels)) if sea_labels else np.zeros_like(free)
    land = ~sea          # 해안선 자체도 육지로 친다

    print(f"  육지 마스크 {land.shape}, 육지 칸 {int(land.sum()):,} "
          f"({land.mean() * 100:.0f}%)", flush=True)
    return {
        "land": land,
        "grid": np.array([lon0, lat0, LAND_CELL_M, w, hgt, m_lon, M_PER_DEG_LAT]),
    }


if __name__ == "__main__":
    main()
