"""실행 중에 쓰는 보행 네트워크.

build_walk.py 가 만들어 둔 그래프와 역별 도보권을 올려두고,
  - 역에서 퍼지는 도보: 미리 계산된 도보권을 그대로 읽는다
  - 임의 지점에서 퍼지는 도보: 출발지·도착지는 미리 알 수 없으므로
    그때그때 국소 다익스트라를 돌린다 (반경이 작아 수십 ms 로 끝난다)
"""
from __future__ import annotations

import heapq
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from router import WALK_SPEED

# 보행망에서 이 거리 안에 도로가 없으면 그 지점은 걸어 나갈 수 없다고 본다
SNAP_RADIUS_M = 400.0


@dataclass
class WalkNet:
    node_cell: np.ndarray     # 노드 -> 그래프 격자(촘촘) 셀 번호
    node_lon: np.ndarray      # 노드의 실제 위치 (칸 중심이 아니다)
    node_lat: np.ndarray
    node_shed: np.ndarray     # 노드 -> 저장 격자(성김) 셀 번호
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    station_node: np.ndarray  # 역 -> 노드 (-1 이면 연결 실패)
    shed_cell: np.ndarray     # 저장 격자 셀 번호
    shed_sec: np.ndarray
    shed_ptr: np.ndarray
    lon0: float
    lat0: float
    cell_m: float             # 그래프 격자
    grid_w: int
    grid_h: int
    shed_cell_m: float        # 저장 격자
    shed_w: int
    shed_h: int
    m_per_deg_lon: float
    m_per_deg_lat: float
    _dist: np.ndarray | None = None
    _prev: np.ndarray | None = None
    # 거리 배열은 호출마다 새로 잡기엔 너무 커서(노드 476만 개) 재사용한다.
    # 대신 여러 요청이 동시에 들어오면 서로의 계산을 덮어쓰므로 잠가야 한다.
    # Flask 는 기본이 멀티스레드라 이게 없으면 도보권이 통째로 망가진다.
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ---------- 좌표 변환 ----------

    def node_points(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """노드의 실제 위치. 경로를 그릴 때는 칸 중심이 아니라 이것을 쓴다.

        칸 중심을 이으면 40 m 격자를 따라 직각 계단이 된다.
        """
        return self.node_lon[nodes], self.node_lat[nodes]

    def cell_center(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gy, gx = np.divmod(np.asarray(cells), self.grid_w)
        lon = self.lon0 + (gx + 0.5) * self.cell_m / self.m_per_deg_lon
        lat = self.lat0 + (gy + 0.5) * self.cell_m / self.m_per_deg_lat
        return lon, lat

    def nearest_node(self, lon: float, lat: float) -> int:
        """이 지점에 가장 가까운 노드. 너무 멀면 -1.

        KD 트리를 쓰면 노드 611만 개에 대해 트리가 250 MB 넘게 나온다.
        노드는 이미 격자 셀 번호 순으로 정렬돼 있으므로, 질의 지점 둘레
        몇 칸만 이분 탐색으로 꺼내 보면 된다. 반경 400 m 면 40 m 칸으로
        10칸, 21x21 = 441칸이다.
        """
        gx = int(np.floor((lon - self.lon0) * self.m_per_deg_lon / self.cell_m))
        gy = int(np.floor((lat - self.lat0) * self.m_per_deg_lat / self.cell_m))
        reach = int(np.ceil(SNAP_RADIUS_M / self.cell_m))

        picked = []
        for row in range(max(gy - reach, 0), min(gy + reach, self.grid_h - 1) + 1):
            lo_col = max(gx - reach, 0)
            hi_col = min(gx + reach, self.grid_w - 1)
            if lo_col > hi_col:
                continue
            # 한 줄은 셀 번호가 연속이라 한 번의 이분 탐색으로 잘린다.
            #
            # 찾는 값을 배열과 같은 자료형으로 맞춰야 한다. int32 배열에
            # 파이썬 정수를 넘기면 numpy 가 배열 쪽을 int64 로 올려 통째로
            # 복사한다. 611만 개 배열이라 건당 10 ms 가 든다 (맞추면 0.008 ms).
            key = self.node_cell.dtype.type
            lo = np.searchsorted(self.node_cell, key(row * self.grid_w + lo_col), "left")
            hi = np.searchsorted(self.node_cell, key(row * self.grid_w + hi_col), "right")
            if hi > lo:
                picked.append(np.arange(lo, hi))
        if not picked:
            return -1

        cand = np.concatenate(picked)
        # 칸을 고를 때는 격자의 경도 폭을 쓰지만 거리는 그 위도의 실제 폭으로 잰다.
        # 전국 격자(기준 36도)에서는 둘이 홋카이도에서 14% 어긋난다.
        dx = (self.node_lon[cand] - lon) * 111_320.0 * np.cos(np.radians(lat))
        dy = (self.node_lat[cand] - lat) * self.m_per_deg_lat
        dist = np.hypot(dx, dy)
        k = int(np.argmin(dist))
        return int(cand[k]) if dist[k] <= SNAP_RADIUS_M else -1

    # ---------- 역 도보권 ----------

    def shed(self, station: int) -> tuple[np.ndarray, np.ndarray]:
        """역 하나의 (저장 격자 셀, 도보 시간). 연결 실패한 역은 빈 배열."""
        lo, hi = self.shed_ptr[station], self.shed_ptr[station + 1]
        return self.shed_cell[lo:hi], self.shed_sec[lo:hi]

    # ---------- 임의 지점에서의 도보 ----------

    def _dijkstra(self, lon: float, lat: float, limit_sec: float, target: int,
                  track_prev: bool = False):
        """공유 거리 배열 위에서 한 번 퍼뜨린다. 잠금은 호출하는 쪽이 쥔다."""
        source = self.nearest_node(lon, lat)
        if source < 0:
            return None, [], None

        if self._dist is None:
            self._dist = np.full(len(self.node_cell), np.inf, dtype=np.float64)
        dist = self._dist
        indptr, indices, data = self.indptr, self.indices, self.data

        # 지점과 스냅된 노드 사이의 오차는 첫 걸음에 얹는다
        offset = float(
            np.hypot(
                (float(self.node_lon[source]) - lon) * self.m_per_deg_lon,
                (float(self.node_lat[source]) - lat) * self.m_per_deg_lat,
            )
            / WALK_SPEED
        )

        prev = self._prev if track_prev else None
        if track_prev and prev is None:
            prev = self._prev = np.full(len(self.node_cell), -1, dtype=np.int32)

        dist[source] = offset
        touched = [source]
        heap = [(offset, source)]
        push, pop = heapq.heappush, heapq.heappop
        settled = None

        while heap:
            d, u = pop(heap)
            if d > dist[u]:
                continue
            if u == target:
                settled = d      # 꺼낸 시점에 확정된 값이다
                break
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                nd = d + data[k]
                if nd < dist[v] and nd <= limit_sec:
                    dist[v] = nd
                    if prev is not None:
                        prev[v] = u
                    touched.append(v)
                    push(heap, (nd, v))

        return dist, touched, settled

    def from_point(self, lon: float, lat: float, limit_sec: float
                   ) -> tuple[np.ndarray, np.ndarray]:
        """(lon, lat) 에서 limit_sec 안에 닿는 노드와 도보 시간."""
        with self._lock:
            dist, touched, _ = self._dijkstra(lon, lat, limit_sec, target=-1)
            if dist is None:
                return np.zeros(0, np.int32), np.zeros(0, np.float32)
            nodes = np.fromiter(set(touched), dtype=np.int32, count=-1)
            secs = dist[nodes].astype(np.float32)
            self._reset(dist, touched)
            return nodes, secs

    def time_to_node(self, lon: float, lat: float, target: int,
                     limit_sec: float) -> float:
        """(lon, lat) 에서 특정 노드까지의 도보 시간. 못 닿으면 inf.

        목표가 확정되는 순간 멈춘다. 이때 다른 노드의 값은 아직 확정 전이라
        쓸 수 없으므로 여기서는 목표값만 돌려준다.
        """
        if target < 0:
            return float("inf")
        with self._lock:
            dist, touched, settled = self._dijkstra(lon, lat, limit_sec, target)
            if dist is None:
                return float("inf")
            self._reset(dist, touched)
            return float(settled) if settled is not None else float("inf")

    def path_to_node(self, lon: float, lat: float, target: int,
                     limit_sec: float) -> list[list[float]]:
        """(lon, lat) 에서 특정 노드까지 실제로 걷는 길. 못 닿으면 빈 목록."""
        if target < 0:
            return []
        with self._lock:
            dist, touched, settled = self._dijkstra(
                lon, lat, limit_sec, target, track_prev=True
            )
            if dist is None or settled is None:
                if dist is not None:
                    self._reset(dist, touched)
                return []

            chain: list[int] = []
            cur = target
            for _ in range(200_000):
                chain.append(cur)
                cur = int(self._prev[cur])
                if cur < 0:
                    break
            self._reset(dist, touched)

        chain.reverse()
        lons, lats = self.node_points(np.array(chain, dtype=np.int64))
        return [[round(float(x), 6), round(float(y), 6)] for x, y in zip(lons, lats)]

    def _reset(self, dist: np.ndarray, touched: list[int]) -> None:
        """다음 호출을 위해 건드린 자리만 되돌린다."""
        idx = np.fromiter(set(touched), dtype=np.int32, count=-1)
        dist[idx] = np.inf
        if self._prev is not None:
            self._prev[idx] = -1

    def station_times_from_point(self, lon: float, lat: float, limit_sec: float,
                                 n_stations: int) -> np.ndarray:
        """임의 지점에서 각 역까지 걸어가는 시간. 못 닿으면 inf."""
        nodes, secs = self.from_point(lon, lat, limit_sec)
        out = np.full(n_stations, np.inf, dtype=np.float64)
        if len(nodes) == 0:
            return out
        # 보행망 크기 배열을 만들지 않고 닿은 노드만 정렬해 역 노드를 찾는다. 전국 보행망은
        # 노드가 3,100만 개라 요청마다 249 MB 를 채웠다.
        order = np.argsort(nodes, kind="stable")
        got = np.asarray(nodes)[order]
        linked = np.flatnonzero(self.station_node >= 0)
        want = self.station_node[linked]
        at = np.minimum(np.searchsorted(got, want), len(got) - 1)
        hit = got[at] == want
        out[linked[hit]] = np.asarray(secs, dtype=np.float64)[order][at[hit]]
        return out


GRAPH_KEYS = ("node_cell", "node_lon", "node_lat", "node_shed", "indptr", "indices", "data",
              "grid", "shed_grid")
SHED_KEYS = ("station_node", "shed_cell", "shed_sec", "shed_ptr")


def _mapped(walk_dir: Path, graph_path: Path, sheds_path: Path):
    """graph.npz·sheds.npz 를 압축 없는 .npy 로 풀어 둔 것을 메모리 매핑으로 연다.

    npz 는 통째로 풀어 올려야 한다. 전국 보행망이면 프로세스마다 1.2 GB 다.
    풀어 둔 .npy 를 매핑하면 쓰는 쪽만 디스크에서 읽고, 서버 프로세스가 여럿이어도
    한 벌을 함께 쓴다. 풀어 둔 것이 없거나 원본(크기·수정 시각)이 바뀌었으면 새로
    푼다. 다른 프로세스가 매핑해 둔 파일은 윈도우에서 덮어쓰지 못하므로, 그때는
    None 을 돌려 npz 를 그대로 올리게 한다.
    """
    import json

    mm = walk_dir / "mm"
    stamp = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in (graph_path, sheds_path)}
    stamp_path = mm / "stamp.json"
    try:
        fresh = json.loads(stamp_path.read_text(encoding="utf-8")) == stamp
    except (OSError, ValueError):
        fresh = False
    if not fresh:
        try:
            mm.mkdir(exist_ok=True)
            stamp_path.unlink(missing_ok=True)
            for src, keys in ((graph_path, GRAPH_KEYS), (sheds_path, SHED_KEYS)):
                z = np.load(src)
                for k in keys:
                    tmp = mm / f"{k}.part.npy"      # np.save 는 .npy 로 안 끝나면 덧붙인다
                    np.save(tmp, z[k])
                    os.replace(tmp, mm / f"{k}.npy")
            stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
        except OSError:
            return None
    # np.asarray 로 memmap 을 보통 배열 뷰로 바꾼다(복사하지 않는다). memmap 을
    # 그대로 두면 다익스트라의 원소 하나씩 읽기가 서브클래스를 거쳐 느려진다.
    return {k: np.asarray(np.load(mm / f"{k}.npy", mmap_mode="r"))
            for k in GRAPH_KEYS + SHED_KEYS}


def load(walk_dir) -> WalkNet | None:
    """보행망이 준비돼 있으면 올리고, 없으면 None (직선거리 근사로 돌아간다)."""
    walk_dir = Path(walk_dir)
    graph_path, sheds_path = walk_dir / "graph.npz", walk_dir / "sheds.npz"
    if not (graph_path.exists() and sheds_path.exists()):
        return None

    # 풀어 둔 파일은 권역마다 수백 MB 라, 서버를 띄울 때 켤 때만 만든다
    got = (_mapped(walk_dir, graph_path, sheds_path)
           if os.environ.get("WALK_MMAP") == "1" else None)
    if got is None:
        g, s = np.load(graph_path), np.load(sheds_path)
    else:
        g = s = got
    lon0, lat0, cell_m, grid_w, grid_h, mlon, mlat = g["grid"]
    _, _, shed_m, shed_w, shed_h, _, _ = g["shed_grid"]
    return WalkNet(
        node_cell=g["node_cell"],
        node_lon=g["node_lon"],
        node_lat=g["node_lat"],
        node_shed=g["node_shed"],
        indptr=g["indptr"],
        indices=g["indices"],
        data=g["data"],
        station_node=s["station_node"],
        shed_cell=s["shed_cell"],
        shed_sec=s["shed_sec"],
        shed_ptr=s["shed_ptr"],
        lon0=float(lon0),
        lat0=float(lat0),
        cell_m=float(cell_m),
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        shed_cell_m=float(shed_m),
        shed_w=int(shed_w),
        shed_h=int(shed_h),
        m_per_deg_lon=float(mlon),
        m_per_deg_lat=float(mlat),
    )
