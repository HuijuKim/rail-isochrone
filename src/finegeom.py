"""그리는 선을 원본 도로 위로 다시 얹는다.

경로 계산은 40 m 격자 그래프가 한다. 거기서 나온 선은 칸마다 점이 하나뿐이고
그 점도 칸 안 원본 점들의 평균이라, 도로에서 최대 40 m 벗어난다. 한 칸에 서로
다른 두 길이 걸리면 평균점이 두 길 사이 허공에 놓인다. 지도에 겹쳐 놓으면
그게 그대로 보인다.

여기서 하는 일은 그리기뿐이다. 소요 시간과 경로 선택은 건드리지 않는다.
성긴 경로가 지나간 회랑 안에서만 원본 선형을 꺼내 다시 이으므로, 엉뚱한
길로 새지 않으면서 선만 도로에 붙는다.

원본 기하는 압축하지 않은 .npy 라 mmap 으로 걸쳐 둔다. 구간 하나가 걸치는
타일만 읽으므로 190 MB 를 통째로 올리지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 성긴 경로에서 이만큼 안쪽에 있는 도로만 후보로 둔다. 좁히면 원본 선형이
# 끊겨 길을 못 찾고 성긴 선으로 되돌아간다. 수도권 시가지 표본 120개로
# 재보면 70 m 는 12%, 150 m 는 4%, 220 m 는 2% 가 되돌아간다. 넓혀도 뽑힌
# 길이는 오히려 조금 짧아져(1.028배 -> 1.024배) 옆길로 새지는 않는다.
# 비용은 구간당 4 ms 에서 5 ms 로 는다.
CORRIDOR_M = 150.0
# 구간 하나가 이보다 많은 타일에 걸치면 그냥 성긴 선을 쓴다. 도보 구간은
# 길어야 몇 km 라 여기 걸릴 일이 없고, 걸린다면 뭔가 잘못된 것이다.
MAX_TILES = 100
# 경로 양 끝을 원본 점에 붙일 때 이보다 멀면 포기한다
SNAP_M = 80.0


@dataclass
class FineGeometry:
    pt: np.ndarray        # (N, 2) int32, 1e-7 도 단위
    seg: np.ndarray       # int32. 선분은 pt[i] -> pt[i+1]
    ptr: np.ndarray       # 타일별 seg 범위 (CSR)
    lon0: float
    lat0: float
    tile_m: float
    grid_w: int
    grid_h: int
    m_lon: float
    m_lat: float

    def _tiles(self, west, south, east, north) -> list[int] | None:
        gx0 = int(np.floor((west - self.lon0) * self.m_lon / self.tile_m))
        gx1 = int(np.floor((east - self.lon0) * self.m_lon / self.tile_m))
        gy0 = int(np.floor((south - self.lat0) * self.m_lat / self.tile_m))
        gy1 = int(np.floor((north - self.lat0) * self.m_lat / self.tile_m))
        gx0, gx1 = max(gx0, 0), min(gx1, self.grid_w - 1)
        gy0, gy1 = max(gy0, 0), min(gy1, self.grid_h - 1)
        if gx0 > gx1 or gy0 > gy1:
            return None
        if (gx1 - gx0 + 1) * (gy1 - gy0 + 1) > MAX_TILES:
            return None
        return [y * self.grid_w + x
                for y in range(gy0, gy1 + 1) for x in range(gx0, gx1 + 1)]

    def _segments(self, tiles: list[int]) -> np.ndarray:
        parts = [self.seg[self.ptr[t]:self.ptr[t + 1]] for t in tiles]
        parts = [p for p in parts if len(p)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)

    def trace(self, path: list[list[float]]) -> list[list[float]]:
        """성긴 경로를 원본 도로 위로 다시 그린다. 실패하면 받은 것을 그대로."""
        if len(path) < 2:
            return path
        try:
            out = self._trace(np.asarray(path, dtype=np.float64))
        except Exception:
            return path
        return path if out is None else out

    def _trace(self, line: np.ndarray) -> list[list[float]] | None:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        margin_lon = CORRIDOR_M / self.m_lon
        margin_lat = CORRIDOR_M / self.m_lat
        tiles = self._tiles(
            line[:, 0].min() - margin_lon, line[:, 1].min() - margin_lat,
            line[:, 0].max() + margin_lon, line[:, 1].max() + margin_lat,
        )
        if not tiles:
            return None

        seg = self._segments(tiles)
        if len(seg) == 0:
            return None

        a = np.asarray(self.pt[seg], dtype=np.float64) / 1e7
        b = np.asarray(self.pt[seg + 1], dtype=np.float64) / 1e7

        # 회랑 밖 도로는 버린다. 성긴 경로를 촘촘히 훑어 지나간 칸을 표시하고
        # 그 칸(과 이웃)에 걸치는 선분만 남긴다. 거리 계산보다 훨씬 싸다.
        marked = self._corridor_cells(line)
        keep = self._in_cells(a, marked) | self._in_cells(b, marked)
        if not keep.any():
            return None
        a, b = a[keep], b[keep]

        # 같은 좌표는 같은 노드다. OSM 교차점은 way 마다 따로 실려 오므로
        # 좌표로 묶지 않으면 길이 서로 이어지지 않는다.
        both = np.concatenate([a, b])
        uniq, inv = np.unique(np.rint(both * 1e7).astype(np.int64), axis=0,
                              return_inverse=True)
        n = len(uniq)
        if n < 2:
            return None
        ia, ib = inv[:len(a)], inv[len(a):]

        pts = uniq.astype(np.float64) / 1e7
        px, py = pts[:, 0] * self.m_lon, pts[:, 1] * self.m_lat
        w = np.hypot(px[ib] - px[ia], py[ib] - py[ia])

        src = self._nearest(px, py, line[0])
        dst = self._nearest(px, py, line[-1])
        if src < 0 or dst < 0 or src == dst:
            return None

        graph = csr_matrix(
            (np.concatenate([w, w]),
             (np.concatenate([ia, ib]), np.concatenate([ib, ia]))),
            shape=(n, n),
        )
        dist, prev = dijkstra(graph, indices=src, return_predecessors=True)
        if not np.isfinite(dist[dst]):
            return None

        chain = [dst]
        while chain[-1] != src:
            nxt = prev[chain[-1]]
            if nxt < 0 or len(chain) > n:
                return None
            chain.append(int(nxt))
        chain.reverse()

        traced = pts[chain]
        # 양 끝의 실제 지점(집 앞, 역 출입구)은 도로 위가 아니다. 그대로 잇는다.
        out = [line[0].tolist()] + traced.tolist() + [line[-1].tolist()]
        return [[round(p[0], 6), round(p[1], 6)] for p in _dedupe(out)]

    def _corridor_cells(self, line: np.ndarray) -> set[int]:
        """성긴 경로가 지나간 칸들 (이웃 한 겹 포함)."""
        step = CORRIDOR_M / 2.0
        xs, ys = [], []
        for k in range(len(line) - 1):
            p, q = line[k], line[k + 1]
            d = np.hypot((q[0] - p[0]) * self.m_lon, (q[1] - p[1]) * self.m_lat)
            m = max(int(d / step), 1) + 1
            t = np.linspace(0.0, 1.0, m)
            xs.append(p[0] + (q[0] - p[0]) * t)
            ys.append(p[1] + (q[1] - p[1]) * t)
        gx = np.floor(np.concatenate(xs) * self.m_lon / CORRIDOR_M).astype(np.int64)
        gy = np.floor(np.concatenate(ys) * self.m_lat / CORRIDOR_M).astype(np.int64)
        cells = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cells.update(((gy + dy) * 1_000_003 + (gx + dx)).tolist())
        return cells

    def _in_cells(self, p: np.ndarray, cells: set[int]) -> np.ndarray:
        gx = np.floor(p[:, 0] * self.m_lon / CORRIDOR_M).astype(np.int64)
        gy = np.floor(p[:, 1] * self.m_lat / CORRIDOR_M).astype(np.int64)
        key = gy * 1_000_003 + gx
        return np.fromiter((k in cells for k in key.tolist()), dtype=bool, count=len(key))

    def _nearest(self, px: np.ndarray, py: np.ndarray, target: np.ndarray) -> int:
        d = np.hypot(px - target[0] * self.m_lon, py - target[1] * self.m_lat)
        i = int(np.argmin(d))
        return i if d[i] <= SNAP_M else -1


def _dedupe(points: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for p in points:
        q = [round(float(p[0]), 6), round(float(p[1]), 6)]
        if not out or out[-1] != q:
            out.append(q)
    return out


def load(walk_dir) -> FineGeometry | None:
    """원본 기하가 준비돼 있으면 올린다. 없으면 None (성긴 선을 그대로 쓴다)."""
    walk_dir = Path(walk_dir)
    names = ("fine_pt.npy", "fine_seg.npy", "fine_ptr.npy", "fine_grid.npy")
    if not all((walk_dir / n).exists() for n in names):
        return None

    grid = np.load(walk_dir / "fine_grid.npy")
    lon0, lat0, tile_m, w, h, m_lon, m_lat = grid
    return FineGeometry(
        pt=np.load(walk_dir / "fine_pt.npy", mmap_mode="r"),
        seg=np.load(walk_dir / "fine_seg.npy", mmap_mode="r"),
        ptr=np.load(walk_dir / "fine_ptr.npy"),
        lon0=float(lon0),
        lat0=float(lat0),
        tile_m=float(tile_m),
        grid_w=int(w),
        grid_h=int(h),
        m_lon=float(m_lon),
        m_lat=float(m_lat),
    )
