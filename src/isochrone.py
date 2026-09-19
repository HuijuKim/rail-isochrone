"""역별 도착 시각을 지도 위의 도달권역 폴리곤으로 바꾼다.

각 역에 남은 시간만큼 걸어서 퍼질 수 있으므로, 격자 위에 "출발지에서 이 지점
까지 걸리는 총 시간" 을 그린 뒤 등고선을 따면 그것이 곧 등시선이다.
"""
from __future__ import annotations

import numpy as np

from router import (
    DETOUR_FACTOR,
    INF,
    WALK_SPEED,
    Graph,
    haversine_m,
    walk_seconds,
)

# 하차 후 걸어서 퍼지는 시간의 상한. 제한이 없으면 외곽 역 주변으로 권역이
# 과도하게 부풀어 실제 생활권과 어긋난다.
EGRESS_WALK_MAX_SEC = 20 * 60
# 격자 한 칸의 크기 (m)
CELL_SIZE_M = 150.0
# 격자가 지나치게 커지지 않도록 하는 한 변의 최대 칸 수
MAX_GRID_CELLS = 1400


class Field:
    """등거리 근사 좌표계 위의 소요시간 격자."""

    def __init__(self, lon0: float, lat0: float, radius_m: float | None = None,
                 x: np.ndarray | None = None, y: np.ndarray | None = None,
                 cell: float | None = None):
        self.lon0, self.lat0 = lon0, lat0
        self.m_per_deg_lat = 111_132.0
        self.m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))

        if x is not None and y is not None:
            # 보행망에서 실제로 값이 있는 범위에 맞춰 잘라 쓰는 경우
            self.x = np.asarray(x, dtype=np.float32)
            self.y = np.asarray(y, dtype=np.float32)
            self.cell = float(cell if cell is not None else (self.x[1] - self.x[0]))
        else:
            step = max(CELL_SIZE_M, 2 * radius_m / MAX_GRID_CELLS)
            self.cell = step
            n = int(np.ceil(radius_m / step))
            self.x = (np.arange(-n, n + 1) * step).astype(np.float32)  # 동쪽(+) m
            self.y = (np.arange(-n, n + 1) * step).astype(np.float32)  # 북쪽(+) m
        self.grid = np.full((len(self.y), len(self.x)), np.float32(np.inf), dtype=np.float32)

    def to_xy(self, lon, lat):
        return (
            (np.asarray(lon) - self.lon0) * self.m_per_deg_lon,
            (np.asarray(lat) - self.lat0) * self.m_per_deg_lat,
        )

    def to_lonlat(self, x, y):
        return (
            self.lon0 + np.asarray(x) / self.m_per_deg_lon,
            self.lat0 + np.asarray(y) / self.m_per_deg_lat,
        )

    def paint(self, x: float, y: float, base_sec: float, radius_m: float) -> None:
        """(x, y) 를 중심으로 걸어서 퍼지는 소요시간을 격자에 덧칠한다."""
        if radius_m <= 0:
            return
        lo_i = np.searchsorted(self.y, y - radius_m, "left")
        hi_i = np.searchsorted(self.y, y + radius_m, "right")
        lo_j = np.searchsorted(self.x, x - radius_m, "left")
        hi_j = np.searchsorted(self.x, x + radius_m, "right")
        if lo_i >= hi_i or lo_j >= hi_j:
            return

        dy = self.y[lo_i:hi_i] - y
        dx = self.x[lo_j:hi_j] - x
        dist = np.hypot(dy[:, None], dx[None, :])
        sec = base_sec + dist * DETOUR_FACTOR / WALK_SPEED
        window = self.grid[lo_i:hi_i, lo_j:hi_j]
        np.minimum(window, sec.astype(np.float32), out=window)


def build_field(
    g: Graph,
    lon: float,
    lat: float,
    depart_sec: int,
    best: np.ndarray,
    budget_sec: int,
    egress_max_sec: int = EGRESS_WALK_MAX_SEC,
) -> Field:
    """역별 도착 시각을 격자 위의 소요시간 지도로 펼친다."""
    elapsed = best.astype(np.float64) - depart_sec
    usable = (best < INF) & (elapsed <= budget_sec) & np.isfinite(g.coords[:, 0])

    remaining = np.minimum(budget_sec - elapsed, egress_max_sec)
    egress_radius = np.where(usable, remaining, 0.0) * WALK_SPEED / DETOUR_FACTOR
    egress_radius = np.maximum(egress_radius, 0.0)

    # 격자 크기: 가장 멀리 있는 도달역 + 그 역에서의 도보 반경
    if usable.any():
        dist = haversine_m(lon, lat, g.coords[:, 0], g.coords[:, 1])
        span = np.nanmax(np.where(usable, dist + egress_radius, 0.0))
    else:
        span = 0.0
    origin_radius = min(budget_sec, egress_max_sec) * WALK_SPEED / DETOUR_FACTOR
    radius_m = float(max(span, origin_radius) * 1.05 + CELL_SIZE_M)

    field = Field(lon, lat, radius_m)

    # 출발지에서 대중교통 없이 그냥 걸어가는 경우
    field.paint(0.0, 0.0, 0.0, origin_radius)

    idx = np.flatnonzero(usable & (egress_radius > 0))
    xs, ys = field.to_xy(g.coords[idx, 0], g.coords[idx, 1])
    for k, i in enumerate(idx):
        field.paint(float(xs[k]), float(ys[k]), float(elapsed[i]), float(egress_radius[i]))

    return field


# 보행망 노드가 놓인 칸만 값을 갖기 때문에 그대로 등고선을 따면 건물 안쪽이
# 전부 구멍이 된다. 도로에서 이 거리 안은 걸어 들어갈 수 있다고 보고 메운다.
SPREAD_RADIUS_M = 200.0
# 등고선을 딸 격자의 한 변 최대 칸 수
MAX_FIELD_CELLS = 2200
# 출발지 둘레 이 반경은 언제나 권역으로 친다
ORIGIN_SEED_M = 250.0


def _spread(grid: np.ndarray, cell_m: float, radius_m: float) -> np.ndarray:
    """도로 위의 값을 주변 칸으로 번지게 한다 (번진 거리만큼 시간이 붙는다)."""
    reach = int(np.floor(radius_m / cell_m))
    if reach < 1:
        return grid
    out = grid.copy()
    for di in range(-reach, reach + 1):
        for dj in range(-reach, reach + 1):
            if di == 0 and dj == 0:
                continue
            dist = np.hypot(di, dj) * cell_m
            if dist > radius_m:
                continue
            extra = np.float32(dist * DETOUR_FACTOR / WALK_SPEED)
            src = grid[
                max(0, -di): grid.shape[0] - max(0, di),
                max(0, -dj): grid.shape[1] - max(0, dj),
            ]
            dstv = out[
                max(0, di): out.shape[0] - max(0, -di),
                max(0, dj): out.shape[1] - max(0, -dj),
            ]
            np.minimum(dstv, src + extra, out=dstv)
    return out


def _cells_around(walk, lon: float, lat: float, radius_m: float) -> np.ndarray:
    """(lon, lat) 둘레 radius_m 안의 저장 격자 칸 번호."""
    step = int(np.ceil(radius_m / walk.shed_cell_m))
    gx = int(np.floor((lon - walk.lon0) * walk.m_per_deg_lon / walk.shed_cell_m))
    gy = int(np.floor((lat - walk.lat0) * walk.m_per_deg_lat / walk.shed_cell_m))
    dy, dx = np.mgrid[-step:step + 1, -step:step + 1]
    keep = (dx * dx + dy * dy) <= step * step
    cx, cy = gx + dx[keep], gy + dy[keep]
    ok = (cx >= 0) & (cx < walk.shed_w) & (cy >= 0)
    return (cy[ok] * walk.shed_w + cx[ok]).astype(np.int32)


def build_walk_only_field(walk, lon: float, lat: float, budget_sec: int) -> Field:
    """전철을 빼고 걷기만 했을 때의 도달권역.

    역 도보권을 겹칠 것 없이 출발지에서 한 번 퍼뜨리면 끝이다.
    """
    nodes, secs = walk.from_point(lon, lat, budget_sec)
    seed = _cells_around(walk, lon, lat, ORIGIN_SEED_M)
    if len(nodes) == 0:
        return _field_from_cells(
            walk, lon, lat, seed.astype(np.int64), np.zeros(len(seed), np.float32)
        )

    cells = walk.node_shed[nodes]
    ok = cells >= 0
    return _field_from_cells(
        walk, lon, lat,
        np.concatenate([cells[ok].astype(np.int64), seed.astype(np.int64)]),
        np.concatenate([secs[ok].astype(np.float32), np.zeros(len(seed), np.float32)]),
    )


def build_field_network(
    walk,
    lon: float,
    lat: float,
    depart_sec: int,
    best: np.ndarray,
    budget_sec: int,
    egress_max_sec: int = EGRESS_WALK_MAX_SEC,
    origin_walk_max_sec: float | None = None,
) -> Field:
    """실제 보행로를 따라 퍼지는 도달권역.

    역마다 미리 계산해 둔 도보권을 도착 시각만큼 밀어 올려 겹친다. 원을
    그리는 대신 도로를 따르므로 강 건너편이나 선로 반대편으로 새지 않는다.
    """
    elapsed = best.astype(np.float64) - depart_sec
    usable = np.flatnonzero((best < INF) & (elapsed <= budget_sec))

    parts_cell: list[np.ndarray] = []
    parts_time: list[np.ndarray] = []

    for i in usable:
        cells_i, secs = walk.shed(int(i))
        if len(cells_i) == 0:
            continue
        cand = secs.astype(np.float32) + np.float32(elapsed[i])
        keep = (secs <= egress_max_sec) & (cand <= budget_sec)
        if keep.any():
            parts_cell.append(cells_i[keep])
            parts_time.append(cand[keep])

    # 전철을 타지 않고 출발지에서 그냥 걸어가는 경우. 이건 그래프 노드 단위로
    # 나오니 저장 격자 칸으로 옮겨 붙인다.
    # 출발지에서 그냥 걸어가는 경우. 권역 자체는 상한 안이어야 하므로 여기서는
    # 상한을 넘겨 걸을 이유가 없다 (넘기면 그 권역에 속하지 않는다).
    onodes, osecs = walk.from_point(
        lon, lat, min(origin_walk_max_sec or budget_sec, budget_sec)
    )
    if len(onodes):
        ocells = walk.node_shed[onodes]
        ok = ocells >= 0
        if ok.any():
            parts_cell.append(ocells[ok].astype(np.int32))
            parts_time.append(osecs[ok].astype(np.float32))

    if not parts_cell:
        return Field(lon, lat, radius_m=1000.0)

    return _field_from_cells(
        walk, lon, lat, np.concatenate(parts_cell), np.concatenate(parts_time)
    )


def _field_from_cells(walk, lon: float, lat: float,
                      all_cell: np.ndarray, all_time: np.ndarray) -> Field:
    """(격자 칸, 소요 시간) 목록을 등고선을 딸 수 있는 격자로 편다."""
    # 한 칸에 여러 도보권이 겹치므로 칸별 최솟값을 남긴다. np.minimum.at 은
    # 수백만 원소에서 몇 초씩 걸려서, 칸 순으로 정렬한 뒤 구간별로 줄인다.
    order = np.argsort(all_cell, kind="stable")
    all_cell = all_cell[order]
    all_time = all_time[order]
    starts = np.flatnonzero(
        np.concatenate(([True], all_cell[1:] != all_cell[:-1]))
    )
    cells = all_cell[starts].astype(np.int64)
    node_time_active = np.minimum.reduceat(all_time, starts)
    gy, gx = np.divmod(cells, walk.shed_w)

    # 값이 있는 범위 + 번짐 여유만큼만 잘라 쓴다
    pad = int(np.ceil(SPREAD_RADIUS_M / walk.shed_cell_m)) + 1
    x0, x1 = int(gx.min()) - pad, int(gx.max()) + pad
    y0, y1 = int(gy.min()) - pad, int(gy.max()) + pad
    w, h = x1 - x0 + 1, y1 - y0 + 1

    # 너무 넓으면 칸을 묶어 해상도를 낮춘다
    step = max(1, int(np.ceil(max(w, h) / MAX_FIELD_CELLS)))
    cell_m = walk.shed_cell_m * step
    w, h = (w + step - 1) // step, (h + step - 1) // step

    raster = np.full((h, w), np.float32(np.inf), dtype=np.float32)
    ry = (gy - y0) // step
    rx = (gx - x0) // step
    np.minimum.at(raster, (ry, rx), node_time_active)

    raster = _spread(raster, cell_m, SPREAD_RADIUS_M)

    # 전역 격자 좌표를 출발지 기준 미터로 옮긴다
    lon_w = walk.lon0 + (x0 + 0.5) * walk.shed_cell_m / walk.m_per_deg_lon
    lat_s = walk.lat0 + (y0 + 0.5) * walk.shed_cell_m / walk.m_per_deg_lat
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat))
    x_axis = (lon_w - lon) * m_per_deg_lon + np.arange(w) * cell_m
    y_axis = (lat_s - lat) * 111_132.0 + np.arange(h) * cell_m

    field = Field(lon, lat, x=x_axis, y=y_axis, cell=cell_m)
    field.grid = raster
    return field


def _rings_to_polygons(rings: list[np.ndarray]) -> list[list[np.ndarray]]:
    """등고선 고리들을 외곽선과 구멍으로 묶는다.

    matplotlib 이 돌려주는 고리는 외곽/구멍이 섞여 있어서, 면적이 큰 순서로
    훑으며 다른 고리 안에 들어있는지 검사해 포함 깊이를 센다. 깊이가 짝수면
    외곽선, 홀수면 구멍이다.
    """
    from matplotlib.path import Path as MplPath

    def area(r: np.ndarray) -> float:
        x, y = r[:, 0], r[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    items = [(abs(area(r)), r) for r in rings if len(r) >= 4]
    items.sort(key=lambda t: -t[0])
    paths = [MplPath(r) for _, r in items]

    polygons: list[list[np.ndarray]] = []
    owner: list[int] = []  # 각 고리가 속한 외곽 폴리곤 번호

    for i, (_, ring) in enumerate(items):
        depth = 0
        parent = -1
        point = ring[0]
        for j in range(i):
            if paths[j].contains_point(point):
                depth += 1
                parent = j
        if depth % 2 == 0:
            owner.append(len(polygons))
            polygons.append([ring])
        else:
            owner.append(-1)
            # 구멍은 자신을 감싸는 가장 안쪽 외곽선에 붙인다
            host = owner[parent] if parent >= 0 else -1
            if host >= 0:
                polygons[host].append(ring)
    return polygons


def contour_geojson(field: Field, thresholds_sec: list[int]) -> dict:
    """소요시간 격자에서 임계값별 도달권역을 GeoJSON 으로 뽑는다."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    grid = field.grid
    # 무한대는 등고선 계산에서 걸리적거리므로 충분히 큰 유한값으로 바꾼다
    big = float(max(thresholds_sec)) * 10.0
    z = np.where(np.isfinite(grid), grid, big)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    features = []
    try:
        for t in sorted(thresholds_sec):
            if not (z <= t).any():
                continue
            cs = ax.contourf(field.x, field.y, z, levels=[-1.0, float(t)])
            rings: list[np.ndarray] = []
            for path in cs.get_paths():
                rings.extend(p for p in path.to_polygons() if len(p) >= 4)
            if not rings:
                continue
            polygons = _rings_to_polygons(rings)

            coords = []
            for poly in polygons:
                converted = []
                for ring in poly:
                    lon, lat = field.to_lonlat(ring[:, 0], ring[:, 1])
                    converted.append(np.stack([lon, lat], axis=1).round(6).tolist())
                coords.append(converted)

            features.append(
                {
                    "type": "Feature",
                    "properties": {"minutes": int(round(t / 60))},
                    "geometry": {"type": "MultiPolygon", "coordinates": coords},
                }
            )
            ax.clear()
    finally:
        plt.close(fig)

    # 큰 권역이 작은 권역을 덮지 않도록 넓은 것부터 그리게 정렬해 둔다
    features.sort(key=lambda f: -f["properties"]["minutes"])
    return {"type": "FeatureCollection", "features": features}
