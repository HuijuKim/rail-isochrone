"""앱이 실제로 동작하는 범위.

행정구역 경계로는 답이 안 나온다. 도쿄도가 오가사와라까지 포함해서 간토
추출본 폴리곤이 태평양을 통째로 감싸기 때문에, 바다 한복판이 "권역 안" 이
되어 버린다.

보행망이 깔린 범위로 잡아도 틀린다. 도로는 간토 전역에 있지만 우리가 아는
역은 그 일부만 지난다. 그러면 선 안쪽에 "다룰 수 없는 역" 이 잔뜩 들어간다.

그래서 **우리가 실제로 쓰는 역들의 도보권 합집합**을 범위로 삼는다. 이건
이미 계산해 둔 값이라 따로 만들 것도 없다. 여기 안에서 출발하면 반드시
걸어 닿는 역이 있다는 뜻이고, 그게 곧 앱이 답을 낼 수 있는 조건이다.
"""
from __future__ import annotations

import numpy as np

# 범위 마스크 한 칸의 크기. 해안선을 따라 깎을 때 이 크기가 곧 계단의
# 크기가 되므로 촘촘해야 한다.
CELL_M = 1000.0
# 이 반경만큼 부풀렸다 되돌려 노선 사이 빈 곳을 메운다 (칸 수).
# 내륙 윤곽은 "대충 이 언저리" 면 되므로 크게 잡아 덩어리 하나로 만든다.
CLOSE_CELLS = 18
# 역 하나가 끌어안는 반경은 "옆 역까지의 절반" 으로 잡는다. 역이 촘촘한
# 도심은 작게, 역이 드문 외곽은 크게 잡히고, 다음 역이 있는 방향으로는
# 그 중간에서 끊긴다.
HALF_SPACING_MIN_KM = 1.5
HALF_SPACING_MAX_KM = 6.0
# 다만 노선의 끝은 다르다. 그 바깥으로는 다음 역이 없으니 자를 이유가 없고,
# 자르면 미사키구치 너머 미우라 시가지나 다테야마 남쪽처럼 "들어갈 수 있는
# 지도가 없는" 곳이 생긴다. 이런 끝만 이만큼 넉넉히 끌어안는다.
TERMINUS_KM = 10.0
# 넉넉히 잡아도 되는 것은 그 너머에 육지가 없는 곳뿐이다. 육지가 이어지면
# 우리가 안 가진 노선이 그리로 계속 간다. 조에쓰선은 미나카미에서,
# 우쓰노미야선은 구로이소에서, 조반선은 다카하기에서 데이터가 끝나지만
# 실제로는 도아이·다카쿠·오쓰코가 이어진다. 거기를 10 km 넓히면 다룰 수
# 없는 역이 선 안에 들어온다.
#
# "바다가 가까운가" 로는 갈리지 않는다. 다카하기는 해안 도시라 반경 안
# 43%가 바다지만 해안선은 북으로 계속 이어진다. 역이 놓인 방향의 바깥쪽을
# 봐야 한다. 미사키구치·우라가·외카와는 바깥이 3% 이하이고, 다카하기·조시·
# 아와카모가와는 43% 이상이다.
LAND_BEYOND_MAX = 0.25
# 바깥을 살필 때 쓰는 부채꼴 (도) 과 방향 수
OUTWARD_FAN_DEG = 30.0
OUTWARD_RAYS = 7
# 권역 바깥에 남은 육지 조각 중, 권역에 붙어 있고 이만큼보다 작으면 채운다.
# 바다에 막혀 끝나는 반도나 해안 마을을 선으로 가로질러 자를 이유가 없다.
# 다른 지방으로 이어지는 내륙(1만 km² 이상)은 그대로 둔다. 섬은 권역에
# 붙어 있지 않으므로 자연히 빠진다.
MAX_POCKET_KM2 = 3000.0
# 마스크에서 이만큼 떨어진 곳까지는 "권역 안의 바다나 산" 으로 본다.
# 도쿄만 한복판은 뭍에서 10 km 남짓이라 이 안에 들어오고, 태평양이나
# 나고야는 훨씬 멀어서 "권역 밖" 으로 갈린다.
ENVELOPE_M = 30_000.0
# 윤곽선에서 이 길이 미만의 고리는 버린다. 격자 잡음(둘레 2-3 km)만 걸러낼
# 정도로 낮게 둔다. 예전에 60 km 로 잡았더니 주오방파제 매립지(둘레 12 km)
# 처럼 멀쩡한 섬이 통째로 안 그려졌다. 판정은 "안" 인데 선만 빠져서,
# 클릭은 되는데 경계 밖처럼 보이는 상태가 됐다.
MIN_RING_KM = 5.0
# 역에서 걸어 닿는 곳까지를 지원 범위로 본다. 기본값을 쓰면 미우라 반도나
# 보소 남단처럼 역이 드문 곳의 시가지가 통째로 빠진다. 사전 계산해 둔
# 도보권 전체를 쓰고, 상한은 데이터가 가진 만큼으로 둔다.
CATCHMENT_SEC = None


def _perimeter_km(ring: np.ndarray) -> float:
    lat = np.radians(ring[:-1, 1])
    dx = (ring[1:, 0] - ring[:-1, 0]) * np.cos(lat) * 111.320
    dy = (ring[1:, 1] - ring[:-1, 1]) * 111.132
    return float(np.hypot(dx, dy).sum())


class Coverage:
    """보행망이 깔린 범위를 성긴 격자 마스크로."""

    def __init__(self, walk, land_npz=None, catchment_sec: float | None = CATCHMENT_SEC,
                 stations=None, line_ends=None):
        self.lon0, self.lat0 = walk.lon0, walk.lat0
        self.m_lon, self.m_lat = walk.m_per_deg_lon, walk.m_per_deg_lat
        self.cell = CELL_M

        # 역들의 도보권(저장 격자 100 m)을 성긴 격자에 찍는다.
        # 보행망 노드 전체가 아니라 이것이 실제 지원 범위다.
        if stations is not None:
            # 부르는 쪽이 이미 걸러 온다. 보행망에 못 붙는 역(야마나시·이즈)은
            # 애초에 쓸 수 없고, 같은 역의 노선별 중복 행도 합쳐져 있어야 한다.
            # 중복이 섞이면 이웃 거리가 0 이 되어 반경 계산이 무너진다.
            pts = stations
            cx = np.floor((pts[:, 0] - self.lon0) * self.m_lon / CELL_M).astype(np.int64)
            cy = np.floor((pts[:, 1] - self.lat0) * self.m_lat / CELL_M).astype(np.int64)
            self._pending_radius = (pts, line_ends)
        else:
            cells = (walk.shed_cell if catchment_sec is None
                     else walk.shed_cell[walk.shed_sec <= catchment_sec])
            gy, gx = np.divmod(cells.astype(np.int64), walk.shed_w)
            step = CELL_M / walk.shed_cell_m
            cx = (gx / step).astype(np.int64)
            cy = (gy / step).astype(np.int64)

        # 육지 / 바다 마스크 (build_walk.py 가 OSM 해안선으로 구워 둔 것).
        # 반경을 정할 때 "바다에 막힌 끝인가" 를 물어야 하므로 먼저 올린다.
        from pathlib import Path

        self.land = None
        self.land_grid = None
        if land_npz is not None and Path(land_npz).exists():
            z = np.load(land_npz)
            self.land = z["land"]
            self.land_grid = z["grid"]

        if getattr(self, "_pending_radius", None) is not None:
            pts, line_ends = self._pending_radius
            self._radius_cells = self._half_spacing_cells(pts, line_ends)

        self.x0, self.y0 = int(cx.min()), int(cy.min())
        w = int(cx.max()) - self.x0 + 1
        h = int(cy.max()) - self.y0 + 1
        # 가장자리가 잘리지 않도록 여유를 두고 잡는다
        pad = CLOSE_CELLS + int(np.ceil(TERMINUS_KM * 1000 / CELL_M)) + 1
        self.x0 -= pad
        self.y0 -= pad
        w += 2 * pad
        h += 2 * pad
        mask = np.zeros((h, w), dtype=bool)
        mask[cy - self.y0, cx - self.x0] = True
        self.mask = mask

        from scipy import ndimage

        def disk(r: int) -> np.ndarray:
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            return (yy * yy + xx * xx) <= r * r

        # 역마다 제 반경만큼 칠한다
        grown = np.zeros((h, w), dtype=bool)
        if getattr(self, "_radius_cells", None) is not None:
            for r in np.unique(self._radius_cells):
                sel = self._radius_cells == r
                layer = np.zeros((h, w), dtype=bool)
                layer[cy[sel] - self.y0, cx[sel] - self.x0] = True
                grown |= ndimage.binary_dilation(layer, structure=disk(int(r)))
        else:
            grown = mask

        # 노선 사이 빈 곳을 메우고(원으로 해야 모서리가 각지지 않는다)
        # 해안선에 맞춰 깎는다. 이 하나가 "앱이 동작하는 범위" 다.
        region = ndimage.binary_closing(grown, structure=disk(CLOSE_CELLS))
        region = ndimage.binary_fill_holes(region)

        # 넓히다 보면 도쿄만이나 앞바다로 넘친다. 해안선에 맞춰 깎되,
        # 역이 실제로 있는 칸(매립지 등)은 되살린다. 판정용은 이 격자로 두고,
        # 지도에 그릴 선은 육지 마스크 해상도에서 따로 만든다 (아래 참조).
        self.region = region & (self._land_mask_on(region.shape) | mask)
        self._coarse = region
        # 판정과 표시가 같은 선을 쓰도록, 해안선 해상도 마스크를 한 번 만들어 둔다
        self._fine, self._fine_grid = self._fine_region()

        dist_cells = ndimage.distance_transform_edt(~self.region)
        self.envelope = dist_cells <= (ENVELOPE_M / CELL_M)

    def _cell(self, lon: float, lat: float) -> tuple[int, int] | None:
        j = int(np.floor((lon - self.lon0) * self.m_lon / self.cell)) - self.x0
        i = int(np.floor((lat - self.lat0) * self.m_lat / self.cell)) - self.y0
        if not (0 <= i < self.mask.shape[0] and 0 <= j < self.mask.shape[1]):
            return None
        return i, j

    def contains(self, lon: float, lat: float) -> bool:
        """앱이 동작하는 범위 안인가. 지도에 그리는 선과 같은 기준이다."""
        if self._fine is not None:
            lon0, lat0, cell, m_lon, m_lat = self._fine_grid
            j = int(np.floor((lon - lon0) * m_lon / cell))
            i = int(np.floor((lat - lat0) * m_lat / cell))
            if not (0 <= i < self._fine.shape[0] and 0 <= j < self._fine.shape[1]):
                return False
            return bool(self._fine[i, j])
        cell = self._cell(lon, lat)
        return False if cell is None else bool(self.region[cell])

    def in_envelope(self, lon: float, lat: float) -> bool:
        """권역 봉투 안인가.

        "여기 길이 있는가" 는 마스크가 아니라 보행망에 직접 물어야 한다.
        마스크는 격자라 가장자리에서 어긋나고, 그 오차가 그대로 오판이 된다.
        """
        cell = self._cell(lon, lat)
        if cell is None:
            return False
        i, j = cell
        return bool(self.envelope[i, j])

    def is_sea(self, lon: float, lat: float) -> bool | None:
        """바다인가. 육지 마스크가 없으면 None (모름).

        거리로는 바다와 이웃 지방을 가를 수 없다. 앞바다 20 km 와 권역 경계
        밖 20 km 는 같은 거리인데 하나는 바다, 하나는 다른 지방이다. 그래서
        OSM 해안선으로 구운 육지 마스크에 직접 물어본다.
        """
        if self.land is None:
            return None
        lon0, lat0, cell, w, h, m_lon, m_lat = self.land_grid
        j = int(np.floor((lon - lon0) * m_lon / cell))
        i = int(np.floor((lat - lat0) * m_lat / cell))
        if not (0 <= i < self.land.shape[0] and 0 <= j < self.land.shape[1]):
            return None
        return not bool(self.land[i, j])

    def _half_spacing_cells(self, pts: np.ndarray, line_ends=None) -> np.ndarray:
        """역마다 반경을 정한다.

        기본은 "가장 가까운 옆 역까지의 절반". 역이 촘촘한 도심은 작게,
        드문 외곽은 크게 잡히고, 다음 역이 있는 방향으로는 그 중간에서 끊긴다.

        노선의 끝은 바깥으로 다음 역이 없으니 자를 이유가 없다. 다만 넉넉히
        잡아도 되는 것은 바다에 막혀 끝나는 곳뿐이다. 내륙에서 끊긴 끝은
        우리가 안 가진 노선이 그 너머로 이어지고 있어서, 넓히면 다룰 수 없는
        역을 선 안에 넣게 된다.

        끝인지 아닌지는 부르는 쪽이 노선 정보를 보고 정해 온다. 예전에는
        "이웃 역이 한쪽에만 몰려 있는가" 로 짐작했는데, 그건 노선의 끝이
        아니라 역 분포의 가장자리를 재는 것이라 망이 성겨지는 외곽에서는
        하스다·가모노미야 같은 중간역도 끝으로 잡혔다.
        """
        from scipy.spatial import cKDTree

        xy = np.stack([pts[:, 0] * self.m_lon, pts[:, 1] * self.m_lat], axis=1)
        d, _ = cKDTree(xy).query(xy, k=min(2, len(xy)))
        near = d[:, 1] if d.ndim == 2 and d.shape[1] > 1 else np.full(len(xy), np.inf)
        half_km = np.clip(near / 2000.0, HALF_SPACING_MIN_KM, HALF_SPACING_MAX_KM)

        radius_km = half_km
        if line_ends is not None:
            # 바깥 방향 = 가장 가까운 옆 역의 반대쪽
            _, nn = cKDTree(xy).query(xy, k=min(2, len(xy)))
            away = xy - xy[nn[:, 1]] if nn.ndim == 2 and nn.shape[1] > 1 else np.zeros_like(xy)
            norm = np.linalg.norm(away, axis=1, keepdims=True)
            outward = np.divide(away, norm, out=np.zeros_like(away), where=norm > 0)

            generous = (np.asarray(line_ends, dtype=bool)
                        & self._land_ends_outward(pts, outward))
            radius_km = np.where(generous, np.maximum(half_km, TERMINUS_KM), half_km)
        return np.maximum(1, np.rint(radius_km * 1000 / CELL_M)).astype(np.int64)

    def _land_ends_outward(self, pts: np.ndarray, outward: np.ndarray) -> np.ndarray:
        """역이 놓인 방향의 바깥쪽에서 육지가 끝나는가.

        노선의 끝에서 바깥으로 부채꼴을 훑어, 넉넉한 반경의 바깥쪽 절반에
        육지가 얼마나 남았는지 센다. 반도 끝이면 거의 바다이고, 해안선이
        계속 이어지는 곳이면 여전히 육지다.
        """
        if self.land is None:
            return np.ones(len(pts), dtype=bool)

        lon0, lat0, cell, _, _, m_lon, m_lat = self.land_grid
        angles = np.radians(np.linspace(-OUTWARD_FAN_DEG, OUTWARD_FAN_DEG, OUTWARD_RAYS))
        steps = np.arange(TERMINUS_KM * 500.0, TERMINUS_KM * 1000.0 + 1.0, 500.0)

        out = np.zeros(len(pts), dtype=bool)
        for k in range(len(pts)):
            dx, dy = outward[k]
            if dx == 0.0 and dy == 0.0:
                continue
            # 부채꼴로 돌린 방향들 x 거리들
            rx = dx * np.cos(angles) - dy * np.sin(angles)
            ry = dx * np.sin(angles) + dy * np.cos(angles)
            ox = pts[k, 0] + np.outer(rx, steps) / m_lon
            oy = pts[k, 1] + np.outer(ry, steps) / m_lat
            j = np.floor((ox - lon0) * m_lon / cell).astype(np.int64)
            i = np.floor((oy - lat0) * m_lat / cell).astype(np.int64)
            ok = ((i >= 0) & (i < self.land.shape[0])
                  & (j >= 0) & (j < self.land.shape[1]))
            hits = np.zeros(ok.shape, dtype=bool)
            hits[ok] = self.land[i[ok], j[ok]]
            out[k] = hits.mean() <= LAND_BEYOND_MAX
        return out

    def _land_mask_on(self, shape) -> np.ndarray:
        """범위 격자와 같은 모양으로 육지 여부를 옮겨 담는다."""
        if self.land is None:
            return np.ones(shape, dtype=bool)
        lon0, lat0, cell, w, h, m_lon, m_lat = self.land_grid
        gy, gx = np.mgrid[0:shape[0], 0:shape[1]]
        lon = self.lon0 + (gx + self.x0 + 0.5) * self.cell / self.m_lon
        lat = self.lat0 + (gy + self.y0 + 0.5) * self.cell / self.m_lat
        j = np.floor((lon - lon0) * m_lon / cell).astype(np.int64)
        i = np.floor((lat - lat0) * m_lat / cell).astype(np.int64)
        ok = (i >= 0) & (i < self.land.shape[0]) & (j >= 0) & (j < self.land.shape[1])
        out = np.zeros(shape, dtype=bool)
        out[ok] = self.land[i[ok], j[ok]]
        return out

    def _fine_region(self):
        """해안선 해상도에서 다시 만든 범위 마스크와 그 격자.

        범위 자체는 1 km 격자로 잡아도 되지만, 해안선은 지도에서 그대로
        눈에 띄어서 그 해상도로 깎아야 한다. 성긴 마스크를 잘게 펴고
        육지 마스크와 곱한 뒤, 펴면서 생긴 계단을 둥글려 준다.
        """
        from scipy import ndimage

        if self.land is None:
            return None, None
        lon0, lat0, cell, w, h, m_lon, m_lat = self.land_grid
        cell = float(cell)

        # 성긴 격자의 각 칸이 고운 격자의 어느 칸에 해당하는지
        gy, gx = np.mgrid[0:self.land.shape[0], 0:self.land.shape[1]]
        lon = lon0 + (gx + 0.5) * cell / m_lon
        lat = lat0 + (gy + 0.5) * cell / m_lat
        j = np.floor((lon - self.lon0) * self.m_lon / self.cell).astype(np.int64) - self.x0
        i = np.floor((lat - self.lat0) * self.m_lat / self.cell).astype(np.int64) - self.y0
        ok = ((i >= 0) & (i < self._coarse.shape[0])
              & (j >= 0) & (j < self._coarse.shape[1]))

        fine = np.zeros(self.land.shape, dtype=bool)
        fine[ok] = self._coarse[i[ok], j[ok]]

        # 펴면서 생긴 1 km 계단을 둥글린다
        r = int(round(self.cell / cell))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (yy * yy + xx * xx) <= r * r
        fine = ndimage.binary_closing(fine, structure=disk)
        fine = ndimage.binary_opening(fine, structure=disk)
        fine &= self.land

        # 바다에 막혀 끝나는 육지는 통째로 넣는다. 권역에 맞닿아 있으면서
        # 작은 조각만 채우므로, 다른 지방으로 이어지는 내륙이나 멀리 떨어진
        # 섬은 들어오지 않는다.
        gap = self.land & ~fine
        labels, n = ndimage.label(gap)
        if n:
            touching = np.unique(labels[ndimage.binary_dilation(fine) & gap])
            touching = touching[touching > 0]
            areas = np.bincount(labels.ravel(), minlength=n + 1) * (cell / 1000.0) ** 2
            keep = touching[areas[touching] < MAX_POCKET_KM2]
            if len(keep):
                fine |= np.isin(labels, keep)

        return fine, (lon0, lat0, cell, m_lon, m_lat)

    def boundary_geojson(self) -> dict:
        """마스크의 경계선. 지도에 권역 윤곽으로 그린다."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        # 판정과 같은 마스크를 그린다. 성긴 격자를 그리면 선이 판정과 어긋나
        # 육지 한가운데를 가로지르게 된다.
        if self._fine is not None:
            solid = self._fine
            lon0, lat0, cell, m_lon, m_lat = self._fine_grid
            h, w = solid.shape
            xs = lon0 + (np.arange(w) + 0.5) * cell / m_lon
            ys = lat0 + (np.arange(h) + 0.5) * cell / m_lat
        else:
            solid = self.region
            h, w = solid.shape
            xs = self.lon0 + (np.arange(w) + self.x0 + 0.5) * self.cell / self.m_lon
            ys = self.lat0 + (np.arange(h) + self.y0 + 0.5) * self.cell / self.m_lat

        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            cs = ax.contour(xs, ys, solid.astype(np.float32), levels=[0.5])
            rings = []
            for path in cs.get_paths():
                for poly in path.to_polygons(closed_only=False):
                    if len(poly) < 3:
                        continue
                    if _perimeter_km(poly) < MIN_RING_KM:
                        continue
                    # 250 m 격자라 소수점 4자리(약 10 m)면 충분하다
                    rings.append(np.round(poly, 4).tolist())
        finally:
            plt.close(fig)

        return {
            "type": "Feature",
            "properties": {"cell_km": self.cell / 1000},
            "geometry": {"type": "MultiLineString", "coordinates": rings},
        }
