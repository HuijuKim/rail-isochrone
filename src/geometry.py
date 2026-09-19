"""경로를 지도에 그릴 선으로 바꾼다.

역과 역을 직선으로 이으면 실제 선로와 어긋난다. mini-tokyo-3d 의
coordinates.json 에 노선별 선형이 들어 있으므로 역 사이 구간만 잘라 쓴다.

주의할 점이 하나 있다. 쇼난신주쿠라인처럼 다른 노선의 선로를 빌려 쓰는
계통은 자체 선형이 거의 비어 있다. 그래서 "역이 속한 노선" 만 보지 않고,
그 역 근처를 지나는 아무 노선의 선형이나 후보로 두고 두 역을 모두 지나는
것을 고른다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 역이 선형에서 이만큼 안쪽에 있어야 그 선형 위에 있다고 본다
SNAP_TOLERANCE_M = 350.0
# 잘라낸 구간이 두 역 직선거리의 이 배를 넘으면 엉뚱한 선을 고른 것으로 본다
MAX_DETOUR_RATIO = 2.5
# 두 노선을 이어 붙일 때는 더 깐깐하게 본다. 잘못 이으면 크게 돌아가는데,
# 그건 직선으로 긋는 것보다 오히려 눈에 거슬린다.
MAX_BRIDGE_RATIO = 1.8
# 이어 붙인 경로에서 한 걸음이 이 길이를 넘으면 버린다. 전체 길이만 줄이려
# 들면 엉뚱한 방향으로 조금 간 뒤 크게 점프하는 경로를 고르게 되는데,
# 지도에서는 그 점프가 그대로 긴 직선으로 보인다. 이음매뿐 아니라 선형
# 자체가 성긴 구간도 같은 문제를 일으키므로 경로 전체를 본다.
MAX_BRIDGE_GAP_M = 1200.0


def _max_step_m(line: np.ndarray) -> float:
    if len(line) < 2:
        return 0.0
    lat = np.radians(line[:-1, 1])
    dx = (line[1:, 0] - line[:-1, 0]) * np.cos(lat) * 111_320.0
    dy = (line[1:, 1] - line[:-1, 1]) * 111_132.0
    return float(np.hypot(dx, dy).max())


def _concat_sublines(entry: dict) -> np.ndarray:
    """subline 들을 순서대로 이어 하나의 선형으로."""
    pts: list[list[float]] = []
    for sub in entry.get("sublines", []):
        coords = sub.get("coords") or []
        if pts and coords and pts[-1] == coords[0]:
            coords = coords[1:]
        pts.extend(coords)
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))


def _length_m(line: np.ndarray) -> float:
    if len(line) < 2:
        return 0.0
    lat = np.radians(line[:-1, 1])
    dx = (line[1:, 0] - line[:-1, 0]) * np.cos(lat) * 111_320.0
    dy = (line[1:, 1] - line[:-1, 1]) * 111_132.0
    return float(np.hypot(dx, dy).sum())


class Geometry:
    """노선 선형과, 각 역이 어느 선형 위 어디에 놓이는지."""

    def __init__(self, coordinates_path, station_railway: list[str], coords: np.ndarray):
        from scipy.spatial import cKDTree

        raw = json.loads(Path(coordinates_path).read_text(encoding="utf-8"))
        self.lines: dict[str, np.ndarray] = {}
        for entry in raw["railways"]:
            line = _concat_sublines(entry)
            if len(line) >= 2:
                self.lines[entry["id"]] = line

        # 순환선(야마노테 등)은 양 끝이 맞물린다. 구간을 자를 때 어느 쪽으로
        # 돌지 정해야 하므로 미리 표시해 둔다.
        self.closed = {
            rid: _length_m(np.array([line[0], line[-1]])) < 1500.0
            for rid, line in self.lines.items()
        }

        self.railway = list(station_railway)
        self.coords = coords

        # 모든 노선의 점을 한 트리에 모아, 역 근처를 지나는 노선을 한 번에 찾는다
        rids, idxs, pts = [], [], []
        for rid, line in self.lines.items():
            rids.append(np.full(len(line), rid, dtype=object))
            idxs.append(np.arange(len(line)))
            pts.append(line)
        self._pt_rid = np.concatenate(rids)
        self._pt_idx = np.concatenate(idxs)
        flat = np.concatenate(pts)

        scale = np.cos(np.radians(35.7))
        tree = cKDTree(np.stack([flat[:, 0] * scale * 111_320.0,
                                 flat[:, 1] * 111_132.0], axis=1))

        # 역 -> {노선: 선형 위 위치}
        self.on_line: list[dict[str, int]] = []
        for i in range(len(station_railway)):
            if not np.isfinite(coords[i, 0]):
                self.on_line.append({})
                continue
            q = [coords[i, 0] * scale * 111_320.0, coords[i, 1] * 111_132.0]
            near = tree.query_ball_point(q, SNAP_TOLERANCE_M)
            best: dict[str, tuple[float, int]] = {}
            for k in near:
                rid = self._pt_rid[k]
                d = float(np.hypot(flat[k, 0] * scale * 111_320.0 - q[0],
                                   flat[k, 1] * 111_132.0 - q[1]))
                if rid not in best or d < best[rid][0]:
                    best[rid] = (d, int(self._pt_idx[k]))
            self.on_line.append({rid: v[1] for rid, v in best.items()})

    def _arc(self, rid: str, a: int, b: int) -> np.ndarray:
        """선형 위 두 위치 사이를 잘라낸다. 순환선이면 짧은 쪽으로 돈다."""
        line = self.lines[rid]
        direct = line[a:b + 1] if a <= b else line[b:a + 1][::-1]
        if not self.closed.get(rid):
            return direct

        lo, hi = (a, b) if a <= b else (b, a)
        wrap = np.concatenate([line[hi:], line[:lo + 1]])
        if a > b:
            wrap = wrap[::-1]
        return direct if _length_m(direct) <= _length_m(wrap) else wrap

    def _segment(self, s: int, t: int) -> np.ndarray | None:
        """두 역 사이를 실제 선로로. 마땅한 선형이 없으면 None."""
        straight = _length_m(np.array([self.coords[s], self.coords[t]]))
        limit = max(straight * MAX_DETOUR_RATIO, 500.0)

        # 역이 속한 노선을 먼저 보고 싶지만, 그 노선의 선형이 성긴 경우가
        # 많다. 게이오 다카오선은 메이다이마에-기타노를 7점으로 그어 한 걸음이
        # 27 km 다. 같은 구간을 게이오 본선은 453점으로 따라간다. 그래서
        # "촘촘한가" 를 먼저 보고, 그 다음에 소속 노선을 본다.
        candidates = []
        for rid in set(self.on_line[s]) & set(self.on_line[t]):
            arc = self._arc(rid, self.on_line[s][rid], self.on_line[t][rid])
            if len(arc) < 2 or _length_m(arc) > limit:
                continue
            step = _max_step_m(arc)
            # 역 사이가 원래 먼 구간은 한 걸음이 클 수밖에 없다
            sparse = step > MAX_BRIDGE_GAP_M and step > straight * 0.5
            candidates.append((sparse, rid != self.railway[s], _length_m(arc), arc))

        if candidates:
            return min(candidates, key=lambda c: c[:3])[3]

        return self._bridge(s, t, max(straight * MAX_BRIDGE_RATIO, 500.0))

    def _bridge(self, s: int, t: int, limit: float) -> np.ndarray | None:
        """두 역을 함께 지나는 선형이 없을 때, 각자의 선형을 이어 붙인다.

        요코스카선의 신카와사키-요코하마 구간처럼 원본 선형이 중간에 끊긴
        데가 있다. 이럴 때 그냥 직선을 그으면 10 km 가 일직선으로 나온다.
        대신 출발역 쪽 선형과 도착역 쪽 선형에서 서로 가장 가까워지는 지점을
        찾아 거기서 갈아탄다. 끊긴 구간만 직선이 되고 나머지는 선로를 탄다.
        """
        if not self.on_line[s] or not self.on_line[t]:
            return None

        best_arc = None
        best_len = limit
        for ra, ia in self.on_line[s].items():
            la = self.lines[ra]
            for rb, ib in self.on_line[t].items():
                if ra == rb:
                    continue
                lb = self.lines[rb]
                # 두 선형이 가장 가까워지는 지점 (성기게 훑어 비용을 아낀다)
                step_a = max(1, len(la) // 400)
                step_b = max(1, len(lb) // 400)
                sub_a, sub_b = la[::step_a], lb[::step_b]
                scale = np.cos(np.radians(float(self.coords[s, 1])))
                dx = (sub_a[:, None, 0] - sub_b[None, :, 0]) * scale
                dy = sub_a[:, None, 1] - sub_b[None, :, 1]
                flat = int(np.argmin(dx * dx + dy * dy))
                ja = (flat // len(sub_b)) * step_a
                jb = (flat % len(sub_b)) * step_b

                arc_a = self._arc(ra, ia, min(ja, len(la) - 1))
                arc_b = self._arc(rb, min(jb, len(lb) - 1), ib)
                if len(arc_a) < 2 or len(arc_b) < 2:
                    continue
                joined = np.concatenate([arc_a, arc_b])

                # 어디서든 긴 직선이 끼면 이을 값어치가 없다
                if _max_step_m(joined) > MAX_BRIDGE_GAP_M:
                    continue

                length = _length_m(joined)
                if length < best_len:
                    best_len, best_arc = length, joined

        return best_arc

    def ride_path(self, stations: list[int]) -> list[list[float]]:
        """정차역 목록을 실제 선로를 따르는 선으로.

        직통운전은 도중에 노선이 바뀌므로 역 쌍마다 따로 고른다. 마땅한
        선형이 없는 구간만 직선으로 잇는다.
        """
        out: list[list[float]] = []

        def push(points) -> None:
            for p in points:
                pair = [round(float(p[0]), 6), round(float(p[1]), 6)]
                if not out or out[-1] != pair:
                    out.append(pair)

        for k in range(len(stations) - 1):
            s, t = stations[k], stations[k + 1]
            arc = self._segment(s, t)
            push(arc if arc is not None else [self.coords[s], self.coords[t]])
        return out
