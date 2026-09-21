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
import math
from pathlib import Path

import numpy as np

# 역이 선형에서 이만큼 안쪽에 있어야 그 선형 위에 있다고 본다
SNAP_TOLERANCE_M = 450.0
# 그려진 선의 끝이 역에서 이만큼 안에는 있어야 한다. 선로를 따라 그은
# 선은 끝이 선로 위에 있고 역 중심은 그 옆이라, 큰 역에서는 150m 쯤
# 벌어진다. 그걸 "역에 닿지 않는다" 로 보면 더 거친 선형이 뽑힌다.
REACH_TOLERANCE_M = 200.0
# 잘라낸 구간이 두 역 직선거리의 이 배를 넘으면 엉뚱한 선을 고른 것으로 본다
MAX_DETOUR_RATIO = 2.5
# 역에서 역까지 선로를 따라 찾아 둔 선형의 출처 표시. build_track.py 가
# 구간마다 남겨 둔 것이라 투영도 자르기도 필요 없는, 있는 그대로의 선로다.
TRACK_SRC = "선로"
# 지도에 그릴 때 이보다 큰 걸음이 있으면 선형이 빠진 자리로 보고 끊는다.
# region.SPLIT_GAP_M 이 같은 값을 쓴다. 둘이 어긋나면 여기서 고른 호가
# 지도에서 토막 난다.
#
# 1km 로 두었더니 터널을 잘랐다. OSM 은 긴 터널이나 곧은 구간을 두 점으로
# 그려 두는데, 그 걸음 자체가 옳은 선형이다. 세 권역의 호를 전부 재어
# 보니 0.6~3km 에 324개가 몰려 있고 3km 를 넘는 것은 3개뿐이다. 그 셋은
# 오사키-후지사와 14.8km 처럼 도식적인 선형이 남긴 진짜 현이다. 사이가
# 비어 있어 3km 로 가른다.
DRAW_GAP_M = 3000.0
# 후보를 고를 때 "얼마나 돌아가는가" 를 재는 눈금. 이만큼씩 묶어서 본다.
DETOUR_STEP = 0.25
# 노선 제 선형에 제 역을 붙일 때 이만큼까지 본다. 다른 노선의
# 선형은 보지 않으므로 넉넉해도 엉뚱한 선로를 집지 않는다.
OWN_COVER_M = 400.0
# 후보가 얼마나 거친지 재는 눈금. 이만큼씩 묶어서 본다.
COARSE_STEP_M = 250.0
# 순환선으로 볼 조건. 한 바퀴라 할 만큼 길고, 끝이 제 길이에 견주어
# 거의 맞물려 있어야 한다.
LOOP_MIN_M = 3000.0
LOOP_GAP_M = 1500.0
LOOP_GAP_RATIO = 0.02
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


# 구간 선형이 역을 지나쳐 나갔다가 되돌아오는 꼬리를 자른다.
#
# build_track 은 역 반경 안의 선로 노드 하나를 골라 거기서 거기까지 잇는다.
# 그 노드가 역 너머에 있으면 선이 역을 지나쳐 갔다가, 다음 구간이 같은
# 노드에서 시작해 그대로 되돌아온다. 화면에서는 역에서 뾰족하게 찌르고
# 돌아오는 모양이 된다. 소테츠 계통의 海老名, 긴자선 赤坂見附·溜池山王,
# 사이쿄선 赤羽 가 그 자리다.
#
# 자르는 것은 앞뒤 꼬리뿐이다. 가운데는 건드리지 않는다. 한쪽이 선형의
# 4분의 1을 넘게 잘려 나가면 되돌아온 꼬리가 아니라 우리가 역을 잘못
# 짚은 것이므로 그대로 둔다.
_TRIM_MAX_FRAC = 0.25
_TRIM_MIN_M = 25.0


def _trim_overshoot(arc, pa, pb):
    if arc is None or len(arc) < 4:
        return arc
    a = np.asarray(arc, dtype=np.float64)
    scale = float(np.cos(np.radians(float(pa[1]))))
    def near(p):
        d = np.hypot((a[:, 0] - p[0]) * scale * 111_320.0,
                     (a[:, 1] - p[1]) * 111_132.0)
        return int(np.argmin(d)), float(d.min())
    i, di = near(pa)
    j, dj = near(pb)
    if i >= j:
        return arc
    cap = int(len(a) * _TRIM_MAX_FRAC)
    if i > cap or (len(a) - 1 - j) > cap:
        return arc
    # 잘라 낼 만큼 나가 있지 않으면 그대로 둔다. 몇 m 짜리는 그냥 선로다.
    head = _length_m(a[:i + 1]) if i else 0.0
    tail = _length_m(a[j:]) if j < len(a) - 1 else 0.0
    if head < _TRIM_MIN_M:
        i = 0
    if tail < _TRIM_MIN_M:
        j = len(a) - 1
    if i == 0 and j == len(a) - 1:
        return arc
    return a[i:j + 1]


# 이음매에서 앞 구간의 끝과 뒤 구간의 시작이 같은 길을 되짚는 일이 있다.
# 두 구간이 역 옆 같은 노드를 공유하는데 그 노드가 본선에서 벗어난 자리일
# 때다. 겹친 만큼을 양쪽에서 걷어낸다. 걷어낼 것이 없으면 0 을 돌려준다.
_UNWIND_SAME_M = 5.0
_UNWIND_MAX = 40


def unwind_retrace(path: list, arc) -> int:
    """되짚는 만큼 path 뒤를 지우고, arc 에서 건너뛸 점 수를 돌려준다."""
    k = 0
    while (len(path) >= 2 and k + 1 < len(arc) and k < _UNWIND_MAX
           and _length_m(np.array([path[-2], arc[k + 1]])) < _UNWIND_SAME_M):
        path.pop()
        k += 1
    return k


# 그린 선이 갔다가 되돌아왔다 다시 가는 자리를 걷어낸다.
#
# 원인은 자리마다 다르다. 이음매에서 두 구간이 조금 겹치기도 하고, 빌려
# 온 선형이 몇 점 어긋나기도 하고, 선로 그래프가 회차선을 물고 오기도
# 한다. 그런데 모양은 늘 같다. 짧은 걸음 하나가 거꾸로 박혀 있고 그
# 앞뒤가 이어진다. 사이쿄선 赤羽·大崎·新宿 이 그 자리다.
#
# 얼마나 걷어낼지는 길이로 정하지 않는다. 스위치백이 있는 노선을 따로
# 적어 두는 것은 손으로 적는 일이라 빠뜨리기 쉽다. 대신 **걷어낼 자리에
# 그 노선의 역이 있으면 두는** 것으로 가른다. 지켜야 하는 것은 스위치백
# 자체가 아니라 "선이 제 역에 닿는다" 이기 때문이다. 스위치백은 갔다가
# 같은 선로로 돌아 나오므로, 그 왕복을 지워도 선은 그 자리를 그대로
# 지난다. 역만 안 잘리면 눈에 보이는 손해가 없다.
SPIKE_TURN_DEG = 120.0
SPIKE_MIN_LEG_M = 5.0
# 역에서 되돌아가는 진짜 스위치백만 지킨다(하코네 등산철도 大平台).
# 조건을 좁게 잡는다. 점이 역에 딱 붙어 있고(AT), 빼고 나면 선이 그
# 역에서 한참(AWAY) 떨어질 때만 둔다. 넓게 잡으면 되꺾임이 대개 역
# 근처라 정상 제거까지 막혀 도리어 나빠진다. 재 보고 정한 값이다.
SPIKE_AT_STATION_M = 30.0
SPIKE_AWAY_M = 200.0
SPIKE_MAX_M = 2000.0     # 이보다 길게 되돌아가면 그대로 둔다


def turn_deg(a, b, c, scale):
    """세 점에서 가운데 꼭짓점이 꺾이는 각(도). 0 이면 곧다."""
    v1 = ((b[0] - a[0]) * scale, b[1] - a[1])
    v2 = ((c[0] - b[0]) * scale, c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    d = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def smooth_spikes(path, scale, keep=None):
    """되돌아왔다 다시 가는 점을 뺀다. 뺄 것이 없으면 그대로 돌려준다.

    앞에서부터 한 번 훑으면서, 새 점을 담을 때마다 방금 담은 셋이
    되꺾이는지 본다. 되꺾이면 가운데를 빼고 다시 본다. 뺀 자리에서
    또 되꺾일 수 있어 뒤로 물러 가며 본다.

    keep 은 그 노선의 역 좌표다. 지우려는 점이 역 옆이면 두고 넘어간다.
    """
    if len(path) < 4:
        return path
    K = None
    if keep is not None and len(keep):
        K = np.asarray(keep, dtype=np.float64).reshape(-1, 2)

    def m(a, b):
        return math.hypot((b[0] - a[0]) * scale * 111_320.0,
                          (b[1] - a[1]) * 111_132.0)

    def loses_station(a, b, c):
        """b 를 빼면 선이 역에서 떨어지는가.

        b 가 역 옆이라는 것만으로는 뺄 수 없다는 뜻이 아니다. 빼고 난
        선분 a-c 가 그 역 옆을 그대로 지나면 아무것도 잃지 않는다.
        되꺾임은 대개 역에서 생기므로 이 구분이 없으면 하나도 못 뺀다.
        """
        if K is None:
            return False
        db = np.hypot((K[:, 0] - b[0]) * scale * 111_320.0,
                      (K[:, 1] - b[1]) * 111_132.0)
        k = int(np.argmin(db))
        if db[k] >= SPIKE_AT_STATION_M:
            return False                      # 역에 붙어 있는 점이 아니다
        # 역에서 선분 a-c 까지의 거리
        ax, ay = a[0] * scale * 111_320.0, a[1] * 111_132.0
        cx, cy = c[0] * scale * 111_320.0, c[1] * 111_132.0
        px, py = K[k, 0] * scale * 111_320.0, K[k, 1] * 111_132.0
        vx, vy = cx - ax, cy - ay
        n = vx * vx + vy * vy
        t = 0.0 if n == 0 else max(0.0, min(1.0, ((px - ax) * vx +
                                                  (py - ay) * vy) / n))
        return math.hypot(px - (ax + t * vx), py - (ay + t * vy)) >= SPIKE_AWAY_M

    def sweep(pinned):
        out, idx = [], []
        for i, p in enumerate(path):
            out.append([float(p[0]), float(p[1])])
            idx.append(i)
            while len(out) >= 3:
                a, b, c = out[-3], out[-2], out[-1]
                if idx[-2] in pinned:
                    break
                l1, l2 = m(a, b), m(b, c)
                if l1 < SPIKE_MIN_LEG_M or l2 < SPIKE_MIN_LEG_M:
                    break
                if turn_deg(a, b, c, scale) <= SPIKE_TURN_DEG:
                    break
                if l1 + l2 - m(a, c) > SPIKE_MAX_M * 2:
                    break
                if loses_station(a, b, c):
                    break       # 지우면 선이 그 역에서 떨어진다.
                out.pop(-2)
                idx.pop(-2)
        return out

    out = sweep(set())
    if K is None:
        return out
    # 하나씩은 작게 빼도 되풀이하면 스위치백 끝을 통째로 먹는다. 역에서
    # 30m 안의 점만 지키므로, 끝이 조금 물러나면 그다음부터는 못 막는다.
    # 養老線 은 大垣 에서 방향을 바꾸는데 선이 역에서 924m 떨어졌다.
    # 원래 선이 지나던 역에서 정리한 선이 멀어졌으면 그 역에 가장 가까운
    # 원래 점을 고정하고 다시 정리한다.
    P = np.asarray(path, dtype=np.float64)
    pinned = set()
    for k in range(len(K)):
        d0 = np.hypot((P[:, 0] - K[k, 0]) * scale * 111_320.0,
                      (P[:, 1] - K[k, 1]) * 111_132.0)
        if d0.min() >= SPIKE_AT_STATION_M:
            continue
        Q = np.asarray(out, dtype=np.float64)
        d1 = np.hypot((Q[:, 0] - K[k, 0]) * scale * 111_320.0,
                      (Q[:, 1] - K[k, 1]) * 111_132.0)
        if d1.min() >= SPIKE_AWAY_M:
            pinned.add(int(np.argmin(d0)))
    return sweep(pinned) if pinned else out

class Geometry:
    """노선 선형과, 각 역이 어느 선형 위 어디에 놓이는지."""

    def __init__(self, coordinates_path, station_railway: list[str],
                 coords: np.ndarray, station_ids: list[str] | None = None,
                 segments_path=None):
        from scipy.spatial import cKDTree

        raw = json.loads(Path(coordinates_path).read_text(encoding="utf-8"))
        self.lines: dict[str, np.ndarray] = {}
        for entry in raw["railways"]:
            line = _concat_sublines(entry)
            if len(line) >= 2:
                self.lines[entry["id"]] = line

        # 순환선(야마노테 등)은 양 끝이 맞물린다. 구간을 자를 때 어느 쪽으로
        # 돌지 정해야 하므로 미리 표시해 둔다.
        #
        # 끝 사이 거리만 절대값으로 보면 안 된다. 선형이 노선 하나가 아니라
        # 짧은 토막으로도 들어오게 되면서, 길이 1.5km 짜리 곧은 토막이
        # 끝 사이 1.47km 라는 이유로 순환선이 되어 버린다(간토 OSM 판은
        # 1,570개 중 1,300개가 그랬다). 순환선이라면 제 길이에 견주어 끝이
        # 거의 맞물려 있고, 한 바퀴라 할 만큼 길다.
        self.closed = {}
        for rid, line in self.lines.items():
            gap = _length_m(np.array([line[0], line[-1]]))
            total = _length_m(line)
            self.closed[rid] = (total >= LOOP_MIN_M and gap < LOOP_GAP_M
                                and gap < total * LOOP_GAP_RATIO)

        # build_track.py 가 역에서 역까지 따로 남긴 선로. 있으면 이것이
        # 답이다. 이어 붙인 선형 위에 역을 다시 투영해 자르는 방식은
        # 역 근처에서 모서리를 질러간다. 열쇠는 "노선|역|역" 이다.
        # 노선을 빼면 같은 역 쌓을 지나는 다른 노선끼리 덮어쓴다.
        self.segments = {}
        if segments_path is not None and Path(segments_path).exists():
            table = json.loads(Path(segments_path).read_text(encoding="utf-8"))
            for k, v in table.items():
                bits = k.split("|")
                if len(bits) == 3 and len(v) >= 2:
                    self.segments[tuple(bits)] = np.asarray(v, dtype=np.float64)

        self.ids = list(station_ids) if station_ids else []
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
        # line[hi:] + line[:lo+1] 은 언제나 hi 에서 출발해 lo 에서 끝난다.
        # a <= b 였으면 그 길이 b -> a, 곧 도착 -> 출발 순서라 뒤집어야
        # 한다. a > b 였으면 이미 a -> b 라 그대로 둔다.
        wrap = np.concatenate([line[hi:], line[:lo + 1]])
        if a <= b:
            wrap = wrap[::-1]
        return direct if _length_m(direct) <= _length_m(wrap) else wrap

    def own_pieces(self, rid: str, rows: list[int]):
        """노선 제 선형 위에 제 역을 투영해 구간마다 잘라 낸다.

        후보를 여럿 놓고 점수로 고르는 대신 그 노선의 선형만 본다. 자른
        구간이 곧 그 두 역 사이의 선로라, 이음매가 어긋날 일도 없는
        구간을 현으로 때울 일도 없다. 제 선형이 두 역에 닿지 않는
        자리에서만 끊는데, 거기는 OSM 에 그 구간 선로가 없는 것이다.

        노선 하나의 선형이 여러 조각(rid~1)이나 여러 출처(rid#rel)로
        들어오므로 전부 후보로 두되, 다른 노선의 선형은 보지 않는다.
        """
        base = rid.split("#")[0].split("~")[0]
        variants = [k for k in self.lines
                    if k.split("#")[0].split("~")[0] == base]
        if not variants:
            return []

        proj = {}
        for v in variants:
            line = self.lines[v]
            scale = np.cos(np.radians(float(line[:, 1].mean())))
            lx = line[:, 0] * scale * 111_320.0
            ly = line[:, 1] * 111_132.0
            for i in rows:
                d = np.hypot(lx - self.coords[i, 0] * scale * 111_320.0,
                             ly - self.coords[i, 1] * 111_132.0)
                k = int(np.argmin(d))
                proj[(v, i)] = (float(d[k]), k)

        pieces, path = [], []

        def flush():
            if len(path) >= 2:
                pieces.append(list(path))
            path.clear()

        for a, b in zip(rows, rows[1:]):
            if a == b:
                continue
            straight = _length_m(np.array([self.coords[a], self.coords[b]]))
            best = None
            for v in variants:
                da, ia = proj[(v, a)]
                db, ib = proj[(v, b)]
                if max(da, db) > OWN_COVER_M or ia == ib:
                    continue
                arc = self._arc(v, ia, ib)
                if len(arc) < 2:
                    continue
                # 차고로 들어갔다 나오는 길은 거른다
                if _length_m(arc) > max(straight * MAX_DETOUR_RATIO, 500.0):
                    continue
                if best is None or max(da, db) < best[0]:
                    best = (max(da, db), arc)
            if best is None:
                flush()
                continue
            for q in best[1]:
                pair = [round(float(q[0]), 6), round(float(q[1]), 6)]
                if not path or path[-1] != pair:
                    path.append(pair)
        flush()
        return pieces

    def _operator_rank(self, rid: str, s: int) -> int:
        """같은 회사 선형인가. 0 이면 같은 회사.

        노선 id 는 "Keikyu.Main", "JR-East.Tokaido" 처럼 회사 이름으로
        시작한다. 게이큐 본선 쓰루미 언저리는 JR 과 200m 옆에서 나란히
        달려, 두 역이 JR 선형 열몇 개에도 붙는다. 정작 게이큐 제 선형은
        도식적이라 역에 안 닿고, 그래서 "역에 닿는가" 만 보면 게이큐가
        JR 다카사키선 선로를 따라 그려진다. 같은 회사 선형을 먼저 본다.
        게이큐 구리하마선 선형이 본선 구간을 제대로 덮고 있었다.
        """
        own = self.railway[s]
        mine = own.split(".")[0] if own else ""
        return 0 if mine and rid.split(".")[0] == mine else 1

    def _own_rank(self, rid: str, s: int) -> int:
        """이 선형이 그 역의 노선 것인가. 작을수록 가깝다.

        한 노선의 선형이 여러 조각으로 나뉘어 있을 수 있고(rid~1, rid~2),
        같은 노선의 다른 출처가 함께 들어 있을 수도 있다(rid#mt3d, rid~1#rel).
        조각 번호는 떼고 보되 출처 표시는 남겨야 한다. 둘 다 달린 id 에서
        조각 번호를 먼저 떼면 출처 표시까지 같이 떨어져 나가, 거친 폴백
        선형이 제 노선 선형과 같은 0순위를 받고 더 짧다는 이유로 이긴다.
        출처를 먼저 가른다.
        """
        own = self.railway[s]
        head, _, source = rid.partition("#")
        if head.split("~")[0] != own:
            return 2
        return 1 if source else 0

    def _segment(self, s: int, t: int, rid: str | None = None) -> np.ndarray | None:
        """두 역 사이를 실제 선로로. 마땅한 선형이 없으면 None."""
        return self.segment_source(s, t, rid=rid)[0]

    def track_segment(self, rid, s: int, t: int):
        """그 노선의 두 역 사이 선로. 없으면 None."""
        if not rid or not self.segments or not self.ids:
            return None
        if not (0 <= s < len(self.ids) and 0 <= t < len(self.ids)):
            return None
        sa, sb = self.ids[s], self.ids[t]
        arc = self.segments.get((rid, sa, sb))
        if arc is None:
            arc = self.segments.get((rid, sb, sa))
            if arc is None:
                return None
            arc = arc[::-1]
        return _trim_overshoot(arc, self.coords[s], self.coords[t])

    def segment_source(self, s: int, t: int, prefer: str | None = None,
                       rid: str | None = None):
        """(선형, 어느 선형에서 왔는지).

        prefer 는 앞 구간이 쓴 선형이다. 같은 선형을 계속 쓰면 이음매가
        어긋나지 않는다. 구간마다 제일 좋은 것을 따로 고르면 선형이
        번갈아 바뀌고, 그 자리마다 역에서 수백 m 씩 벌어져 사선이 그어진다.
        도부 고이즈미선이 네 구간에서 세 번 갈아탔다.

        출처를 함께 돌려주는 것은, 그리는 쪽이 "이 걸음이 실제 좌표인가
        없는 구간을 현으로 때운 것인가" 를 가려야 하기 때문이다. 단나
        터널은 7.8km 를 곧게 뚫어 걸음도 7.8km 인데 그게 실제 선형이다.
        길이만 보면 이것과 도식적인 선형의 현을 가를 수 없다.
        """
        # 그 구간의 선로를 이미 찾아 두었으면 고를 것이 없다.
        arc = self.track_segment(rid, s, t)
        if arc is not None:
            return arc, TRACK_SRC

        straight = _length_m(np.array([self.coords[s], self.coords[t]]))
        limit = max(straight * MAX_DETOUR_RATIO, 500.0)

        # 역이 속한 노선을 먼저 보고 싶지만, 그 노선의 선형이 성긴 경우가
        # 많다. 게이오 다카오선은 메이다이마에-기타노를 7점으로 그어 한 걸음이
        # 27 km 다. 같은 구간을 게이오 본선은 453점으로 따라간다. 그래서
        # "촘촘한가" 를 먼저 보고, 그 다음에 소속 노선을 본다.
        candidates = []
        for rid in set(self.on_line[s]) & set(self.on_line[t]):
            arc = self._arc(rid, self.on_line[s][rid], self.on_line[t][rid])
            if len(arc) < 2:
                continue
            arc_len = _length_m(arc)
            if arc_len > limit:
                continue
            step = _max_step_m(arc)
            # 역 사이가 원래 먼 구간은 한 걸음이 클 수밖에 없다
            sparse = step > MAX_BRIDGE_GAP_M and step > straight * 0.5
            # 그리는 쪽은 한 걸음이 DRAW_GAP_M 을 넘으면 거기서 선을
            # 끊는다. 여기서는 1,200m 까지 봐주고 있어 잣대가 어긋났고,
            # 그래서 "괜찮다" 고 고른 호가 지도에서는 두 토막이 됐다.
            # 더 촘촘한 후보가 있으면 그쪽을 고른다.
            holed = step > DRAW_GAP_M
            # 선이 두 역에 실제로 닿는가. 끝이 역에서 멀거나 호가 직선보다
            # 짧으면 그 구간을 다 덮지 못한 것이다. 게이세이 본선은 우에노로
            # 들어가는 지하 구간이 OSM 에 없어 터널 입구에서 끊기는데,
            # 그걸 "자기 노선" 이라는 이유로 고르면 눈에 그대로 보인다.
            reach = max(_length_m(np.array([arc[0], self.coords[s]])),
                        _length_m(np.array([arc[-1], self.coords[t]])),
                        straight * 0.98 - arc_len)
            # 단계로 나눈다. 다 미달일 때 덜 나쁜 쪽이 이기되, 100m 안쪽은
            # 눈에 띄지 않으니 한 단계로 묶어 다른 잣대가 정하게 둔다.
            miss = int(max(reach, 0.0) // REACH_TOLERANCE_M)
            # 크게 돌아가는 호는 대개 차고로 들어갔다 나오거나 엉뚱한
            # 가지를 탄 것이다. 이노카시라선 구가야마-후지미가오카는
            # 803m 인데 차고를 도는 1,959m 짜리 호가 "자기 노선" 이라는
            # 이유로 뽑혀, 선이 선로를 벗어나 그어졌다. 자기 노선인지보다
            # 먼저 본다. 다만 눈금을 굵게 두어, 비슷한 길이끼리는 여전히
            # 자기 노선이 이기게 한다(간다-아키하바라에서 주오선 선로를
            # 따라가지 않도록).
            detour = int(max(arc_len / max(straight, 100.0) - 1.0, 0.0)
                         / DETOUR_STEP)
            # 앞 구간이 쓴 선형을 이어 쓰는 것은 "자기 노선인가" 뒤에
            # 둔다. 앞에 두었더니 한 번 거친 선형을 잡으면 계속 거기
            # 머물러, 선로에서 뽑은 정밀한 선형이 31% 구간에서만 쓰였다.
            # 같은 회사 안에서는 촘촘한 쪽을 먼저 본다. 이것이 없으면
            # 거친 제 노선 선형이 촘촘한 다른 출처를 이긴다. 도부
            # 고이즈미선 히가시코이즈미-고이즈미초가 그랬다(걸음 424m,
            # 1.15배가 걸음 143m, 1.00배를 이겼다).
            rough = int(step / COARSE_STEP_M)
            candidates.append((miss, sparse, holed, detour,
                               self._operator_rank(rid, s), rough,
                               self._own_rank(rid, s),
                               0 if rid == prefer else 1, arc_len, arc, rid))

        if candidates:
            best = min(candidates, key=lambda c: c[:9])
            return best[9], best[10]

        # 이어 붙인 것은 끊긴 구간이 직선이다. 출처를 남기지 않는다.
        return self._bridge(s, t, max(straight * MAX_BRIDGE_RATIO, 500.0)), None

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

    def ride_path(self, stations: list[int],
                  rid: str | None = None) -> list[list[float]]:
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
            arc = self._segment(s, t, rid)
            if arc is None:
                arc = [self.coords[s], self.coords[t]]
            skip = unwind_retrace(out, arc)
            push(arc[skip:])
        return out
