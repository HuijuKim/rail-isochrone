"""도달 범위 안 장소 수(Google Places Aggregate API).

API 는 구멍 없는 고리 하나씩만 받는다(area_rings). 요청 한 건마다 과금되므로
서버가 대신 부르면서 같은 범위·종류는 기억해 두고, 한 곳(IP)이 한 시간에 세는
횟수와 하루 전체의 요청 수를 막는다. 키는 서버 전용을 쓴다(이 API 만 허용하고,
되도록 서버 IP 로 제한). 브라우저 키로 화면이 직접 부르면 Referer 를 꾸민
요청에 뚫린다.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict, deque

import numpy as np
import shapely

# 화면의 고르기 목록과 같다. 이것 말고는 받지 않는다.
TYPES = ("restaurant", "cafe", "convenience_store", "supermarket", "hospital", "lodging")
PER_IP_HOUR = 20       # 한 곳에서 한 시간에 세는 횟수
PER_DAY = 150          # 하루에 구글로 나가는 요청. 31일이면 4,650건으로 월 무료 5,000건 안이다
KEEP = 2000            # 기억해 두는 (범위, 종류) 수
# 받는 도달 범위의 크기 상한. 고리로 바꾸는 계산(조각 사이 거리 행렬)이 조각 수의
# 제곱이라, 이보다 크면 한도를 세기 전에 서버가 멈출 수 있다. 東京 에서 신칸센 180분이
# 조각 600개, 꼭짓점 1만 8천 개다.
MAX_PARTS = 3000
MAX_VERTICES = 100_000
URL = "https://areainsights.googleapis.com/v1:computeInsights"


class Refused(Exception):
    """세지 않고 돌려보낼 때. 문구는 화면에 그대로 보인다."""

    def __init__(self, message: str, status: int = 429):
        super().__init__(message)
        self.status = status


def area_rings(geom: dict, max_vertices: int = 6900, max_rings: int = 4) -> list:
    """등시선을 구멍 없는 고리 몇 개로. Places Aggregate API 가 고리 하나씩만 받는다.

    조각마다, 구멍마다 따로 물으면 요청이 수십 번이 된다. 떨어진 조각은 폭
    2 m 통로로 잇고(조각 중심점의 최소 신장 트리를 따라), 구멍은 꼭대기에서
    정북으로 바깥이나 다른 구멍까지 폭 20 cm 틈을 내 연다. 북쪽 틈끼리는 서로
    엇갈리지 않고, 서로 다른 경계를 잇는 틈은 땅을 둘로 가르지 않는다. 틈이
    통로보다 좁아야 한다. 같으면 남북으로 난 통로를 따라 올라가며 통로를 통째로
    지워 땅이 갈렸다. 늘고 주는 넓이는 통로·틈 길이 곱하기 폭이라 무시할 만하다.

    꼭짓점이 API 상한(7000)을 넘으면 트리를 따라 가까운 조각끼리 묶어 고리를
    여럿 만든다(도쿄역에서 신칸센 180분은 조각 600개, 꼭짓점 1만 8천 개다).
    그래도 max_rings 를 넘으면 더 줄여 다시 한다. 고리는 반시계 방향이다.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from shapely.geometry import LineString, Polygon, box, shape
    from shapely.geometry.polygon import orient
    from shapely.ops import nearest_points, unary_union

    half, slit = 1e-5, 1e-6    # 통로 반폭(약 1 m), 틈 반폭(약 10 cm)
    base = shape(geom).buffer(0)

    def spanning(parts):
        if len(parts) < 2:
            return []
        c = np.array([[q.centroid.x, q.centroid.y] for q in parts])
        d = np.hypot(*(c[:, None, :] - c[None, :, :]).transpose(2, 0, 1))
        tree = minimum_spanning_tree(d + 1e-12).tocoo()
        return list(zip(tree.row.tolist(), tree.col.tolist()))

    def grouped(parts, budget):
        # 가장 큰 조각부터 트리를 깊이 우선으로 훑으며 꼭짓점 budget 씩 끊는다
        nbr = [[] for _ in parts]
        for i, j in spanning(parts):
            nbr[i].append(j)
            nbr[j].append(i)
        first = max(range(len(parts)), key=lambda i: parts[i].area)
        order, seen, stack = [], {first}, [first]
        while stack:
            i = stack.pop()
            order.append(i)
            for j in nbr[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        # 통로 하나, 틈 하나에 꼭짓점이 6개쯤 붙는다. 고리 수는 최소로, 크기는 고르게.
        size = [shapely.get_num_coordinates(parts[i]) + 6 + 6 * len(parts[i].interiors)
                for i in order]
        k = -(-sum(size) // budget)
        target = sum(size) / k
        out, cur, n = [], [], 0
        for i, v in zip(order, size):
            if cur and n + v > target and len(out) < k - 1:
                out.append(cur)
                cur, n = [], 0
            cur.append(parts[i])
            n += v
        return out + [cur]

    def joined(parts):
        pieces = list(parts)
        for i, j in spanning(parts):
            a, b = nearest_points(parts[i], parts[j])
            v = np.array([b.x - a.x, b.y - a.y])
            n = float(np.hypot(*v))
            if n == 0:
                continue
            # 양 끝을 조각 안으로 조금 들여야 점으로만 맞닿지 않는다
            e = v / n * 3 * half
            pieces.append(LineString([(a.x - e[0], a.y - e[1]), (b.x + e[0], b.y + e[1])])
                          .buffer(half, cap_style="flat"))
        g = unary_union(pieces)
        for _ in range(3):
            polys = [q for q in getattr(g, "geoms", [g]) if q.geom_type == "Polygon"]
            main = max(polys, key=lambda q: q.area)
            if not main.interiors:
                # 틈이 땅을 갈랐으면 떨어져 나간 만큼 덜 센다. 그런 고리는 안 쓴다.
                return main if main.area >= 0.999 * sum(q.area for q in polys) else None
            edge = main.boundary
            top = main.bounds[3] + 1
            cuts = []
            for h in main.interiors:
                xy = np.asarray(h.coords)
                x, y = xy[np.argmax(xy[:, 1])]
                hit = LineString([(x, y), (x, top)]).intersection(edge)
                ys = [q.y for q in getattr(hit, "geoms", [hit])
                      if q.geom_type == "Point" and q.y > y + 1e-9]
                if ys:
                    cuts.append(box(x - slit, y - slit, x + slit, min(ys) + slit))
            g = main.difference(unary_union(cuts))
        return None

    tried = {}

    def at(tol):
        # 줄이는 정도마다 한 번만 나눈다. 고리가 몇 개 필요한지 돌려준다.
        if tol not in tried:
            g = base.simplify(tol, preserve_topology=True)
            # 부스러기는 버리고, 작은 구멍(약 100 m 사방보다 작은 것)은 메운다
            parts = [Polygon(q.exterior, [h for h in q.interiors if Polygon(h).area > 1e-6])
                     for q in getattr(g, "geoms", [g])
                     if q.geom_type == "Polygon" and q.area > 2e-7]
            batch = grouped(parts, max_vertices - 300) if parts else []
            tried[tol] = (len(batch), batch, None)
        return tried[tol][0]

    def build(tol):
        n, batch, rings = tried[tol]
        if rings is None:
            rings = [joined(b) for b in batch]
            ok = all(r is not None and len(r.exterior.coords) <= max_vertices for r in rings)
            rings = [orient(r, 1.0) for r in rings] if ok else []
            tried[tol] = (n, batch, rings)
        return rings

    # 요청은 고리 수만큼 나가므로 고리 수를 먼저 줄인다. 한 고리 안에서는 10 m,
    # 20 m, 40 m 로 줄여 본다(등시선도 도보 시간 어림이라 그만한 오차는 있다).
    plans = [(k, tol) for k in range(1, max_rings + 1) for tol in (1e-4, 2e-4, 4e-4)]
    plans += [(max_rings, tol) for tol in (8e-4, 1.6e-3, 3.2e-3)]
    for k, tol in plans:
        if 0 < at(tol) <= k:
            rings = build(tol)
            if rings:
                return rings
    raise Refused("도달 범위를 고리로 나누지 못했습니다", 400)


class Counter:
    """구글에 묻는 창구. 여러 스레드가 함께 쓴다."""

    def __init__(self, key: str, ask=None, per_ip_hour: int = PER_IP_HOUR,
                 per_day: int = PER_DAY):
        self.key = key
        self.ask = ask or self._ask
        self.per_ip_hour, self.per_day = per_ip_hour, per_day
        self.lock = threading.Lock()
        self.known: OrderedDict = OrderedDict()
        self.recent: dict = {}          # IP -> 최근 한 시간의 세기 시각
        self.day, self.used = None, 0   # 오늘 날짜와 오늘 나간 요청 수

    def count(self, geometry: dict, kind, ip: str, now: float | None = None) -> int:
        if kind not in TYPES:
            raise Refused("모르는 장소 종류입니다: " + repr(kind), 400)
        parts, verts = _size(geometry)
        if parts > MAX_PARTS or verts > MAX_VERTICES:
            raise Refused("도달 범위가 너무 복잡합니다", 400)
        k = (hashlib.sha1(json.dumps(geometry, sort_keys=True).encode()).hexdigest(), kind)
        with self.lock:
            if k in self.known:
                self.known.move_to_end(k)
                return self.known[k]
        now = time.time() if now is None else now
        with self.lock:
            # 무거운 고리 계산 전에 시간당 한도부터 본다. 넘은 곳이 되풀이해 부르면
            # 매번 계산만 하고 거절됐다.
            self._take(ip or "?", 0, now, count=False)
        rings = area_rings(geometry)
        with self.lock:
            self._take(ip or "?", len(rings), now)
        total = sum(self.ask(ring, kind) for ring in rings)
        with self.lock:
            self.known[k] = total
            while len(self.known) > KEEP:
                self.known.popitem(last=False)
        return total

    def _take(self, ip: str, n: int, now: float, count: bool = True) -> None:
        """한도를 넘으면 Refused. count 가 거짓이면 보기만 하고 쓰지 않는다."""
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if day != self.day:
            self.day, self.used = day, 0
        if len(self.recent) > 10000:    # 오래 안 온 곳은 잊는다
            self.recent = {a: q for a, q in self.recent.items() if q and q[-1] > now - 3600}
        log = self.recent.setdefault(ip, deque())
        while log and log[0] <= now - 3600:
            log.popleft()
        if len(log) >= self.per_ip_hour:
            raise Refused(f"장소 수는 한 시간에 {self.per_ip_hour}번까지 셀 수 있습니다. "
                          "잠시 뒤에 다시 해 주세요.")
        if self.used + n > self.per_day or (not count and self.used >= self.per_day):
            raise Refused("오늘 장소 수를 셀 수 있는 양을 다 썼습니다. 내일 다시 해 주세요.")
        if count:
            log.append(now)
            self.used += n


def _size(geometry: dict) -> tuple[int, int]:
    """GeoJSON (Multi)Polygon 의 (고리 수, 꼭짓점 수). 모양이 틀리면 (0, 0)."""
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    polys = coords if geometry.get("type") == "MultiPolygon" else [coords]
    parts = verts = 0
    for poly in polys or ():
        for ring in poly or ():
            parts += 1
            verts += len(ring) if isinstance(ring, list) else 0
    return parts, verts

    def _ask(self, ring, kind: str) -> int:
        body = {
            "insights": ["INSIGHT_COUNT"],
            "filter": {
                "locationFilter": {"customArea": {"polygon": {"coordinates": [
                    {"latitude": y, "longitude": x} for x, y in ring.exterior.coords]}}},
                "typeFilter": {"includedTypes": [kind]},
                "operatingStatus": ["OPERATING_STATUS_OPERATIONAL"],
            },
        }
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "X-Goog-Api-Key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return int(json.load(r).get("count", 0))
        except urllib.error.HTTPError as err:
            try:
                why = json.load(err).get("error", {}).get("message", "")
            except ValueError:
                why = ""
            raise Refused(f"구글이 거절했습니다({err.code}) {why}".strip(), 502) from None
        except (urllib.error.URLError, TimeoutError) as err:
            raise Refused(f"구글에 닿지 못했습니다: {err}", 502) from None
