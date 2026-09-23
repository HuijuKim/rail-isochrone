"""출발 지점과 출발 시각이 주어졌을 때 모든 역의 최早 도착 시각을 구한다.

RAPTOR 를 numpy 로 벡터화한 라운드 방식이다. 라운드마다 전체 정차 이벤트를
한 번씩 훑으면서 (1) 지금까지의 최선 도착 시각으로 탈 수 있는 열차를 찾고
(2) 그 열차의 이후 정차역 도착 시각을 갱신한다. 더 나아지는 역이 없을 때까지
반복하므로 환승 횟수에 제한을 두지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 도보 속도 (m/s). 분당 80 m 는 일본 부동산 광고의 관행적 기준이다.
WALK_SPEED = 80.0 / 60.0
# 직선거리를 실제 보행거리로 보정하는 계수
DETOUR_FACTOR = 1.30
# 출발지에서 역까지 걸어가는 시간에는 따로 상한을 두지 않는다. 소요 시간
# 상한 자체가 한계이기 때문이다. 3시간 안에 도착하면 되는 여정이라면 3시간을
# 걷는 것도 선택지다 (실제로는 거의 항상 전철이 빠르므로 답에 안 나온다).
# 지점을 쓸 수 있는지 미리 걸러낼 때만 이 넉넉한 기본값을 쓴다.
ACCESS_GATE_SEC = 60 * 60
# "제한 없음" 일 때 쓰는 사실상의 무제한. 가장 가까운 역이 몇 시간 거리인
# 벽지에서도 답이 나오게 하려는 값이라 소요 시간 상한과 무관하게 크다.
# 완전히 무한이면 탐색이 끝나지 않으므로 현실적인 천장만 둔다.
ACCESS_UNLIMITED_SEC = 6 * 60 * 60
# 열차에서 내려 같은 역에서 다시 탈 때 필요한 최소 여유
MIN_BOARD_BUFFER = 60

INF = np.int32(2**31 - 1)


@dataclass
class Graph:
    coords: np.ndarray
    ev_stop: np.ndarray
    ev_arr: np.ndarray
    ev_dep: np.ndarray
    trip_start: np.ndarray
    tr_to: np.ndarray
    tr_cost: np.ndarray
    tr_ptr: np.ndarray
    # 이벤트마다 소속 운행의 시작 인덱스 (라운드 계산에 쓰는 보조 배열)
    ev_trip_start: np.ndarray

    @property
    def n_stations(self) -> int:
        return len(self.coords)


def load_graph(path) -> Graph:
    """빌드해 둔 시각표 배열을 올린다."""
    z = np.load(path)
    trip_start = z["trip_start"]
    counts = np.diff(trip_start)
    ev_trip_start = np.repeat(trip_start[:-1], counts)
    return Graph(
        coords=z["coords"],
        ev_stop=z["ev_stop"],
        ev_arr=z["ev_arr"],
        ev_dep=z["ev_dep"],
        trip_start=trip_start,
        tr_to=z["tr_to"],
        tr_cost=z["tr_cost"],
        tr_ptr=z["tr_ptr"],
        ev_trip_start=ev_trip_start,
    )


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """위경도 배열 사이의 대권거리(미터)."""
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def walk_seconds(distance_m: np.ndarray) -> np.ndarray:
    return distance_m * DETOUR_FACTOR / WALK_SPEED


def access_seconds(g: Graph, lon: float, lat: float, limit_sec: float,
                   walk=None) -> np.ndarray:
    """지점에서 각 역까지 걸어가는 시간. 보행망이 있으면 실제 도로를 따른다.

    되돌림은 "보행망에 아예 붙지 못하는 지점" 에만 쓴다. 붙어 있는데 역이
    안 잡힌다고 직선거리로 넘어가면, 없는 연결을 지어내서 상한을 넓힐수록
    오히려 도달 역이 줄어드는 이상한 일이 생긴다.
    """
    if walk is not None and walk.nearest_node(lon, lat) >= 0:
        return walk.station_times_from_point(lon, lat, limit_sec, g.n_stations)
    return walk_seconds(haversine_m(lon, lat, g.coords[:, 0], g.coords[:, 1]))


def initial_labels(g: Graph, lon: float, lat: float, depart_sec: int,
                   limit_sec: float, walk=None):
    """출발 지점에서 걸어서 닿는 역들에 최초 라벨을 심는다.

    (도착 시각, 걸은 시간)을 함께 돌려준다. 걸은 시간은 같은 시각에 닿는
    길이 여럿일 때 덜 걷는 쪽을 고르는 데 쓴다.
    """
    best = np.full(g.n_stations, INF, dtype=np.int32)
    secs = access_seconds(g, lon, lat, limit_sec, walk)
    reachable = np.isfinite(secs) & (secs <= limit_sec)
    best[reachable] = (depart_sec + secs[reachable]).astype(np.int32)
    return best, secs


def relax_transfers(g: Graph, best: np.ndarray, trace: Trace | None = None,
                    walked: np.ndarray | None = None) -> np.ndarray:
    """역 구내 환승 간선을 한 번 완화한다."""
    if len(g.tr_to) == 0:
        return best
    counts = np.diff(g.tr_ptr)
    src = np.repeat(np.arange(g.n_stations), counts)
    base = best[src]
    alive = base < INF
    if not alive.any():
        return best
    cand = base[alive].astype(np.int64) + g.tr_cost[alive]

    if trace is None:
        np.minimum.at(best, g.tr_to[alive], np.minimum(cand, INF).astype(np.int32))
        return best

    # 경로를 복원하려면 어느 역에서 넘어왔는지도 남겨야 해서, 최솟값만 취하는
    # 대신 도착역별로 가장 빠른 후보와 그 출발역을 함께 고른다.
    dst = g.tr_to[alive]
    src_alive = src[alive]
    if walked is None:
        keys, values, picks = _best_per_group(dst, cand)
        improved = values < best[keys]
    else:
        # 시각이 같으면 덜 걷는 쪽. 片瀬江ノ島 에서 江ノ島 까지 9분을 걸어도
        # 目白山下 까지 15분을 걸어도 같은 모노레일을 타는데, 시각만 보면
        # 둘이 같아 먼저 훑은 쪽이 남았다.
        cand_walk = walked[src_alive].astype(np.int64) + g.tr_cost[alive]
        keys, values, picks = _best_per_group(dst, cand, cand_walk)
        wpick = cand_walk[picks]
        improved = (values < best[keys]) | ((values == best[keys])
                                            & (wpick < walked[keys]))
    trace.set_transfer(keys[improved], values[improved], src_alive[picks][improved])
    best[keys[improved]] = values[improved].astype(np.int32)
    if walked is not None:
        walked[keys[improved]] = np.minimum(wpick[improved], INF).astype(np.int32)
    return best


def _least_walk_boarding(g, sel, boardable, walked, ev_stop, fallback):
    """고른 하차 이벤트마다 같은 운행 안에서 덜 걷고 탈 수 있는 승차 이벤트.

    도착 시각은 같은 운행 안이면 어디서 타든 같으므로, 여기서 고르는 것은
    "어디서 탔다고 적을지" 뿐이다. 걸은 시간이 같으면 늦게 타는 쪽으로 둔다.
    """
    out = np.asarray(fallback).astype(np.int64).copy()
    for k, e in enumerate(np.asarray(sel).tolist()):
        start = int(g.ev_trip_start[e])
        if e <= start:
            continue
        ok = np.flatnonzero(boardable[start:e])
        if len(ok) == 0:
            continue
        w = walked[ev_stop[start:e][ok]].astype(np.int64)
        out[k] = start + int(ok[np.lexsort((-ok, w))[0]])
    return out


def _best_per_group(keys: np.ndarray, values: np.ndarray, second=None):
    """키별 최솟값과 그 원소의 위치. 키 순, 값 순으로 정렬해 첫 원소를 고른다.

    second 를 주면 값이 같을 때 그것이 작은 쪽을 고른다(걸은 시간).
    """
    order = (np.lexsort((values, keys)) if second is None
             else np.lexsort((second, values, keys)))
    k = keys[order]
    first = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
    picks = order[first]
    return k[first], values[picks], picks


class Trace:
    """각 역에 어떻게 도달했는지. 경로 복원에만 쓴다.

    kind: 0 도달 못함 / 1 출발지에서 도보 / 2 역 구내 환승 / 3 승차
    승차는 (승차 이벤트, 하차 이벤트), 환승은 (직전 역, -1) 을 담는다.
    """

    UNREACHED, ACCESS, TRANSFER, RIDE = 0, 1, 2, 3

    def __init__(self, n_stations: int):
        self.kind = np.zeros(n_stations, dtype=np.int8)
        self.a = np.full(n_stations, -1, dtype=np.int64)
        self.b = np.full(n_stations, -1, dtype=np.int64)

    def set_access(self, stations: np.ndarray) -> None:
        self.kind[stations] = self.ACCESS
        self.a[stations] = -1
        self.b[stations] = -1

    def set_transfer(self, stations, _values, from_stations) -> None:
        self.kind[stations] = self.TRANSFER
        self.a[stations] = from_stations
        self.b[stations] = -1

    def set_ride(self, stations, board_events, alight_events) -> None:
        self.kind[stations] = self.RIDE
        self.a[stations] = board_events
        self.b[stations] = alight_events


def earliest_arrivals(
    g: Graph,
    lon: float,
    lat: float,
    depart_sec: int,
    horizon_sec: int,
    max_rounds: int = 12,
    walk=None,
    trace: bool = False,
    access_limit: float | None = None,
):
    """각 역의 최早 도착 시각(자정 기준 초). 도달 불가면 INF.

    trace 를 켜면 (도착 시각, Trace) 를 함께 돌려준다. 경로를 복원하려면
    어느 열차를 어디서 탔는지까지 남겨야 하는데, 등시선에는 필요 없는
    비용이라 도착지를 조회할 때만 켠다.
    """
    deadline = depart_sec + horizon_sec
    # 출발지에서 역까지 걷는 시간의 상한. 따로 주지 않으면 소요 시간 상한이
    # 곧 한계다 (그 안에 도착해야 하므로).
    best, access_secs = initial_labels(
        g, lon, lat, depart_sec, access_limit if access_limit is not None else horizon_sec, walk
    )
    best[best > deadline] = INF

    tr = Trace(g.n_stations) if trace else None
    walked = None
    if tr is not None:
        tr.set_access(np.flatnonzero(best < INF))
        # 역마다 여기까지 오며 걸은 시간. 등시선에는 필요 없는 셈이라
        # 경로를 복원할 때만 센다.
        walked = np.full(g.n_stations, INF, dtype=np.int32)
        live = best < INF
        walked[live] = np.minimum(access_secs[live], INF).astype(np.int32)

    best = relax_transfers(g, best, tr, walked)
    best[best > deadline] = INF

    ev_stop, ev_arr, ev_dep = g.ev_stop, g.ev_arr, g.ev_dep
    positions = np.arange(len(ev_stop), dtype=np.int64)

    for _ in range(max_rounds):
        prev = best.copy()

        # (1) 각 정차 이벤트에서 승차가 가능한가
        ready = best[ev_stop].astype(np.int64) + MIN_BOARD_BUFFER
        boardable = (ready <= ev_dep) & (ev_dep <= deadline)

        # (2) 같은 운행 안에서 "이 이벤트 이전에 이미 탔는가" 를 구한다.
        #     승차 가능 지점의 인덱스를 누적 최대로 흘려보낸 뒤, 그 값이 해당
        #     운행의 시작 인덱스 이상이면 이미 탑승한 상태다.
        marks = np.where(boardable, positions, -1)
        running = np.maximum.accumulate(marks)
        boarded_before = np.empty_like(running)
        boarded_before[0] = -1
        boarded_before[1:] = running[:-1]
        onboard = boarded_before >= g.ev_trip_start

        # (3) 탑승 중인 열차의 도착 시각으로 각 역 라벨을 갱신
        hit = onboard & (ev_arr <= deadline)
        if hit.any():
            if tr is None:
                np.minimum.at(best, ev_stop[hit], ev_arr[hit])
            else:
                idx = np.flatnonzero(hit)
                if walked is None:
                    keys, values, picks = _best_per_group(ev_stop[idx], ev_arr[idx])
                    improved = values < best[keys]
                else:
                    # 타고 가는 동안은 걷지 않는다. 탄 역까지 걸은 시간을 그대로 옮긴다.
                    ride_walk = walked[ev_stop[boarded_before[idx]]].astype(np.int64)
                    keys, values, picks = _best_per_group(ev_stop[idx], ev_arr[idx], ride_walk)
                    wpick = ride_walk[picks]
                    improved = (values < best[keys]) | ((values == best[keys])
                                                        & (wpick < walked[keys]))
                sel = idx[picks][improved]
                board = boarded_before[sel]
                if walked is not None:
                    # 도착 시각은 어느 역에서 타든 같다. 그러면 덜 걷고 타는
                    # 역으로 적는다. 江ノ島線 片瀬江ノ島 에서 모노레일을 탈 때
                    # 9분 걸어 湘南江の島 에서 타나 16분 걸어 目白山下 에서
                    # 타나 같은 열차인데, 늘 늦게 타는 쪽이 적혔다.
                    board = _least_walk_boarding(g, sel, boardable, walked,
                                                 ev_stop, board)
                tr.set_ride(keys[improved], board, sel)
                best[keys[improved]] = values[improved].astype(np.int32)
                if walked is not None:
                    walked[keys[improved]] = np.minimum(wpick[improved], INF).astype(np.int32)

        best = relax_transfers(g, best, tr, walked)
        best[best > deadline] = INF

        if np.array_equal(best, prev):
            break

    return (best, tr) if trace else best


def reconstruct(g: Graph, trace: Trace, station: int, max_legs: int = 40) -> list[dict]:
    """도착역에서 출발지 쪽으로 거슬러 올라가 여정을 구간별로 편다.

    구간마다 도착 시각이 반드시 줄어들므로(환승 비용과 승차 시간이 모두
    양수다) 되짚기는 반드시 끝난다. max_legs 는 혹시 모를 안전장치다.
    """
    legs: list[dict] = []
    cur = int(station)

    for _ in range(max_legs):
        kind = int(trace.kind[cur])

        if kind == Trace.ACCESS:
            legs.append({"type": "access", "to": cur})
            break

        if kind == Trace.TRANSFER:
            prev = int(trace.a[cur])
            legs.append({"type": "transfer", "from": prev, "to": cur})
            cur = prev
            continue

        if kind == Trace.RIDE:
            board, alight = int(trace.a[cur]), int(trace.b[cur])
            path = g.ev_stop[board:alight + 1].tolist()
            legs.append(
                {
                    "type": "ride",
                    "from": int(g.ev_stop[board]),
                    "to": int(g.ev_stop[alight]),
                    "depart": int(g.ev_dep[board]),
                    "arrive": int(g.ev_arr[alight]),
                    "path": path,
                }
            )
            cur = int(g.ev_stop[board])
            continue

        break  # 도달하지 못한 역

    legs.reverse()
    return legs
