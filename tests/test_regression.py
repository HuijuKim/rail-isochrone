"""기준선을 박아두는 회귀 테스트.

여기 적힌 숫자는 2026-09-19 시점의 시각표와 보행망에서 실측한 값이다.
도보 속도나 환승 시간 같은 상수를 만졌을 때 무엇이 움직이는지 바로 보이게
하는 것이 목적이므로, 값이 달라지면 "고장" 이 아니라 "의도한 변화인지"를
확인하고 기준선을 갱신하면 된다.

시각표를 다시 받으면(fetch_data.py) 다이어 개정으로 소요 시간이 몇 분
움직일 수 있어 여유를 두고 비교한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import minutes_to

DEPART_MORNING = 9 * 3600
DEPART_LATE = 23 * 3600 + 30 * 60
BUDGET = 120 * 60

# 다이어 개정으로 몇 분은 움직일 수 있다
TOLERANCE_MIN = 5


# ---------------------------------------------------------------------------
# 1. 소요 시간 기준값
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "station_id, expected_min",
    [
        ("JR-East.Yamanote.Shibuya", 11),
        ("JR-East.Tokaido.Yokohama", 40),
        ("JR-East.Yokosuka.Kamakura", 64),
    ],
)
def test_travel_time_baseline(graph, walk, station_index, shinjuku, station_id, expected_min):
    """신주쿠 09:00 출발 기준 주요 역 도착 시각."""
    from router import earliest_arrivals

    lon, lat = shinjuku
    best = earliest_arrivals(graph, lon, lat, DEPART_MORNING, BUDGET, walk=walk)

    got = minutes_to(best, DEPART_MORNING, station_index[station_id])
    assert got is not None, f"{station_id} 에 도달하지 못했습니다"
    assert abs(got - expected_min) <= TOLERANCE_MIN, (
        f"{station_id}: 기준 {expected_min}분, 실측 {got:.0f}분"
    )


# ---------------------------------------------------------------------------
# 2. 시간대별 감편이 반영되는가
# ---------------------------------------------------------------------------

def test_late_night_reaches_fewer_stations(graph, walk, shinjuku):
    """심야 출발은 아침보다 닿는 역이 적어야 한다.

    이게 깨지면 시각표의 시간 의존성이 사라졌다는 뜻이다. 출발 시각과 무관하게
    같은 답이 나오면 라우터가 시각을 무시하고 있는 것이다.
    """
    from router import INF, earliest_arrivals

    lon, lat = shinjuku
    budget = 60 * 60
    morning = earliest_arrivals(graph, lon, lat, DEPART_MORNING, budget, walk=walk)
    late = earliest_arrivals(graph, lon, lat, DEPART_LATE, budget, walk=walk)

    n_morning = int((morning < INF).sum())
    n_late = int((late < INF).sum())

    assert n_late < n_morning, f"아침 {n_morning}역, 심야 {n_late}역 — 감편이 반영되지 않았습니다"
    assert n_late > 0, "심야에 아무 역도 닿지 않습니다"


# ---------------------------------------------------------------------------
# 3. 도보권 규모
# ---------------------------------------------------------------------------

def test_walkshed_size(walk):
    """역당 저장 칸 수.

    간선 비용을 셀 중심이 아니라 원본 점 좌표로 재면 격자를 거의 공짜로
    가로지르게 되고, 이 값이 1,700 대에서 11,000 대로 튄다. 그 버그를 잡는
    자리다.
    """
    linked = int((walk.station_node >= 0).sum())
    per_station = len(walk.shed_cell) / linked

    assert 1500 <= per_station <= 2500, (
        f"역당 도보권 칸 {per_station:.0f}개 — 간선 비용이나 도보 상한을 확인하세요"
    )
    assert linked > 2400, f"보행망에 연결된 역이 {linked}개뿐입니다"


# ---------------------------------------------------------------------------
# 4. 닿지 않는 지점
# ---------------------------------------------------------------------------

def test_unreachable_point_is_not_walkable(walk):
    """태평양 한복판은 보행망에 붙지 않아야 한다.

    도보 대안에 상한이 없으면 "걸어서 4,055분" 같은 답이 나온다.
    """
    assert walk.nearest_node(141.5, 34.0) < 0, "바다 위 좌표가 보행망에 붙었습니다"

    nodes, _ = walk.from_point(141.5, 34.0, 60 * 60)
    assert len(nodes) == 0


# ---------------------------------------------------------------------------
# 5. 직통운전 병합
# ---------------------------------------------------------------------------

def test_through_service_merge_logic():
    """nt 체인을 따라 하나의 연속 운행으로 이어야 한다."""
    from build_graph import merge_through_services

    trips = {
        # A -> B -> C 로 이어지는 직통
        "A": {"id": "A", "nt": ["B"], "tt": []},
        "B": {"id": "B", "nt": ["C"], "tt": []},
        "C": {"id": "C", "tt": []},
        # 직통이 없는 단독 운행
        "D": {"id": "D", "tt": []},
    }
    chains = merge_through_services(trips)
    by_len = sorted(([t["id"] for t in c] for c in chains), key=len, reverse=True)

    assert by_len[0] == ["A", "B", "C"], f"직통이 이어지지 않았습니다: {by_len}"
    assert ["D"] in by_len
    assert sum(len(c) for c in chains) == len(trips), "운행이 중복되거나 누락됐습니다"


def test_through_service_merge_handles_cycles():
    """야마노테선처럼 체인 시작점이 없는 순환 구조도 빠뜨리지 않아야 한다."""
    from build_graph import merge_through_services

    trips = {
        "X": {"id": "X", "nt": ["Y"], "tt": []},
        "Y": {"id": "Y", "nt": ["X"], "tt": []},
    }
    chains = merge_through_services(trips)
    assert sum(len(c) for c in chains) == 2


def test_built_graph_trip_count():
    """빌드 산출물의 운행 수. 병합이 바뀌면 여기가 먼저 움직인다."""
    from conftest import REGION

    counts = {}
    for calendar in ("Weekday", "SaturdayHoliday"):
        z = np.load(REGION / f"graph-{calendar}.npz")
        counts[calendar] = len(z["trip_start"]) - 1

    assert counts["Weekday"] == 34243, f"평일 운행 {counts['Weekday']:,}편 (기준 34,243)"
    assert counts["SaturdayHoliday"] == 30035, (
        f"토휴일 운행 {counts['SaturdayHoliday']:,}편 (기준 30,035)"
    )


# ---------------------------------------------------------------------------
# 6. 경로 복원
# ---------------------------------------------------------------------------

def test_trace_does_not_change_arrivals(graph, walk, shinjuku):
    """추적을 켜도 도착 시각은 같아야 한다.

    추적 모드는 최솟값만 취하는 대신 역별로 가장 빠른 후보를 골라내는 다른
    경로를 탄다. 그 과정에서 결과가 달라지면 안 된다.
    """
    from router import earliest_arrivals

    lon, lat = shinjuku
    plain = earliest_arrivals(graph, lon, lat, DEPART_MORNING, BUDGET, walk=walk)
    traced, _ = earliest_arrivals(
        graph, lon, lat, DEPART_MORNING, BUDGET, walk=walk, trace=True
    )
    assert np.array_equal(plain, traced)


def test_journey_is_self_consistent(graph, walk, station_index, shinjuku):
    """복원한 구간들이 시간 순으로 이어지고 도착 시각과 맞아야 한다."""
    from router import earliest_arrivals, reconstruct

    lon, lat = shinjuku
    best, trace = earliest_arrivals(
        graph, lon, lat, DEPART_MORNING, BUDGET, walk=walk, trace=True
    )
    target = station_index["JR-East.Yokosuka.Kamakura"]
    legs = reconstruct(graph, trace, target)

    assert legs and legs[0]["type"] == "access", "출발지 도보로 시작해야 합니다"

    rides = [leg for leg in legs if leg["type"] == "ride"]
    assert rides, "승차 구간이 하나도 없습니다"

    previous_arrival = DEPART_MORNING
    for ride in rides:
        assert ride["depart"] >= previous_arrival, "이전 구간보다 먼저 출발합니다"
        assert ride["arrive"] > ride["depart"]
        previous_arrival = ride["arrive"]

    assert rides[-1]["to"] == target
    assert rides[-1]["arrive"] == int(best[target])


def test_journey_uses_transfer_for_kamakura(graph, walk, station_index, shinjuku):
    """신주쿠에서 가마쿠라는 오후나에서 한 번 갈아타는 경로여야 한다."""
    from router import earliest_arrivals, reconstruct

    lon, lat = shinjuku
    _, trace = earliest_arrivals(
        graph, lon, lat, DEPART_MORNING, BUDGET, walk=walk, trace=True
    )
    legs = reconstruct(graph, trace, station_index["JR-East.Yokosuka.Kamakura"])

    rides = [leg for leg in legs if leg["type"] == "ride"]
    assert len(rides) == 2, f"승차 구간이 {len(rides)}개입니다 (기준 2개)"
    assert sum(1 for leg in legs if leg["type"] == "transfer") == 1
