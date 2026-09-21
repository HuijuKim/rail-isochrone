"""노선을 만들 때 쓰는 순수 함수들의 단위 검사.

데이터가 없어도 돈다. 여기 있는 것은 전부 실제로 틀렸던 자리다.

- 사슬을 이을 때 자유로운 끝을 거꾸로 잡아, 이어 붙일 수 있는 웨이가
  한 조각씩 따로 놀았다.
- 웨이를 관계 멤버 순서대로 읽어 역 차례가 뒤죽박죽이 됐다. 도부 도조선은
  나리마스가 세 번 나왔다.
- 순서를 고칠 때 특급의 진짜 긴 구간까지 자를 뻔했다.
- 신칸센 애칭을 이름 한가운데서 찾아 ゆりかもめ 가 통째로 사라졌다.
- 호를 순환선 반대쪽으로 돌 때 방향이 뒤집혀 나왔다.
- 관계에 안 실린 역을 끼울 때 제자리를 못 찾은 것이 목록 맨 앞뒤로 갔다.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("REGION", "kanto_osm")


# --------------------------------------------------------------------------
# 사슬 잇기와 정차 순서
# --------------------------------------------------------------------------

def _line(ids, lon0=139.0, step=0.05, lat=35.7):
    """번호 순서대로 한 줄로 늘어선 역들.

    build_rail 이 쓰는 꼴은 (경도, 위도, 이름들, 이름) 이다. 이름 없는 역
    노드는 fill_missing 이 후보로 안 받으므로 이름을 붙여 둔다.
    """
    return {i: (lon0 + i * step, lat, {}, f"역{i}") for i in ids}


def test_stops_on_ways_orders_scrambled_members():
    """관계에 웨이가 뒤섞여 있어도 노드 번호로 이어 순서를 세운다."""
    from build_rail import stops_on_ways

    refs = {10: [5, 6, 7], 11: [1, 2, 3], 12: [7, 8, 9], 13: [3, 4, 5]}
    stations = _line([1, 3, 5, 7, 9])
    assert stops_on_ways([10, 11, 12, 13], refs, stations, 0.81) == [1, 3, 5, 7, 9]
    assert stops_on_ways([11, 13, 10, 12], refs, stations, 0.81) == [1, 3, 5, 7, 9]
    # 반대 방향으로 세워져도 한 줄이면 된다
    assert stops_on_ways([12, 10, 13, 11], refs, stations, 0.81) == [9, 7, 5, 3, 1]


def test_stops_on_ways_keeps_branch_stations():
    """지선이 붙어 있어도 역을 버리지 않는다."""
    from build_rail import stops_on_ways

    refs = {10: [5, 6, 7], 11: [1, 2, 3], 12: [7, 8, 9], 13: [3, 4, 5],
            14: [5, 100, 101]}
    stations = _line([1, 3, 5, 7, 9])
    stations[101] = (139.25, 35.60, {}, "")   # 5번 역에서 갈라진 지선 끝
    seq = stops_on_ways([10, 11, 12, 13, 14], refs, stations, 0.81)
    assert set(seq) == {1, 3, 5, 7, 9, 101}


def _chain(lon0=139.0, lon1=139.30, lat=35.7):
    """역과 같은 줄에 깔린 선로 좌표열.

    투영은 좌표열의 점까지 재니 점이 촘촘해야 한다. 45m 마다 찍는다.
    """
    n = int((lon1 - lon0) / 0.0005) + 1
    return [(lon0 + i * 0.0005, lat) for i in range(n)]


def test_fill_missing_inserts_a_station_in_order():
    """관계가 안 들고 있는 역을 제 자리에 끼운다."""
    from build_rail import fill_missing

    stations = _line([1, 2, 3, 4, 5])
    geom = {10: _chain()}
    # 2번과 4번만 관계에 실려 있다. 3번은 선로 옆에만 있다.
    seq = fill_missing([1, 2, 4, 5], [10], geom, stations, [3], 0.81)
    assert seq == [1, 2, 3, 4, 5]


def test_fill_missing_keeps_the_list_direction():
    """정차 목록이 거꾸로 서 있어도 그 방향으로 끼운다."""
    from build_rail import fill_missing

    stations = _line([1, 2, 3, 4, 5])
    geom = {10: _chain()}
    seq = fill_missing([5, 4, 2, 1], [10], geom, stations, [3], 0.81)
    assert seq == [5, 4, 3, 2, 1]


def test_fill_missing_adds_a_missing_terminus():
    """종착역이 빠졌으면 목록 끝에 붙인다. JR닛코선의 日光 가 그랬다."""
    from build_rail import fill_missing

    stations = _line([1, 2, 3, 4, 5])
    geom = {10: _chain()}
    # 2·3·4 만 실려 있고 1 과 5 는 양 끝 밖이다.
    assert fill_missing([2, 3, 4], [10], geom, stations, [1, 5], 0.81) == [1, 2, 3, 4, 5]


def test_fill_missing_drops_stations_off_the_middle():
    """붙일 역이 목록 한가운데면 어느 쪽으로 뻗는지 알 수 없어 버린다."""
    from build_rail import fill_missing

    stations = _line([1, 2, 3, 4, 5])
    stations[99] = (139.16, 35.55, {}, "옆선")     # 본선에서 16 km 떨어진 곳
    geom = {10: _chain()}
    # 1·2·3 뒤에 99 가 오지만 99 는 사슬에서 멀어 후보가 못 된다
    assert fill_missing([1, 2, 3], [10], geom, stations, [99], 0.81) == [1, 2, 3]


def test_fill_missing_uses_one_known_station_and_its_neighbour():
    """사슬 위에 아는 역이 하나뿐이면 이웃 역으로 방향을 정해 끼운다."""
    from build_rail import fill_missing

    stations = _line([1, 2, 3, 4, 5])
    # 3·4·5 만 짧은 사슬 위에 있고, 그중 아는 역은 5 하나다.
    geom = {10: _chain(139.14, 139.30)}
    seq = [1, 2, 5]
    assert fill_missing(seq, [10], geom, stations, [3, 4], 0.81) == [1, 2, 3, 4, 5]


def test_fill_missing_skips_a_station_already_there():
    """같은 자리에 이미 정차역이 있으면 이름이 달라도 넘긴다."""
    from build_rail import fill_missing

    stations = _line([1, 2, 3, 4, 5])
    stations[2] = (139.10, 35.7, {}, "浜坂")
    stations[99] = (139.1005, 35.7, {}, "浜坂駅")    # 40m 옆의 같은 역
    geom = {10: _chain()}
    assert fill_missing([1, 2, 3], [10], geom, stations, [99], 0.81) == [1, 2, 3]


def test_fill_missing_survives_a_shredded_track():
    """선로가 토막 나 아는 역이 없는 토막에 있는 역도 제자리에 끼운다.

    関西本線 은 역 구내 측선 때문에 사슬이 191개로 쪼개져, 柘植 너머
    新堂·佐那具·伊賀上野·島ヶ原 이 아는 역 없는 토막에 떨어져 버려졌다.
    """
    from build_rail import fill_missing

    stations = _line([1, 2, 3, 4, 5])
    # 역마다 따로 떨어진 짧은 토막
    geom = {10 + i: _chain(139.0 + i * 0.05 - 0.01, 139.0 + i * 0.05 + 0.01)
            for i in range(1, 6)}
    seq = fill_missing([1, 2], list(geom), geom, stations, [5, 3, 4], 0.81)
    assert seq == [1, 2, 3, 4, 5]


def test_fill_missing_skips_a_station_on_another_track():
    """제 선로 200m 안이어도 가장 가까운 선로가 남의 것이면 버린다.

    JR 長島 가 近鉄長島 옆이라 近鉄名古屋線 에 딸려 왔다.
    """
    from build_rail import fill_missing

    stations = _line([1, 2, 3])
    stations[99] = (139.125, 35.7005, {}, "長島")    # 역 사이 55m 옆, 남의 선로 위
    geom = {10: _chain()}
    owner = {99: {20}}.get
    assert fill_missing([1, 2, 3], [10], geom, stations, [99], 0.81,
                        owner=lambda n: owner(n, {10})) == [1, 2, 3]


def test_fill_missing_keeps_a_junction_terminus():
    """가장 가까운 선로가 남의 것이어도 제 선로 가까운 노선 끝은 받는다.

    参宮線 은 分岐駅 多気 에서 끝나는데, 多気 에 가장 가까운 선로는
    紀勢本線 이다. 한가운데 역만 거르던 조건에 종점이 같이 걸렸다.
    """
    from build_rail import fill_missing

    stations = _line([1, 2, 3])
    stations[99] = (139.2003, 35.7003, {}, "多気")    # 3 너머 노선 끝, 선로 40m 옆
    geom = {10: _chain(139.0, 139.21)}
    assert fill_missing([1, 2, 3], [10], geom, stations, [99], 0.81,
                        owner=lambda n: {20}) == [1, 2, 3, 99]


def test_drop_strays_keeps_the_occurrence_that_fits():
    """같은 역이 두 번 나오면 떠돌이 쪽을 버린다.

    紀勢本線 은 "下里 那智 和歌山市 …" 로 시작하고 下里·那智 가 제자리에
    또 나왔다. 처음 나온 것을 남기면 노선이 331km 에서 427km 가 됐다.
    """
    from build_rail import _drop_strays

    st = _line(range(10))
    xy = lambda n: (st[n][0] * 0.81 * 111_320.0, st[n][1] * 111_132.0)
    seq = [8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert _drop_strays(seq, xy, lambda n: n) == list(range(10))


def test_join_runs_flips_and_places_fragments():
    """토막은 끝 역이 가까운 쪽으로, 방향까지 맞춰 붙인다."""
    from build_rail import join_runs

    st = _line(range(12))
    assert join_runs([[1, 2, 3, 4], [7, 6, 5]], st, 0.81) == [1, 2, 3, 4, 5, 6, 7]
    assert join_runs([[2, 3, 4, 5], [1]], st, 0.81) == [1, 2, 3, 4, 5]
    assert join_runs([[3, 4, 5], [8, 7, 6], [2, 1]], st, 0.81) == [1, 2, 3, 4, 5, 6, 7, 8]
    # 토막이 겹치면 한 번만 담는다
    assert join_runs([[1, 2, 3, 4], [4, 5, 6]], st, 0.81) == [1, 2, 3, 4, 5, 6]
    assert join_runs([[1, 2, 3]], st, 0.81) == [1, 2, 3]
    assert join_runs([], st, 0.81) == []


def test_repair_order_leaves_a_good_sequence_alone():
    """이미 옳은 순서는 건드리지 않는다."""
    from build_rail import repair_order

    cpos = {i: (139.0 + i * 0.05, 35.7) for i in range(12)}
    seq = list(range(12))
    assert repair_order(seq, cpos, 0.81) is seq


def test_repair_order_fixes_stray_terminus_and_flipped_block():
    """맨 뒤로 밀린 종점과 뒤집힌 토막을 제자리로 돌린다.

    아가쓰마선은 시부카와가 맨 뒤에 있어 43 km 짜리 직선이 그어졌고,
    료모선은 뒤쪽 토막이 뒤집혀 마에바시에서 이세사키로 거슬러 갔다.
    """
    from build_rail import repair_order

    cpos = {i: (139.0 + i * 0.05, 35.7) for i in range(12)}
    good = list(range(12))
    assert repair_order(list(range(1, 12)) + [0], cpos, 0.81) == good
    assert repair_order(list(range(6)) + list(range(11, 5, -1)), cpos, 0.81) == good


def test_repair_order_keeps_real_long_gaps():
    """특급의 진짜 긴 구간은 자르지 않는다.

    슈퍼하코네는 신주쿠-오다와라가 68 km 인데 그게 맞다. 시골 노선에
    20 km 짜리 구간이 하나 있는 경우도 마찬가지다.
    """
    from build_rail import repair_order

    express = {0: (139.70, 35.69), 1: (139.15, 35.25), 2: (139.10, 35.23)}
    assert repair_order([0, 1, 2], express, 0.81) == [0, 1, 2]

    rural = {i: (139.0 + i * 0.033, 36.0) for i in range(8)}
    rural[8] = (139.0 + 7 * 0.033 + 0.22, 36.0)
    for i in range(9, 13):
        rural[i] = (rural[8][0] + (i - 8) * 0.033, 36.0)
    seq = list(range(13))
    assert repair_order(seq, rural, 0.81) == seq


def test_repair_order_unfolds_a_round_trip():
    """상행과 하행이 한 줄에 섞여 오면 한 방향으로 편다.

    오다큐 오다와라선은 96개 중 서로 다른 역이 47개였고, 그 이음매에서
    신주쿠와 이세하라가 붙어 48 km 짜리 직선이 그어졌다.
    """
    from build_rail import _max_gap_m, repair_order

    cpos = {i: (139.0 + i * 0.03, 35.6) for i in range(10)}
    there_and_back = [0] + list(range(6, 10)) + list(range(9, 0, -1))
    fixed = repair_order(there_and_back, cpos, 0.81)
    assert set(fixed) == set(there_and_back), "역이 사라지면 안 된다"
    assert (_max_gap_m(fixed, cpos, 0.81)
            < _max_gap_m(there_and_back, cpos, 0.81)), "가장 벌어진 자리가 좁아져야 한다"


# --------------------------------------------------------------------------
# 이름 거르기
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, is_shinkansen",
    [
        ("JR東海道新幹線", True),
        ("のぞみ (東京 => 博多)", True),
        ("こまち列車", True),
        ("とき (東京 => 新潟)", True),
        ("あさま (東京 => 長野)", True),
        # 아래는 신칸센이 아니다. 애칭이 이름 한가운데 들어 있을 뿐이다.
        ("ゆりかもめ (新橋 → 豊洲)", False),
        ("ときわ (品川 => いわき)", False),
        ("きぬがわ列車 : 鬼怒川温泉→新宿", False),
        ("さくら夙川行き", False),
    ],
)
def test_shinkansen_filter_matches_whole_names(name, is_shinkansen):
    """신칸센 애칭은 이름의 끝이거나 구분 기호 옆일 때만 본다.

    그냥 찾으면 ゆりかもめ 가 かもめ 에, ときわ 가 とき 에 걸려 노선째
    사라진다. 실제로 둘 다 사라져 있었다.
    """
    from build_rail import SHINKANSEN

    assert bool(SHINKANSEN.search(name)) is is_shinkansen


@pytest.mark.parametrize(
    "name, kind",
    [
        ("踊り子 (伊豆急下田=>東京)", "특급"),
        ("ひだ", "특급"),
        ("しなの (名古屋 => 長野)", "특급"),
        ("サンダーバード: 大阪 -> 敦賀", "특급"),
        # 애칭이 회사·노선 이름 앞머리일 뿐이다.
        ("しなの鉄道線 (軽井沢 => 篠ノ井)", None),
        ("JR伊東線", None),
    ],
)
def test_named_limited_express_is_not_a_line(name, kind):
    """特急 이 안 적힌 특급 애칭도 특급으로 본다.

    종별 없는 노선으로 읽히면 정차역이 많아 뼈대로 먼저 뽑히고, 진짜
    노선을 부분 계통으로 밀어낸다. 伊東線 이 踊り子 밑으로 들어가
    熱海·宇佐美·伊東 를 잃었다.
    """
    from build_rail import kind_of

    assert kind_of(name) == kind


def test_express_kind_labels_match_build_rail():
    """위키에서 읽은 종별 이름을 build_rail 과 같은 말로 옮긴다.

    다르면 build_naive 의 중복 제거 키 (railway, kind) 가 영영 안 맞아,
    이미 있는 통과 계통 위에 한 벌이 더 깔리고 배차가 절반이 된다.
    """
    from build_rail import KINDS, kind_of

    labels = {k for k, _ in KINDS}
    for header in ("特急", "急行", "快速", "新快速", "通勤快速", "準急"):
        assert kind_of(header) in labels, header


# --------------------------------------------------------------------------
# 선형 자르기
# --------------------------------------------------------------------------

def _geometry(line, closed_id="loop"):
    """선형 하나만 담은 Geometry 를 만든다."""
    import json
    import tempfile
    from pathlib import Path

    from geometry import Geometry

    coords = np.array([line[0], line[-1]], dtype=np.float64)
    entry = {"id": closed_id,
             "sublines": [{"type": "main", "coords": [list(p) for p in line]}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump({"railways": [entry], "airways": []}, f)
        path = f.name
    try:
        return Geometry(path, [closed_id, closed_id], coords)
    finally:
        Path(path).unlink(missing_ok=True)


def test_arc_keeps_direction_when_wrapping():
    """순환선 반대쪽으로 돌 때도 출발 -> 도착 차례여야 한다.

    뒤집혀 나오면 그려진 선이 거꾸로 달리고, 이웃 구간과 이어 붙일 때
    양 끝에 긴 직선이 남는다.
    """
    n = 24
    ring = [(139.7 + 0.02 * np.cos(2 * np.pi * k / n),
             35.7 + 0.02 * np.sin(2 * np.pi * k / n)) for k in range(n)]
    ring.append(ring[0])
    geo = _geometry(ring)
    assert geo.closed["loop"], "이 선형은 순환선으로 잡혀야 한다"

    for a, b in ((2, 20), (20, 2)):
        arc = geo._arc("loop", a, b)
        head = np.hypot(*(np.array(arc[0]) - np.array(ring[a])))
        tail = np.hypot(*(np.array(arc[-1]) - np.array(ring[b])))
        assert head < 1e-9 and tail < 1e-9, (
            f"a={a} b={b} 인데 호가 {arc[0]} -> {arc[-1]} 로 나왔다")


def test_short_stub_is_not_a_loop():
    """짧고 곧은 토막을 순환선으로 보면 안 된다.

    끝 사이 거리만 절대값으로 보던 때, 간토 OSM 판은 선형 1,570개 중
    1,300개가 순환선으로 잡혔다. 그러면 있지도 않은 반대쪽 길을 탄다.
    """
    stub = [(139.700 + 0.002 * k, 35.700) for k in range(8)]   # 약 1.3 km
    geo = _geometry(stub, "stub")
    assert not geo.closed["stub"]
