"""권역 경계에 대한 회귀 테스트.

여기 있는 것들은 전부 실제로 재발했던 문제다.

- 지도에 그린 보라색 선과 클릭 가능 판정이 서로 다른 마스크를 봐서 어긋났다.
  세 번 재발했고 매번 사용자가 먼저 발견했다.
- 노선의 끝을 "이웃 역이 한쪽에 몰렸는가" 로 짐작하다가, 역 분포의 가장자리에
  있는 중간역(하스다, 가모노미야)까지 끝으로 잡았다.
- 데이터가 끊긴 지점(미나카미, 구로이소, 다카하기)을 진짜 종점으로 보고
  10 km 를 넓혀서, 다룰 수 없는 역(도아이, 다카쿠)이 선 안에 들어왔다.
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def kanto():
    import region as region_mod

    reg = region_mod.load("kanto")
    if reg.coverage is None:
        pytest.skip("보행망이 없습니다 (build_walk.py 미실행)")
    return reg


def station(reg, name):
    """역 이름으로 좌표를 찾는다. 없으면 건너뛴다."""
    for i, ja in enumerate(reg.stops["ja"]):
        if ja == name and np.isfinite(reg.coords[i, 0]):
            return float(reg.coords[i, 0]), float(reg.coords[i, 1])
    pytest.skip(f"{name} 이 데이터에 없습니다")


def north_of(point, km):
    return point[0], point[1] + km / 111.132


def south_of(point, km):
    return point[0], point[1] - km / 111.132


def test_boundary_line_matches_click_gate(kanto):
    """그린 선 안쪽이 곧 클릭 가능한 곳이어야 한다.

    선은 boundary_geojson 이, 판정은 contains 가 낸다. 둘이 다른 마스크를
    보면 "선 안인데 클릭이 안 되는" 상태가 된다. 실제로 세 번 그랬다.
    """
    from matplotlib.path import Path as MplPath

    geo = kanto.coverage_geojson
    rings = []
    for feature in geo.get("features", [geo]):
        geom = feature.get("geometry", feature)
        lines = geom.get("coordinates", [])
        if geom.get("type") == "LineString":
            lines = [lines]
        for line in lines:
            if len(line) >= 4:
                rings.append(MplPath(np.asarray(line, dtype=np.float64)))
    assert rings, "경계선이 비어 있습니다"

    lons = np.concatenate([r.vertices[:, 0] for r in rings])
    lats = np.concatenate([r.vertices[:, 1] for r in rings])

    rng = np.random.default_rng(20260919)
    sample_lon = rng.uniform(lons.min(), lons.max(), 4000)
    sample_lat = rng.uniform(lats.min(), lats.max(), 4000)

    # 고리 안에 몇 겹 들어있는지로 안팎을 가린다 (홀수면 안쪽)
    depth = np.zeros(len(sample_lon), dtype=np.int32)
    points = np.stack([sample_lon, sample_lat], axis=1)
    for ring in rings:
        depth += ring.contains_points(points).astype(np.int32)
    inside_line = (depth % 2) == 1

    gate = np.array([kanto.coverage.contains(float(x), float(y))
                     for x, y in points])

    # 경계 바로 위의 점은 반올림 때문에 갈릴 수 있으므로 비율로 본다
    agree = (inside_line == gate).mean()
    assert agree > 0.98, (
        f"선과 판정이 {(1 - agree) * 100:.1f}% 어긋납니다. "
        "boundary_geojson 과 contains 가 같은 마스크를 보는지 확인하세요."
    )


@pytest.mark.parametrize(
    "name, beyond_km",
    [
        ("水上", 6.0),     # 조에쓰선. 실제로는 유비소·도아이가 이어진다
        ("黒磯", 6.0),     # 우쓰노미야선. 실제로는 다카쿠가 이어진다
        ("高萩", 6.0),     # 조반선. 실제로는 오쓰코를 지나 후쿠시마로 간다
    ],
)
def test_data_edge_does_not_overshoot(kanto, name, beyond_km):
    """데이터가 끊긴 지점은 넉넉히 넓히면 안 된다.

    그 너머로 실제 노선이 이어지고 있어서, 넓히면 우리가 다룰 수 없는 역이
    선 안에 들어온다.
    """
    point = station(kanto, name)
    far = north_of(point, beyond_km)
    assert not kanto.coverage.contains(*far), (
        f"{name} 북쪽 {beyond_km:.0f} km 가 경계 안입니다. "
        "데이터가 끊긴 지점을 종점으로 보고 있지 않은지 확인하세요."
    )


def test_true_terminus_reaches_the_land_end(kanto):
    """진짜 종점 너머는 넉넉히 끌어안아야 한다.

    미사키구치 너머 미우라 시가지는 여기서 자르면 들어갈 수 있는 지도가 없다.
    """
    point = station(kanto, "三崎口")
    assert kanto.coverage.contains(*south_of(point, 3.0)), (
        "미사키구치 남쪽 3 km 가 경계 밖입니다. "
        "노선의 끝에 주는 여유가 사라지지 않았는지 확인하세요."
    )


def test_supported_stations_have_walksheds(kanto):
    """지원으로 표시한 역은 반드시 도보권이 있어야 한다.

    도보권이 없으면 권역에 아무것도 칠하지 못하고 출발지로도 쓸 수 없다.
    그런 역에 점을 찍으면 지도가 거짓말을 한다.
    """
    walk = kanto.walk
    seen = set()
    for i in range(len(kanto.coords)):
        g = int(kanto.station_group[i])
        if g < 0 or not kanto.supported[g] or g in seen:
            continue
        seen.add(g)

    for g in seen:
        rows = np.flatnonzero(kanto.station_group == g)
        assert any(len(walk.shed(int(i))[0]) > 0 for i in rows), (
            f"묶음 {g} 이 지원으로 표시됐는데 도보권이 없습니다"
        )


def test_unsupported_stations_are_outside_or_unclickable(kanto):
    """지원하지 않는 역 주변은 출발지로 쓸 수 없어야 한다.

    보행망이 없으면 그 자리에서 걸어 닿는 역을 셀 수 없다. 경계 안으로
    보이면서 계산이 안 되는 상태가 가장 나쁘다.
    """
    from router import ACCESS_GATE_SEC, access_seconds

    graph = next(iter(kanto.graphs.values()))
    bad = []
    for i in range(len(kanto.coords)):
        g = int(kanto.station_group[i])
        if g < 0 or kanto.supported[g] or not np.isfinite(kanto.coords[i, 0]):
            continue
        lon, lat = float(kanto.coords[i, 0]), float(kanto.coords[i, 1])
        if not kanto.coverage.contains(lon, lat):
            continue
        # 경계 안이라면 적어도 걸어 닿는 역이 있어야 말이 된다
        secs = access_seconds(graph, lon, lat, ACCESS_GATE_SEC, kanto.walk)
        if not bool((secs <= ACCESS_GATE_SEC).any()):
            bad.append(kanto.stops["ja"][i])

    assert not bad, f"경계 안인데 걸어 닿는 역이 없는 지점: {sorted(set(bad))[:10]}"
