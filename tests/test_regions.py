"""권역 데이터를 매번 같은 잣대로 재는 검사.

고칠 때마다 손으로 재던 것들이다. 수치는 tests/baseline.json 에 적어
두고, 나빠지면 실패한다. 좋아졌으면 기준선을 새로 적으면 된다.

    python -m pytest tests/test_regions.py
    UPDATE_BASELINE=1 python -m pytest tests/test_regions.py   # 기준선 갱신

검사는 셋으로 나뉜다.

  경계   그린 선이 닫혀 있는가, 그 선 안이 곧 클릭 가능한 곳인가,
         알고 있는 지점이 안팎에 제대로 놓이는가
  노선   이웃 역 사이가 터무니없이 벌어진 노선이 몇 개인가,
         지도에 그릴 때 끊기는 자리가 몇 곳인가
  역     지원으로 표시한 역에 도보권이 있는가

전부 실제로 틀렸던 자리다. 간토 해안선이 육지와 바다를 통째로 뒤바꿔
역이 25개만 남은 적이 있고, 도부 도조선의 역 차례가 뒤섞여 지도에
46 km 짜리 직선이 그어진 적이 있다.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

BASELINE = Path(__file__).resolve().parent / "baseline.json"
UPDATE = os.environ.get("UPDATE_BASELINE") == "1"

# 안팎을 알고 있는 지점. (이름, 경도, 위도, 안이어야 하는가)
PROBES = {
    "kanto": [
        ("도쿄역", 139.767, 35.681, True),
        ("요코하마", 139.628, 35.466, True),
        ("도쿄만 한가운데", 139.80, 35.50, False),
        ("나고야", 136.882, 35.171, False),
    ],
    "kanto_osm": [
        ("도쿄역", 139.767, 35.681, True),
        ("마에바시", 139.064, 36.383, True),
        ("우쓰노미야", 139.898, 36.559, True),
        ("미토", 140.476, 36.371, True),
        ("요코하마", 139.628, 35.466, True),
        ("지바", 140.123, 35.605, True),
        ("도쿄만 한가운데", 139.80, 35.50, False),
        ("나고야", 136.882, 35.171, False),
    ],
    "kansai": [
        ("오사카", 135.500, 34.702, True),
        ("고베 산노미야", 135.1955, 34.6918, True),
        ("간사이공항", 135.244, 34.432, True),
        ("교토", 135.759, 34.985, True),
        ("마이즈루", 135.3335, 35.4497, True),
        ("아마노하시다테", 135.1885, 35.5647, True),
        ("오사카만 한가운데", 135.25, 34.50, False),
        ("와카사만", 135.50, 35.65, False),
        ("나고야", 136.882, 35.171, False),
        # 미에는 도카이 권역에 있다. 간사이와 함께 보려면 현 조합으로 만든다.
        ("쓰", 136.509, 34.734, False),
        ("이가우에노", 136.137, 34.767, False),
    ],
    "tokai": [
        ("나고야", 136.882, 35.171, True),
        ("쓰", 136.509, 34.734, True),
        ("시즈오카", 138.389, 34.972, True),
        ("아타미", 139.078, 35.104, True),
        ("다카야마", 137.252, 36.141, True),
        ("이세만 한가운데", 136.75, 34.85, False),
        ("스루가만", 138.60, 34.90, False),
        ("히코네", 136.262, 35.275, False),
        ("고후", 138.569, 35.667, False),
        ("오사카", 135.500, 34.702, False),
    ],
    "kyushu": [
        ("하카타", 130.4206, 33.5897, True),
        ("고쿠라", 130.8823, 33.8869, True),
        ("나가사키", 129.8708, 32.7524, True),
        ("오이타", 131.6065, 33.2334, True),
        ("가고시마추오", 130.5413, 31.5838, True),
        ("미야자키", 131.4309, 31.9160, True),
        # 야마구치는 주고쿠 권역으로 옮겼다. 규슈와 함께 보려면 현 조합으로 만든다.
        ("시모노세키", 130.9230, 33.9496, False),
        ("겐카이나다", 130.20, 33.90, False),
        ("히로시마", 132.475, 34.397, False),
        ("부산", 129.04, 35.10, False),
    ],
    "chugoku": [
        ("히로시마", 132.4755, 34.3977, True),
        ("오카야마", 133.9177, 34.6660, True),
        ("돗토리", 134.2323, 35.4940, True),
        ("마쓰에", 133.0625, 35.4646, True),
        ("시모노세키", 130.9230, 33.9496, True),
        ("히우치나다", 133.30, 34.10, False),
        ("다카마쓰", 134.0467, 34.3505, False),
        ("하카타", 130.4206, 33.5897, False),
    ],
    "shikoku": [
        ("다카마쓰", 134.0467, 34.3505, True),
        ("마쓰야마", 132.7512, 33.8375, True),
        ("고치", 133.5438, 33.5672, True),
        ("도쿠시마", 134.5510, 34.0747, True),
        ("도사만", 133.60, 33.30, False),
        ("오카야마", 133.9177, 34.6660, False),
    ],
    "tohoku_s": [
        ("센다이", 140.8824, 38.2601, True),
        ("야마가타", 140.3279, 38.2486, True),
        ("후쿠시마", 140.4592, 37.7543, True),
        ("고리야마", 140.3886, 37.3985, True),
        ("센다이만", 141.30, 38.10, False),
        ("모리오카", 141.1363, 39.7016, False),
        ("니가타", 139.0613, 37.9121, False),
    ],
    "tohoku_n": [
        ("모리오카", 141.1363, 39.7016, True),
        ("아오모리", 140.7285, 40.8285, True),
        ("아키타", 140.1263, 39.7175, True),
        ("하치노헤", 141.4331, 40.5091, True),
        ("무쓰만", 140.95, 41.05, False),
        ("센다이", 140.8824, 38.2601, False),
        ("하코다테", 140.7266, 41.7738, False),
    ],
    "hokuriku": [
        ("가나자와", 136.6480, 36.5780, True),
        ("도야마", 137.2133, 36.7013, True),
        ("후쿠이", 136.2240, 36.0621, True),
        ("쓰루가", 136.0759, 35.6454, True),
        ("도야마만", 137.30, 36.85, False),
        ("기후", 136.7588, 35.4094, False),
    ],
    "koshinetsu": [
        ("나가노", 138.1887, 36.6432, True),
        ("니가타", 139.0613, 37.9121, True),
        ("고후", 138.5690, 35.6666, True),
        ("마쓰모토", 137.9722, 36.2310, True),
        ("사도섬", 138.40, 38.00, False),   # 철도도 다리도 없다
        ("나고야", 136.882, 35.171, False),
    ],
}

# 이웃 역 사이가 이보다 벌어지면 순서가 틀렸을 만하다고 본다.
# 특급은 원래 멀어서 0 이 될 수는 없다.
FAR_NEIGHBOUR_KM = 20.0


def _regions():
    """기존 권역만. 현 조합 권역(make_region.py)은 쓰는 사람마다 달라 기준선이 없다."""
    import region as region_mod
    from regional import is_custom

    return [r for r in region_mod.available() if not is_custom(r)]


@pytest.fixture(scope="session")
def loaded():
    """권역 하나를 올리는 데 시간이 걸린다. 한 번만 올려 나눠 쓴다."""
    cache = {}

    def get(region_id):
        if region_id not in cache:
            import region as region_mod

            cache[region_id] = region_mod.load(region_id)
        return cache[region_id]

    return get


@pytest.fixture(scope="session")
def measured():
    """측정값을 모아 두었다가 UPDATE_BASELINE=1 이면 파일로 적는다."""
    got = {}
    yield got
    if UPDATE and got:
        old = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
        for rid, vals in got.items():
            old.setdefault(rid, {}).update(vals)
        BASELINE.write_text(json.dumps(old, ensure_ascii=False, indent=1),
                            encoding="utf-8")


def check(measured, region_id, key, value, limit_is_max=True, slack=0.0):
    """기준선과 견준다. 기준선이 없으면 건너뛴다."""
    measured.setdefault(region_id, {})[key] = value
    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    want = base.get(region_id, {}).get(key)
    if UPDATE:
        return
    if want is None:
        pytest.skip(f"{region_id}/{key} 기준선이 없습니다. "
                    f"측정값 {value}. UPDATE_BASELINE=1 로 적어 두세요.")
    if limit_is_max:
        assert value <= want + slack, (
            f"{key} 가 {want} 에서 {value} 로 나빠졌습니다")
    else:
        assert value >= want - slack, (
            f"{key} 가 {want} 에서 {value} 로 나빠졌습니다")


def rings_of(geo):
    """경계 GeoJSON 에서 고리 목록을 꺼낸다. 선으로도 면으로도 온다."""
    out = []
    for feature in geo.get("features", [geo]):
        g = feature.get("geometry", feature)
        kind, co = g.get("type"), g.get("coordinates", [])
        if kind == "Polygon":
            out += co
        elif kind == "MultiPolygon":
            for poly in co:
                out += poly
        elif kind == "LineString":
            out.append(co)
        elif kind == "MultiLineString":
            out += co
    return [np.asarray(r, dtype=np.float64) for r in out if len(r) >= 4]


# --------------------------------------------------------------------------
# 경계
# --------------------------------------------------------------------------

@pytest.mark.parametrize("region_id", _regions())
def test_boundary_rings_are_closed(loaded, region_id):
    """행정경계로 자른 권역의 경계선은 닫힌 고리여야 한다.

    예전에는 고리를 토막 내어 내보냈고, 토막 사이가 몇십 km 벌어져도
    아무도 몰랐다. 간사이 경계에 54 km 짜리 걸음이 있었다.
    """
    reg = loaded(region_id)
    if not (reg.meta.get("prefectures") or []):
        pytest.skip("행정경계로 자르는 권역이 아닙니다")
    if reg.coverage_geojson is None:
        pytest.skip("보행망이 없습니다 (build_walk.py 미실행)")

    rings = rings_of(reg.coverage_geojson)
    assert rings, "경계선이 비어 있습니다"
    open_rings = [r for r in rings if not np.allclose(r[0], r[-1])]
    assert not open_rings, f"닫히지 않은 고리가 {len(open_rings)}개 있습니다"


@pytest.mark.parametrize("region_id", _regions())
def test_boundary_line_matches_click_gate(loaded, region_id, measured):
    """그린 선 안쪽이 곧 클릭 가능한 곳이어야 한다.

    선은 boundary_geojson 이, 판정은 contains 가 낸다. 둘이 다른 것을
    보면 "선 안인데 클릭이 안 되는" 상태가 된다. 세 번 재발했고 매번
    사용자가 먼저 발견했다. 행정경계로 자르는 권역에서는 이 검사가
    아예 안 돌고 있었다.
    """
    reg = loaded(region_id)
    if reg.coverage is None or reg.coverage_geojson is None:
        pytest.skip("보행망이 없습니다 (build_walk.py 미실행)")
    from matplotlib.path import Path as MplPath

    rings = [MplPath(r) for r in rings_of(reg.coverage_geojson)]
    assert rings, "경계선이 비어 있습니다"

    lons = np.concatenate([r.vertices[:, 0] for r in rings])
    lats = np.concatenate([r.vertices[:, 1] for r in rings])
    rng = np.random.default_rng(20260920)
    pts = np.stack([rng.uniform(lons.min(), lons.max(), 4000),
                    rng.uniform(lats.min(), lats.max(), 4000)], axis=1)

    depth = np.zeros(len(pts), dtype=np.int32)
    for ring in rings:
        depth += ring.contains_points(pts).astype(np.int32)
    inside_line = (depth % 2) == 1
    gate = np.array([reg.coverage.contains(float(x), float(y)) for x, y in pts])

    disagree = float((inside_line != gate).mean())
    check(measured, region_id, "boundary_disagree", round(disagree, 4),
          slack=0.005)


@pytest.mark.parametrize("region_id", _regions())
def test_known_points_land_on_the_right_side(loaded, region_id):
    """안팎을 알고 있는 지점이 제대로 갈려야 한다.

    간토 경계가 육지와 바다를 통째로 뒤바꿔 마에바시와 우쓰노미야가
    권역 밖이 된 적이 있다. 그때 역이 2,742개에서 25개로 줄었다.
    """
    reg = loaded(region_id)
    if reg.coverage is None:
        pytest.skip("보행망이 없습니다 (build_walk.py 미실행)")
    probes = PROBES.get(region_id)
    if not probes:
        pytest.skip(f"{region_id} 의 확인 지점을 정해 두지 않았습니다")

    wrong = [name for name, lon, lat, want in probes
             if reg.coverage.contains(lon, lat) is not want]
    assert not wrong, f"안팎이 뒤바뀐 지점: {wrong}"


# --------------------------------------------------------------------------
# 노선
# --------------------------------------------------------------------------

@pytest.mark.parametrize("region_id", _regions())
def test_neighbour_gaps_do_not_grow(loaded, region_id, measured):
    """이웃 역 사이가 터무니없이 벌어진 노선이 늘면 안 된다.

    벌어졌다는 것은 대개 역 차례가 틀렸다는 뜻이다. 도부 도조선은
    나리마스가 세 번 나와 46 km 가, 료모선은 뒤쪽 토막이 뒤집혀 42 km 가
    나왔다. 특급은 원래 멀어서 0 이 될 수는 없다.
    """
    reg = loaded(region_id)
    row = {sid: i for i, sid in enumerate(reg.stops["ids"])}
    coords = reg.coords
    scale = float(np.cos(np.radians(float(np.nanmedian(coords[:, 1])))))

    far = []
    for rid, railway in reg.railways.items():
        rows = [row[s] for s in (railway.get("stations") or [])
                if s in row and np.isfinite(coords[row[s], 0])]
        if len(rows) < 3:
            continue
        P = coords[rows]
        d = np.hypot(np.diff(P[:, 0]) * scale * 111.320,
                     np.diff(P[:, 1]) * 111.132)
        if len(d) and d.max() > FAR_NEIGHBOUR_KM:
            far.append(railway.get("title", {}).get("ja", rid))
    check(measured, region_id, "far_neighbour_lines", len(far))


@pytest.mark.parametrize("region_id", _regions())
def test_drawn_lines_do_not_break_more(loaded, region_id, measured):
    """지도에 그릴 때 끊기는 자리가 늘면 안 된다.

    한 노선이 여러 조각으로 나오면 그 사이가 끊겨 보인다. 선형이 없거나,
    크게 돌아가거나, 이음매가 어긋나거나, 선형 안에 구멍이 있을 때다.
    """
    reg = loaded(region_id)
    shapes = reg.railway_shapes
    if not shapes:
        pytest.skip("그릴 노선이 없습니다")
    pieces = len(shapes)
    lines = len({r["id"] for r in shapes})
    # 노선 하나가 몇 조각으로 나뉘는가. 1.0 이면 하나도 안 끊긴 것이다.
    check(measured, region_id, "pieces_per_line",
          round(pieces / max(lines, 1), 3), slack=0.05)


# --------------------------------------------------------------------------
# 역
# --------------------------------------------------------------------------

@pytest.mark.parametrize("region_id", _regions())
def test_supported_stations_have_walksheds(loaded, region_id):
    """지원으로 표시한 역은 반드시 도보권이 있어야 한다.

    도보권이 없으면 권역에 아무것도 칠하지 못하고 출발지로도 쓸 수 없다.
    그런 역에 점을 찍으면 지도가 거짓말을 한다.
    """
    reg = loaded(region_id)
    if reg.walk is None:
        pytest.skip("보행망이 없습니다 (build_walk.py 미실행)")

    groups = {int(g) for g in reg.station_group
              if g >= 0 and reg.supported[int(g)]}
    assert groups, "지원으로 표시된 역이 하나도 없습니다"
    bad = []
    for g in groups:
        rows = np.flatnonzero(reg.station_group == g)
        if not any(len(reg.walk.shed(int(i))[0]) > 0 for i in rows):
            bad.append(g)
    assert not bad, f"도보권이 없는데 지원으로 표시된 묶음 {len(bad)}개"


@pytest.mark.parametrize("region_id", _regions())
def test_station_count_does_not_drop(loaded, region_id, measured):
    """역 수가 줄면 안 된다.

    현 경계가 잘못되면 역이 통째로 걸러진다. 간토가 2,742개에서 25개로
    줄어든 적이 있는데, 그때 빌드는 아무 오류 없이 끝났다.
    """
    reg = loaded(region_id)
    n = int(np.isfinite(reg.coords[:, 0]).sum())
    check(measured, region_id, "stations", n, limit_is_max=False, slack=5)


def test_osm_station_order_matches_odpt(measured):
    """OSM 에서 뽑은 역 순서를 ODPT 공식 순서와 견준다.

    간토는 같은 지역을 OSM(kanto_osm)과 ODPT(kanto) 두 벌로 들고 있어,
    build_rail 의 역 순서 규칙을 정답에 대어 볼 수 있는 유일한 곳이다.
    노선마다 역 이름이 60% 넘게 겹치는 ODPT 노선에 짝짓고, 두 노선에
    공통인 역만 남겨 OSM 이웃 쌍 중 ODPT 에서도 이웃인 비율을 잰다.
    선로가 고리처럼 도는 ゆりかもめ·ユーカリが丘線 을 길이로 옮겨 틀렸던
    것을 이걸로 잡았다(99.76% -> 99.95%).
    """
    root = Path(__file__).resolve().parent.parent / "data" / "regions"
    need = [root / "kanto" / "stops.json", root / "kanto" / "raw" / "railways.json",
            root / "kanto_osm" / "raw" / "railways.json"]
    if not all(p.exists() for p in need):
        pytest.skip("간토 두 벌이 다 빌드돼 있어야 한다")
    s = json.loads(need[0].read_text(encoding="utf-8"))
    ja = dict(zip(s["ids"], s["ja"]))
    truth = [[ja.get(x, "") for x in r["stations"]]
             for r in json.loads(need[1].read_text(encoding="utf-8"))]
    st = {x["id"]: x["title"]["ja"] for x in json.loads(
        (root / "kanto_osm" / "raw" / "stations.json").read_text(encoding="utf-8"))}
    lines = [[st[x] for x in r["stations"]]
             for r in json.loads(need[2].read_text(encoding="utf-8"))]

    def pairs(seq):
        return {frozenset(p) for p in zip(seq, seq[1:]) if p[0] != p[1]}

    hit = total = 0
    for seq in lines:
        mine = set(seq)
        if len(mine) < 3:
            continue
        best = max(truth, key=lambda t: len(mine & set(t)))
        if len(mine & set(best)) < 0.6 * len(mine):
            continue
        common = mine & set(best)
        a = pairs([x for x in seq if x in common])
        b = pairs([x for x in best if x in common])
        hit += len(a & b)
        total += len(a)
    check(measured, "kanto_osm", "order_precision", round(hit / max(total, 1), 4),
          limit_is_max=False)


# --------------------------------------------------------------------------
# 노선 색
#
# 같은 노선인데 권역마다 색이 갈리는 일이 거듭 났다. 원인은 늘 같았다.
# 이름을 다듬어 노선을 맞추는데, 다듬으면 사업자가 떨어져 나가 이름만
# 같은 남의 노선과 한 열쇠가 된다. 江ノ島電鉄線 이 小田急江ノ島線 의
# 파랑을, 北大阪急行電鉄南北線 이 東京メトロ南北線 의 에메랄드색을 받았다.
# 반대로 같은 노선을 다르게 부르면 못 맞춰 색이 갈렸다.
# --------------------------------------------------------------------------

def test_same_line_has_one_color_across_regions():
    """역 목록이 거의 같은 노선은 권역이 달라도 색이 같아야 한다."""
    import region as region_mod

    rows = []
    for rid in _regions():
        base = region_mod.REGIONS_DIR / rid
        rp = base / "raw" / "railways.json"
        sp = base / "raw" / "stations.json"
        if not (rp.exists() and sp.exists()):
            continue
        st = {x["id"]: x["title"].get("ja", "")
              for x in json.loads(sp.read_text(encoding="utf-8"))}
        for r in json.loads(rp.read_text(encoding="utf-8")):
            names = {st[x] for x in (r.get("stations") or []) if st.get(x)}
            if len(names) >= 3:
                rows.append((rid, r.get("title", {}).get("ja", "") or r["id"],
                             names, region_mod.line_color(r)))

    bad = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i][0] == rows[j][0]:
                continue
            a, b = rows[i][2], rows[j][2]
            share = len(a & b)
            if share < 3 or share < 0.70 * len(a | b):
                continue
            if rows[i][3].lower() != rows[j][3].lower():
                bad.append(f"{rows[i][1]}({rows[i][3]}) != "
                           f"{rows[j][1]}({rows[j][3]})")
    assert not bad, "역 목록이 같은데 색이 다른 노선 " + str(len(bad)) + "개: " + str(bad[:5])


def test_different_operators_do_not_share_a_color_key():
    """회사가 다른데 이름이 비슷한 노선이 같은 색을 받으면 안 된다."""
    import region as region_mod
    from build_colors import operator_of, same_operator

    seen = {}
    bad = []
    for rid in _regions():
        path = region_mod.REGIONS_DIR / rid / "raw" / "railways.json"
        if not path.exists():
            continue
        for r in json.loads(path.read_text(encoding="utf-8")):
            ja = r.get("title", {}).get("ja", "") or r["id"]
            key = region_mod._norm_name(ja)
            op = operator_of(ja)
            if not key or not op:
                continue
            color = region_mod.line_color(r).lower()
            if (key in seen and not same_operator(seen[key][0], op)
                    and seen[key][1] == color):
                bad.append(f"{key}: {seen[key][0]} 와 {op} 가 {color}")
            seen.setdefault(key, (op, color))
    assert not bad, "회사가 다른데 같은 색을 받은 노선: " + str(bad[:5])


@pytest.mark.parametrize("region_id", _regions())
def test_derived_files_match_rail_build(region_id):
    """build_rail 뒤에 후속 단계를 다 돌렸는가.

    build_rail 만 다시 돌리면 역 번호가 바뀔 수 있는데, 시각표(stops.json)와
    구간 선형은 옛 번호를 들고 남는다. 이름과 순서가 같아도 번호가 달라
    서버가 노선을 엉뚱하게 읽었다(주고쿠 山陰本線 이 由良-荒島 50km 로 보였다).
    시각표가 없는 권역만 본다.
    """
    root = Path(__file__).resolve().parent.parent / "data" / "regions" / region_id
    meta = json.loads((root / "region.json").read_text(encoding="utf-8"))
    if meta.get("model") != "naive":
        pytest.skip("시각표 권역은 역 번호를 ODPT 가 정한다")
    rail = json.loads((root / "raw" / "railways.json").read_text(encoding="utf-8"))
    stops = json.loads((root / "stops.json").read_text(encoding="utf-8"))
    want = {s for r in rail for s in r["stations"]}
    have = set(stops["ids"])
    missing = sorted(want - have)
    assert not missing, (f"stops.json 에 없는 역 {len(missing)}개 "
                         f"(build_track 부터 다시 돌리세요): {missing[:5]}")
