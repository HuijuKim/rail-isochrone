"""장소 수 세기: 등시선을 고리로 바꾸기, 기억과 한도."""
import math
from pathlib import Path

import pytest
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from shapely.ops import unary_union

import placecount
from placecount import Counter, Refused, area_rings


def _star(cx, cy, n=400, r1=0.02, r2=0.015):
    """꼭짓점이 많은 별. 가시(폭 약 50 m, 깊이 500 m)가 줄이는 폭(10-40 m)보다 커서
    줄여도 꼭짓점이 남는다. 더 가늘면 줄일 때 가시 모양이 바뀌어 넓이 비교가 흔들린다."""
    return Polygon([(cx + (r1 if i % 2 else r2) * math.cos(2 * math.pi * i / n),
                     cy + (r1 if i % 2 else r2) * math.sin(2 * math.pi * i / n))
                    for i in range(n)])


def _check(rings, geom, limit):
    for r in rings:
        assert r.geom_type == "Polygon" and r.is_valid
        assert not r.interiors
        assert r.exterior.is_ccw
        assert len(r.exterior.coords) <= limit
    whole = unary_union(rings)
    # 통로·틈·줄이기로 늘고 주는 넓이는 아주 작아야 한다
    assert whole.symmetric_difference(geom).area < 0.005 * geom.area


def test_holes_and_pieces_become_one_ring():
    big = box(139.70, 35.60, 139.80, 35.70)
    holes = [box(139.72, 35.62, 139.73, 35.63), box(139.75, 35.66, 139.77, 35.68)]
    main = Polygon(big.exterior, [h.exterior for h in holes])
    far = [box(139.85 + 0.02 * i, 35.61, 139.855 + 0.02 * i, 35.615) for i in range(3)]
    geom = MultiPolygon([main] + far)
    rings = area_rings(mapping(geom))
    assert len(rings) == 1
    _check(rings, geom, 6900)


def test_many_vertices_split_into_rings():
    geom = MultiPolygon([_star(139.7 + 0.05 * i, 35.6) for i in range(10)])
    rings = area_rings(mapping(geom), max_vertices=2000)
    assert 2 <= len(rings) <= 4
    _check(rings, geom, 2000)


def _counter(**kw):
    calls = []

    def ask(ring, kind):
        calls.append(kind)
        return 7

    return Counter("key", ask=ask, **kw), calls


def _square(i):
    return mapping(box(139.7 + 0.01 * i, 35.6, 139.705 + 0.01 * i, 35.605))


def test_same_area_is_asked_once():
    c, calls = _counter()
    assert c.count(_square(0), "cafe", "1.1.1.1", now=0) == 7
    assert c.count(_square(0), "cafe", "1.1.1.1", now=1) == 7
    assert calls == ["cafe"]
    c.count(_square(0), "restaurant", "1.1.1.1", now=2)
    assert calls == ["cafe", "restaurant"]


def test_unknown_type_is_refused():
    c, calls = _counter()
    with pytest.raises(Refused) as err:
        c.count(_square(0), "night_club", "1.1.1.1", now=0)
    assert err.value.status == 400 and not calls


def test_per_ip_limit():
    c, _ = _counter(per_ip_hour=2)
    c.count(_square(0), "cafe", "1.1.1.1", now=0)
    c.count(_square(1), "cafe", "1.1.1.1", now=10)
    with pytest.raises(Refused) as err:
        c.count(_square(2), "cafe", "1.1.1.1", now=20)
    assert err.value.status == 429
    c.count(_square(2), "cafe", "2.2.2.2", now=20)      # 다른 곳은 된다
    c.count(_square(3), "cafe", "1.1.1.1", now=3601)    # 한 시간이 지나면 된다


def test_per_day_limit():
    c, _ = _counter(per_day=3)
    for i in range(3):
        c.count(_square(i), "cafe", f"10.0.0.{i}", now=0)
    with pytest.raises(Refused):
        c.count(_square(3), "cafe", "10.0.0.9", now=0)
    c.count(_square(3), "cafe", "10.0.0.9", now=86400 * 2)   # 다음 날은 된다


def test_types_match_screen():
    """화면의 고르기 목록(poi-type)과 서버가 받는 종류가 같아야 한다."""
    page = (Path(placecount.__file__).resolve().parent.parent / "web" / "index.html").read_text(
        encoding="utf-8")
    block = page.split('<select id="poi-type">', 1)[1].split("</select>", 1)[0]
    shown = tuple(part.split('"', 1)[0] for part in block.split('value="')[1:])
    assert shown == placecount.TYPES
