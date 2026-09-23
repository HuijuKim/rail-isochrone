"""현 조합 권역(make_region.py, regional.py) 단위 검사. 빌드한 데이터 없이 돈다."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import make_region  # noqa: E402
import regional  # noqa: E402


def test_resolve_takes_names_in_any_language():
    got = make_region.resolve(["香川", "오카야마현", "Okayama", "岡山県", "가가와"])
    assert got == ["岡山県", "香川県"]          # 겹친 것은 하나로, 현 번호 순으로


def test_resolve_does_not_eat_the_name():
    """都府県 을 뗄 때 京都府 가 京 이 되면 안 된다."""
    assert make_region.resolve(["京都", "교토부", "東京", "tokyo"]) == ["東京都", "京都府"]


@pytest.mark.parametrize("name", ["서울", "北海道", "沖縄"])
def test_resolve_refuses_unknown_or_railless(name):
    with pytest.raises(ValueError):
        make_region.resolve([name])


def test_combo_id_ignores_order():
    assert make_region.combo_id(["香川県", "岡山県"]) == "custom_33_37"
    assert make_region.combo_id(["岡山県", "香川県"]) == "custom_33_37"


def test_grid_covers_every_prefecture():
    prefs = ["岡山県", "香川県"]
    meta = make_region.region_json("custom_33_37", prefs)
    g = meta["grid"]
    import math
    lon1 = g["lon0"] + g["span_x"] / (111_320.0 * math.cos(math.radians(g["lat_ref"])))
    lat1 = g["lat0"] + g["span_y"] / 111_132.0
    for p in prefs:
        b = make_region.PREFS[p]["rail_bbox"]
        assert g["lon0"] < b[0] and b[2] < lon1 and g["lat0"] < b[1] and b[3] < lat1, p
    assert g["ocean_seeds"] == "auto"
    assert meta["custom"] is True
    assert meta["prefectures"] == prefs
    # 岡山 경계의 일부(兵庫 와의 바다 경계)는 간사이 추출본에만 있다. 이웃 현의 추출본도 읽는다.
    assert meta["osm_extracts"] == ["chugoku", "kansai", "shikoku"]
    # 첫 출발지는 역이 더 많은 현의 중심역. 이름은 다섯 언어로 들어간다.
    assert meta["start"]["names"]["ja"] == "岡山"
    assert meta["start"]["names"]["ko"] == "오카야마"


def test_extra_areas_follow_the_prefecture():
    """北도호쿠가 적어 둔 気仙沼(大船渡線 종점)는 이와테를 고르면 따라온다."""
    assert make_region.region_json("x", ["岩手県"]).get("extra_areas")
    assert "extra_areas" not in make_region.region_json("x", ["岡山県"])


def test_far_apart_prefectures_are_refused():
    plan = make_region.plan(["青森県", "鹿児島県"])
    assert plan["too_big"]
    assert not make_region.plan(["岡山県", "香川県"])["too_big"]


def test_books_merge_only_for_combos(monkeypatch):
    metas = {
        "chugoku": {"prefectures": ["岡山県", "広島県"]},
        "shikoku": {"prefectures": ["香川県", "愛媛県"]},
        "kansai": {"prefectures": ["兵庫県"]},
        "custom_33_37": {"prefectures": ["岡山県", "香川県"], "custom": True},
    }
    monkeypatch.setattr(regional, "_meta", lambda rid: metas.get(rid, {}))
    book = {
        "note": "설명",
        "chugoku": {"JR津山線": {"法界院": ["岡山"]},
                    "JR予讃線": {"빼는 역": ["A"]}},
        "shikoku": {"JR予讃線": {"빼는 역": ["B"], "끼울 역": ["C"]}},
        "kansai": {"JR姫新線": {"播磨高岡": ["姫路"]}},
    }
    got = regional.merge_books(book, "custom_33_37")
    assert got["JR津山線"] == {"法界院": ["岡山"]}
    assert got["JR予讃線"] == {"빼는 역": ["A", "B"], "끼울 역": ["C"]}
    assert "JR姫新線" not in got                 # 현이 안 겹치는 권역 것은 안 온다
    # 원본은 그대로다
    assert book["chugoku"]["JR予讃線"] == {"빼는 역": ["A"]}
    # 기존 권역은 제 것만 쓴다. 제 항목이 없어도 이웃 것을 빌리지 않는다.
    assert regional.merge_books(book, "kansai") == {"JR姫新線": {"播磨高岡": ["姫路"]}}
    assert regional.merge_books({"note": "", "chugoku": {}}, "shikoku") == {}


def test_areas_cover_every_prefecture_once():
    """화면의 지방 묶음. 三重 는 東海 에 둔다(関西 는 2부 4현)."""
    seen = [p for _names, members in make_region.AREAS for p in members]
    assert sorted(seen) == sorted(make_region.PREFS)
    assert len(seen) == len(set(seen))
    area_of = {p: names["ja"] for names, members in make_region.AREAS for p in members}
    assert area_of["三重県"] == "東海"
    assert area_of["大阪府"] == "関西"
