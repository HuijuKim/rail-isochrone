"""도달권역 계산 API 와 웹 UI 를 제공하는 로컬 서버."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import region as region_mod
from isochrone import (
    EGRESS_WALK_MAX_SEC,
    build_field,
    build_field_network,
    build_walk_only_field,
    contour_geojson,
)
from router import (
    ACCESS_GATE_SEC,
    ACCESS_UNLIMITED_SEC,
    INF,
    WALK_SPEED,
    access_seconds,
    earliest_arrivals,
    haversine_m,
    reconstruct,
    walk_seconds,
)

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

# 경로 조회에서 열어두는 탐색 지평. 상한과 무관하게 "갈 수 있나" 를 답한다.
ROUTE_HORIZON_SEC = 300 * 60

app = Flask(__name__, static_folder=None)

REGIONS = {rid: region_mod.load(rid) for rid in region_mod.available()}
if not REGIONS:
    raise SystemExit("권역 데이터가 없습니다. data/regions/<id>/ 를 확인하세요.")
DEFAULT_REGION = "kanto" if "kanto" in REGIONS else next(iter(REGIONS))


def pick_region():
    """요청이 가리키는 권역. 알 수 없으면 기본 권역."""
    return REGIONS.get(request.args.get("region", DEFAULT_REGION), REGIONS[DEFAULT_REGION])


def parse_hhmm(text: str) -> int:
    hh, mm = text.split(":")
    return int(hh) * 3600 + int(mm) * 60


def google_maps_key() -> str | None:
    """환경변수나 config.json 에서 구글 지도 JS API 키를 찾는다.

    키가 없으면 None 이고, 이때 화면은 OpenStreetMap 타일로 돌아간다.
    """
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if key:
        return key
    cfg = ROOT / "config.json"
    if cfg.exists():
        try:
            value = json.loads(cfg.read_text(encoding="utf-8")).get("google_maps_api_key", "")
            return value.strip() or None
        except (json.JSONDecodeError, OSError):
            return None
    return None


def direct_walk_seconds(reg, lon: float, lat: float, dest_lon: float, dest_lat: float,
                        limit: float) -> float:
    """출발지에서 도착지까지 전철 없이 걸었을 때의 시간."""
    WALK = reg.walk
    if WALK is None:
        return float(walk_seconds(haversine_m(lon, lat, dest_lon, dest_lat)))

    # 보행 거리는 대권거리보다 짧을 수 없다. 이 하한만으로 상한을 넘으면
    # 탐색할 필요가 없다. 상한 180분이면 반경 14 km 를 훑게 되는 자리라
    # 이 한 줄이 대부분의 요청을 걸러낸다.
    if float(haversine_m(lon, lat, dest_lon, dest_lat)) / WALK_SPEED > limit:
        return float("inf")

    return WALK.time_to_node(lon, lat, WALK.nearest_node(dest_lon, dest_lat), limit)


def hhmm(seconds: int) -> str:
    """자정 기준 초를 시각 표기로. 자정을 넘긴 운행은 25:13 처럼 적는다."""
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


def station_brief(reg, i: int) -> dict:
    return {"ja": reg.stops["ja"][i], "ko": reg.stops["ko"][i]}


def leg_path(reg, indices: list[int]) -> list[list[float]]:
    """정차역을 실제 선로를 따르는 선으로."""
    usable = [i for i in indices if np.isfinite(reg.coords[i, 0])]
    return reg.geometry.ride_path(usable) if len(usable) >= 2 else []


def walk_path(reg, lon: float, lat: float, station: int, limit: float) -> list[list[float]]:
    """지점과 역 사이 실제로 걷는 길. 보행망이 없으면 직선 두 점."""
    ends = [
        [round(float(lon), 6), round(float(lat), 6)],
        [round(float(reg.coords[station, 0]), 6), round(float(reg.coords[station, 1]), 6)],
    ]
    WALK = reg.walk
    if WALK is None:
        return ends
    path = WALK.path_to_node(lon, lat, int(WALK.station_node[station]), limit)
    if len(path) < 2:
        return ends
    # 다익스트라는 스냅된 노드에서 시작하고 끝나므로 양 끝에 실제 지점을 잇는다
    return [ends[0]] + path + [ends[1]]


def describe_journey(reg, legs: list[dict], best: np.ndarray, depart: int,
                     origin: tuple[float, float]) -> list[dict]:
    """복원한 구간들을 화면에 그대로 쓸 수 있는 형태로 옮긴다."""
    out = []
    for leg in legs:
        if leg["type"] == "access":
            secs = max(0, int(best[leg["to"]]) - depart)
            out.append(
                {
                    "type": "walk",
                    "to": station_brief(reg, leg["to"]),
                    "min": round(secs / 60),
                    "path": walk_path(reg, origin[0], origin[1], leg["to"], secs + 120),
                }
            )
        elif leg["type"] == "transfer":
            out.append(
                {
                    "type": "transfer",
                    "at": station_brief(reg, leg["to"]),
                    "min": max(0, round((int(best[leg["to"]]) - int(best[leg["from"]])) / 60)),
                }
            )
        else:
            rail = reg.railways.get(reg.stops["railway"][leg["from"]], {})
            title = rail.get("title", {})
            out.append(
                {
                    "type": "ride",
                    "from": station_brief(reg, leg["from"]),
                    "to": station_brief(reg, leg["to"]),
                    "depart": hhmm(leg["depart"]),
                    "arrive": hhmm(leg["arrive"]),
                    "min": round((leg["arrive"] - leg["depart"]) / 60),
                    "stops": len(leg["path"]) - 1,
                    "railway": {
                        "ja": title.get("ja", ""),
                        "ko": title.get("ko", "") or title.get("ja", ""),
                        "color": rail.get("color", "#888888"),
                    },
                    "path": leg_path(reg, leg["path"]),
                }
            )
    return out


def read_query() -> dict:
    """출발지·시각·다이어 등 두 엔드포인트가 함께 쓰는 파라미터.

    도보 시간은 두 겹이다. egress 는 "역에서 내려 얼마나 걸을 것인가" 이고,
    walk_total 은 "여정 전체에서 걷는 시간의 상한" 이다. 출발지에서 역까지
    걷는 시간도 이 전체 상한을 따른다.

    walk_total 을 안 주면 소요 시간 상한이 곧 한계다. 상한을 아예 풀려면
    walk_unlimited=1 을 명시해야 한다. 무제한이 기본이면 매 요청이 몇 시간치
    도보를 탐색하게 되어 눈에 띄게 느려진다.
    """
    thresholds = sorted(
        int(x) * 60 for x in request.args.get("thresholds", "30,45,60").split(",")
    )
    raw_total = request.args.get("walk_total", "").strip()
    unlimited = request.args.get("walk_unlimited") in ("1", "true", "yes")
    return {
        "lon": float(request.args["lon"]),
        "lat": float(request.args["lat"]),
        "depart": parse_hhmm(request.args.get("depart", "09:00")),
        "calendar": request.args.get("calendar", "Weekday"),
        "thresholds": thresholds,
        "egress": int(request.args.get("egress", EGRESS_WALK_MAX_SEC // 60)) * 60,
        "walk_total": ACCESS_UNLIMITED_SEC if unlimited else
                      (int(raw_total) * 60 if raw_total else None),
        "walk_unlimited": unlimited,
    }


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/api/regions")
def regions():
    """고를 수 있는 권역 목록."""
    return jsonify(
        {
            "default": DEFAULT_REGION,
            "regions": [
                {
                    "id": r.id,
                    "name": r.name,
                    "name_full": r.meta.get("name_full", r.name),
                    "center": r.meta.get("center"),
                    "zoom": r.meta.get("zoom", 11),
                    "stations": r.n_groups,
                    "note": r.meta.get("note", ""),
                }
                for r in REGIONS.values()
            ],
        }
    )


@app.get("/api/stations")
def stations():
    """검색창 자동완성용 역 목록."""
    return jsonify(pick_region().search_index)


@app.get("/api/config")
def config():
    """화면이 서버와 같은 값을 쓰도록 넘겨준다.

    도보 속도나 도보권 상한을 화면에 따로 적어두면, 서버 상수를 고쳤을 때
    안내 문구만 옛날 값으로 남는다.
    """
    reg = pick_region()
    # 하차 후 도보 상한은 미리 계산해 둔 도보권이 커버하는 만큼까지만 의미가
    # 있다. 그 값을 데이터에서 직접 읽는다.
    max_egress = EGRESS_WALK_MAX_SEC // 60
    if reg.walk is not None and len(reg.walk.shed_sec):
        max_egress = int(float(reg.walk.shed_sec.max()) // 60)
    return jsonify(
        {
            "google_maps_key": google_maps_key(),
            "walk_speed_m_per_min": round(WALK_SPEED * 60),
            "egress_default_min": EGRESS_WALK_MAX_SEC // 60,
            "egress_max_min": max_egress,
            # 보행망이 없으면 직선거리 근사로 떨어진다. 화면 설명이 실제
            # 계산과 달라지지 않도록 어느 쪽인지 알려준다.
            "walk_network": reg.walk is not None,
        }
    )


@app.get("/api/coverage")
def coverage():
    """앱이 동작하는 범위의 윤곽선."""
    return jsonify(pick_region().coverage_geojson)


@app.get("/api/nearest")
def nearest():
    """이 지점에서 걸어서 가장 가까운 역. 핀을 역에 맞출 때 쓴다."""
    reg = pick_region()
    lon = float(request.args["lon"])
    lat = float(request.args["lat"])

    # 라우터가 출발지를 잡을 때와 같은 계산을 쓴다. 따로 구현하면 화면이
    # "가까운 역 없음" 이라는데 권역은 그려지는 식으로 어긋난다.
    g = next(iter(reg.graphs.values()))
    secs = access_seconds(g, lon, lat, ACCESS_UNLIMITED_SEC, reg.walk)
    secs = np.where(np.isfinite(reg.coords[:, 0]), secs, np.inf)
    if not np.isfinite(secs).any():
        return jsonify({"ok": False})

    i = int(np.argmin(secs))
    return jsonify(
        {
            "ok": True,
            "ja": reg.stops["ja"][i],
            "ko": reg.stops["ko"][i],
            "lon": round(float(reg.coords[i, 0]), 6),
            "lat": round(float(reg.coords[i, 1]), 6),
            "walk_min": round(float(secs[i]) / 60),
        }
    )


@app.get("/api/reachable")
def reachable():
    """이 지점을 출발지·도착지로 쓸 수 있는지.

    찍어두고 빈 결과를 보는 것보다 찍히지 않는 편이 낫다. 다만 이유는
    갈라서 알려준다. 권역 밖인지, 권역 안이지만 바다나 길 없는 곳인지,
    길은 있는데 걸어서 닿는 역이 없는지는 서로 다른 이야기다.
    """
    reg = pick_region()
    lon = float(request.args["lon"])
    lat = float(request.args["lat"])
    cover = reg.coverage
    # 화면이 쓰고 있는 도보 상한을 그대로 적용한다. 서버가 제 기준으로
    # 판정하면 "60분 안에 역이 없다" 는데 화면은 25분으로 계산하는 식이 된다.
    raw_gate = request.args.get("walk_total", "").strip()
    if request.args.get("walk_unlimited") in ("1", "true", "yes"):
        gate, raw_gate = ACCESS_UNLIMITED_SEC, ""
    else:
        gate = int(raw_gate) * 60 if raw_gate else ACCESS_GATE_SEC

    # 지도에 그린 선과 같은 기준으로 판정한다. 선 바깥인데 클릭은 되는
    # 식이면 선이 무슨 의미인지 알 수 없다.
    if cover is not None and not cover.contains(lon, lat):
        if cover.is_sea(lon, lat):
            return jsonify({"ok": False, "reason": "sea"})
        return jsonify({"ok": False, "reason": "outside_region"})

    if reg.walk is not None and reg.walk.nearest_node(lon, lat) < 0:
        if cover is not None and cover.is_sea(lon, lat):
            return jsonify({"ok": False, "reason": "sea"})
        return jsonify({"ok": False, "reason": "no_road"})

    # 길은 있어도 걸어서 닿는 역이 없으면 출발지로 쓸 수 없다
    g = next(iter(reg.graphs.values()))
    secs = access_seconds(g, lon, lat, gate, reg.walk)
    if not bool((secs <= gate).any()):
        return jsonify(
            {
                "ok": False,
                "reason": "no_station_in_range",
                "gate_min": None if not raw_gate else gate // 60,
            }
        )

    return jsonify({"ok": True})


@app.get("/api/isochrone")
def isochrone():
    qs = read_query()
    lon, lat, depart = qs["lon"], qs["lat"], qs["depart"]
    thresholds, egress = qs["thresholds"], qs["egress"]

    reg = pick_region()
    g = reg.graphs.get(qs["calendar"]) or next(iter(reg.graphs.values()))
    budget = max(thresholds)

    # 도보 전용 모드: 전철을 아예 빼고 걷기만 한다
    if request.args.get("mode") == "walk" and reg.walk is not None:
        field = build_walk_only_field(reg.walk, lon, lat, budget)
        return jsonify(
            {
                "geojson": contour_geojson(field, thresholds),
                "stats": {"reached": 0, "total": reg.n_groups, "farthest": [], "mode": "walk"},
            }
        )

    # 상한보다 오래 걷는 것은 계산할 필요가 없다. 그렇게 걸어 역에 닿아도
    # 이미 시간이 지나 권역에 들어오지 못한다. "제한 없음" 으로 6시간을
    # 열어두면 65만 노드를 2.5초 훑고도 결과는 같다.
    access_limit = min(qs["walk_total"] or budget, budget)
    best = earliest_arrivals(
        g, lon, lat, depart, budget, walk=reg.walk, access_limit=access_limit
    )

    # 출발지에서 그냥 걸어가는 범위. 역에 닿을 수 있으면 전철이 훨씬 멀리
    # 가므로 도보 원은 어차피 그 안에 묻힌다. 상한만큼 크게 그리려면 반경이
    # 십수 km 라 탐색만 1초 가까이 쓰고 화면에는 보이지도 않는다. 역이 아예
    # 없을 때만 상한만큼 펼친다.
    has_station = bool((best < INF).any())
    origin_walk = min(budget, ACCESS_GATE_SEC) if has_station else budget

    if reg.walk is not None:
        field = build_field_network(
            reg.walk, lon, lat, depart, best, budget,
            egress_max_sec=egress, origin_walk_max_sec=origin_walk,
        )
    else:
        field = build_field(g, lon, lat, depart, best, budget, egress_max_sec=egress)
    geojson = contour_geojson(field, thresholds)

    elapsed = best.astype(np.float64) - depart
    # 역 수는 사람이 아는 "역" 단위로 센다. 역 목록은 노선별로 쪼개져 있어서
    # 그대로 세면 신주쿠 하나가 11개로 잡힌다.
    within = (best < INF) & (elapsed <= budget) & (reg.station_group >= 0)
    reached = int(np.unique(reg.station_group[within]).size)

    # 가장 멀리까지 간 역. 소요 시간이 아니라 출발지에서의 직선거리 순으로
    # 세운다. 같은 시간이라도 얼마나 멀리 나갈 수 있었는지가 궁금한 자리다.
    # 도달 불가 역이 앞으로 오지 않도록 +inf 로 밀어내고, 같은 역의 노선별
    # 중복 항목은 한 번만 센다.
    away = haversine_m(lon, lat, reg.coords[:, 0], reg.coords[:, 1])
    usable = (best < INF) & (elapsed <= budget) & np.isfinite(reg.coords[:, 0])
    rank_key = np.where(usable, -away, np.inf)

    far = []
    seen: set[str] = set()
    for i in np.argsort(rank_key):
        if not np.isfinite(rank_key[i]):
            continue
        name = reg.stops["ja"][i]
        if name in seen:
            continue
        seen.add(name)
        far.append(
            {
                "ja": name,
                "ko": reg.stops["ko"][i],
                "min": round(elapsed[i] / 60),
                "km": round(float(away[i]) / 1000, 1),
                "lon": round(float(reg.coords[i, 0]), 6),
                "lat": round(float(reg.coords[i, 1]), 6),
            }
        )
        if len(far) >= 8:
            break

    return jsonify(
        {
            "geojson": geojson,
            "stats": {
                "reached": reached,
                "total": reg.n_groups,
                "farthest": far,
            },
        }
    )


@app.get("/api/point")
def point():
    """임의의 도착 지점까지 실제로 몇 분 걸리는지, 어디서 내려 얼마나 걷는지."""
    qs = read_query()
    lon, lat, depart = qs["lon"], qs["lat"], qs["depart"]
    egress = qs["egress"]
    budget = max(qs["thresholds"])
    dest_lon = float(request.args["dest_lon"])
    dest_lat = float(request.args["dest_lat"])

    reg = pick_region()
    g = reg.graphs.get(qs["calendar"]) or next(iter(reg.graphs.values()))
    # 상한을 넘더라도 "몇 분 걸리는지" 는 알려주는 편이 쓸모 있다. 상한에
    # 맞춰 좁히면 하코네 고우라(신주쿠에서 168분)처럼 멀쩡히 갈 수 있는 곳이
    # "닿지 않음" 으로 나온다. 넓혀도 비용은 0.25초에서 0.28초 정도다.
    horizon = ROUTE_HORIZON_SEC

    # 도보 상한은 권역을 그릴 때 쓰는 값(하차 후 도보, 전체 도보)을 그대로
    # 쓰면 안 된다. 그건 "권역을 어디까지 칠할까" 의 기준이고, 여기서 묻는
    # 것은 "갈 수 있나" 다. 한 시간 걸어서 닿으면 상한을 넘더라도 보여준다.
    walk_limit = max(egress, qs["walk_total"] or 0, ACCESS_GATE_SEC)
    access_limit = max(qs["walk_total"] or 0, ACCESS_GATE_SEC)
    if qs["walk_unlimited"]:
        access_limit = ACCESS_GATE_SEC   # 먼저 좁게 풀고, 안 되면 아래에서 넓힌다
    best, trace = earliest_arrivals(
        g, lon, lat, depart, horizon, walk=reg.walk, trace=True, access_limit=access_limit
    )
    # 주변에 역이 없을 때만 도보 상한을 넓혀 다시 푼다. 역이 지천인 도심에서
    # 매번 몇 시간치 도보를 훑을 이유가 없다.
    if qs["walk_unlimited"] and not bool((best < INF).any()):
        best, trace = earliest_arrivals(
            g, lon, lat, depart, horizon, walk=reg.walk, trace=True, access_limit=horizon
        )

    elapsed = best.astype(np.float64) - depart
    if reg.walk is not None:
        # 도착지에서 역 쪽으로 걸어가는 시간 (보행로는 양방향이라 그대로 쓴다)
        walk = reg.walk.station_times_from_point(
            dest_lon, dest_lat, walk_limit, len(reg.coords)
        )
    else:
        walk = walk_seconds(haversine_m(dest_lon, dest_lat, reg.coords[:, 0], reg.coords[:, 1]))
    usable = (best < INF) & np.isfinite(walk) & (walk <= walk_limit)
    total = np.where(usable, elapsed + walk, np.inf)

    # 대중교통을 아예 타지 않고 걸어가는 쪽이 빠를 수도 있다. 여기에도 같은
    # 상한을 쓴다. 무제한으로 두면 바다 한복판도 "걸어서 4천 분" 이 된다.
    direct = direct_walk_seconds(reg, lon, lat, dest_lon, dest_lat, walk_limit)
    walkable = np.isfinite(direct) and direct <= walk_limit

    via = None
    journey: list[dict] = []
    best_total = direct if walkable else np.inf
    if np.isfinite(total).any():
        i = int(np.nanargmin(total))
        if total[i] < best_total:
            best_total = float(total[i])
            via = {
                "ja": reg.stops["ja"][i],
                "ko": reg.stops["ko"][i],
                "ride_min": round(elapsed[i] / 60),
                "walk_min": round(walk[i] / 60),
            }
            journey = describe_journey(reg, reconstruct(g, trace, i), best, depart, (lon, lat))
            leg = walk_path(reg, dest_lon, dest_lat, i, float(walk[i]) + 120)
            leg.reverse()   # 역 -> 도착지 방향으로 그린다
            journey.append(
                {
                    "type": "walk",
                    "to": None,
                    "min": round(walk[i] / 60),
                    "final": True,
                    "path": leg,
                }
            )

    if not np.isfinite(best_total):
        return jsonify({"reachable": False})
    best_total = float(best_total)

    return jsonify(
        {
            "reachable": True,
            "total_min": round(best_total / 60),
            "within_budget": best_total <= budget,
            "budget_min": round(budget / 60),
            "via": via,
            "walk_only": via is None,
            "direct_walk_min": round(direct / 60) if walkable else None,
            "journey": journey,
            "transfers": sum(1 for leg in journey if leg["type"] == "transfer"),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5173, debug=False)
