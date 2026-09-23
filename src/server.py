"""도달권역 계산 API 와 웹 UI 를 제공하는 로컬 서버."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import make_region
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
# 소요 시간 상한. 이보다 크면 격자가 수천만 칸이 되어 서버가 멎는다.
MAX_BUDGET_SEC = 300 * 60
# 한 번에 그릴 수 있는 등시간선 개수. 화면은 세 개를 쓴다. 값마다 등고선을
# 따로 뽑으므로, thresholds=1,2,3,... 으로 수만 개를 보내면 그만큼 일한다.
MAX_THRESHOLDS = 12

app = Flask(__name__, static_folder=None)

# 한 권역이 안 올라와도 나머지는 띄운다. 빌드 도중에는 역 목록만 새로
# 쓰이고 도보권은 아직 옛것이라 그 권역만 짝이 안 맞는데, 예전에는 그
# 때문에 서버 전체가 안 떴다. 못 올린 권역은 목록에서 빠진다.
REGIONS = {}
for _rid in region_mod.available():
    try:
        REGIONS[_rid] = region_mod.load(_rid)
    except Exception as _err:      # noqa: BLE001 - 어떤 이유든 그 권역만 뺀다
        print(f"!! 권역 {_rid} 를 못 올렸습니다: {_err}", flush=True)
if not REGIONS:
    raise SystemExit("권역 데이터가 없습니다. data/regions/<id>/ 를 확인하세요.")
DEFAULT_REGION = "kanto" if "kanto" in REGIONS else next(iter(REGIONS))


def pick_region():
    """요청이 가리키는 권역.

    모르는 이름이면 기본 권역으로 때우지 않고 막는다. 오타 하나로 엉뚱한
    권역의 답을 200 으로 받으면 틀린 줄도 모른다.
    """
    name = request.args.get("region")
    if name is None:
        return REGIONS[DEFAULT_REGION]
    if name not in REGIONS:
        raise BadRequest(f"모르는 권역입니다: {name!r} (쓸 수 있는 것: "
                         + ", ".join(sorted(REGIONS)) + ")")
    return REGIONS[name]


class BadRequest(Exception):
    """요청이 잘못됐을 때. 화면이 읽을 수 있는 JSON 으로 돌려준다."""


@app.errorhandler(BadRequest)
def bad_request(err):
    return jsonify({"error": str(err)}), 400


@app.errorhandler(Exception)
def unexpected(err):
    """예상 못 한 예외도 JSON 으로. 화면은 응답을 JSON 으로 읽는다.

    HTML 오류 쪽을 돌려주면 화면에서 "Unexpected token '<'" 같은 소리가
    나와서 무슨 일인지 알 수 없다. 원인은 서버 로그에 그대로 남긴다.

    404 나 405 처럼 HTTP 가 이미 뜻을 정해 둔 것은 그 코드 그대로
    내보낸다. 전부 500 으로 뭉뚱그리면 주소를 잘못 친 것과 서버가
    터진 것을 구별할 수 없다.
    """
    from werkzeug.exceptions import HTTPException

    if isinstance(err, HTTPException):
        return jsonify({"error": f"{err.code} {err.name}"}), err.code

    import traceback

    traceback.print_exc()
    return jsonify({"error": f"{type(err).__name__}: {err}"}), 500


def need_float(name: str) -> float:
    raw = request.args.get(name)
    if raw is None:
        raise BadRequest(f"{name} 파라미터가 없습니다")
    try:
        value = float(raw)
    except ValueError:
        raise BadRequest(f"{name} 값이 숫자가 아닙니다: {raw!r}") from None
    if not np.isfinite(value):
        raise BadRequest(f"{name} 값이 유효하지 않습니다: {raw!r}")
    return value


def need_point(prefix: str = "") -> tuple[float, float]:
    """경위도 한 쌍. 범위를 벗어나면 거기서 막는다."""
    lon = need_float(prefix + "lon")
    lat = need_float(prefix + "lat")
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise BadRequest(f"좌표가 범위를 벗어났습니다: {lon}, {lat}")
    return lon, lat


def parse_hhmm(text: str) -> int:
    try:
        hh, mm = text.split(":")
        value = int(hh) * 3600 + int(mm) * 60
    except ValueError:
        raise BadRequest(f"시각 형식이 HH:MM 이 아닙니다: {text!r}") from None
    if not (0 <= value < 30 * 3600):
        raise BadRequest(f"시각이 범위를 벗어났습니다: {text!r}")
    return value


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


def draw_on_roads(reg, path: list[list[float]]) -> list[list[float]]:
    """그린 선을 원본 도로 위로 옮긴다. 시간과 경로 선택은 그대로 둔다.

    40 m 격자에서 나온 선은 도로에서 최대 40 m 벗어난다. 지도에 겹쳐 놓으면
    건물 위를 지나가는 것처럼 보인다.
    """
    return reg.fine.trace(path) if getattr(reg, "fine", None) is not None else path


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


def direct_walk_path(reg, lon: float, lat: float, dest_lon: float, dest_lat: float,
                     limit: float) -> list[list[float]]:
    """전철을 타지 않고 걸어갈 때 실제로 걷는 길.

    walk_path 는 도착점이 역일 때만 쓸 수 있어서 따로 둔다. 걷는 편이
    빠르다고 답해 놓고 지도에 선이 없으면 어디로 걸으라는 건지 알 수 없다.
    """
    ends = [
        [round(float(lon), 6), round(float(lat), 6)],
        [round(float(dest_lon), 6), round(float(dest_lat), 6)],
    ]
    WALK = reg.walk
    if WALK is None:
        return ends
    path = WALK.path_to_node(lon, lat, WALK.nearest_node(dest_lon, dest_lat), limit)
    if len(path) < 2:
        return ends
    # 다익스트라는 스냅된 노드에서 시작하고 끝나므로 양 끝에 실제 지점을 잇는다
    return draw_on_roads(reg, [ends[0]] + path + [ends[1]])


def hhmm(seconds: int) -> str:
    """자정 기준 초를 시각 표기로. 자정을 넘긴 운행은 25:13 처럼 적는다."""
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


def station_brief(reg, i: int) -> dict:
    """역 이름을 화면이 고를 수 있는 모든 언어로. 어느 말로 보여줄지는 화면이 정한다."""
    return {lang: reg.stops[lang][i] for lang in region_mod.LANGS if lang in reg.stops}


def leg_path(reg, indices: list[int], rid: str | None = None) -> list[list[float]]:
    """정차역을 실제 선로를 따르는 선으로.

    노선을 함께 넘긴다. 그래야 build_track.py 가 남긴 그 노선의
    구간 선형을 쓴다. 안 넘기던 때는 지도에 그린 노선은 선로를
    타는데 경로선만 모서리를 질러, 같은 구간에 선이 둘 보였다.
    """
    usable = [i for i in indices if np.isfinite(reg.coords[i, 0])]
    return reg.geometry.ride_path(usable, rid) if len(usable) >= 2 else []


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
    return draw_on_roads(reg, [ends[0]] + path + [ends[1]])


# 두 역이 이보다 멀면 플랫폼 이동이 아니라 밖으로 걸어 나가는 환승으로 본다.
TRANSFER_WALK_M = 200.0


def _pattern_of(reg, leg) -> dict:
    """탄 운행의 종별과 열차 이름. 각역정차면 빈 값."""
    i = int(leg.get("pat", -1))
    if i < 0 or i >= len(reg.patterns):
        return {}
    pat = reg.patterns[i]
    out = {"kind": pat.get("kind") or ""}
    if pat.get("name"):
        out["train"] = pat["name"]
    return out


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
            # 같은 역 구내에서 플랫폼만 옮기는 환승과, 역과 역 사이를 걸어서
            # 갈아타는 환승은 다르다. 片瀬江ノ島 에서 目白山下 까지는 1.1km 를
            # 걷는데 "환승" 이라고만 적으면 걷는 줄 모른다. 걷는 쪽은 길을
            # 그려 주고 도보로 표시한다.
            a, b = leg["from"], leg["to"]
            secs = max(0, int(best[b]) - int(best[a]))
            item = {
                "type": "transfer",
                "at": station_brief(reg, b),
                "min": round(secs / 60),
            }
            far = haversine_m(float(reg.coords[a, 0]), float(reg.coords[a, 1]),
                              float(reg.coords[b, 0]), float(reg.coords[b, 1]))
            if far > TRANSFER_WALK_M:
                item["walk"] = True
                item["from"] = station_brief(reg, a)
                item["to"] = station_brief(reg, b)
                # 환승 값은 직선거리에 1.25배를 매긴 것이라 실제 길보다
                # 짧을 수 있다. 目白山下 는 언덕을 돌아 올라가서 한도를
                # 빠듯하게 주면 길을 못 찾고 직선만 남는다.
                item["path"] = walk_path(reg, float(reg.coords[a, 0]),
                                         float(reg.coords[a, 1]), b,
                                         max(secs * 2, secs + 300))
            out.append(item)
        else:
            rid = reg.stops["railway"][leg["from"]]
            rail = reg.railways.get(rid, {})
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
                    "railway": dict(
                        {lang: title.get(lang, "") or title.get("ja", "")
                         for lang in region_mod.LANGS},
                        # 지도에 그린 선과 같은 값을 쓴다
                        color=region_mod.line_color(rail),
                        # 각역정차가 아니면 종별과 열차 이름을 함께 준다.
                        # 여러 노선을 이어 달리는 특급은 이것으로만 가려진다.
                        **_pattern_of(reg, leg),
                    ),
                    "path": leg_path(reg, leg["path"], rid),
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
    raw_thresholds = request.args.get("thresholds", "30,45,60")
    try:
        thresholds = sorted({int(x) * 60 for x in raw_thresholds.split(",") if x.strip()})
    except ValueError:
        raise BadRequest(f"thresholds 가 숫자 목록이 아닙니다: {raw_thresholds!r}") from None
    if len(thresholds) > MAX_THRESHOLDS:
        raise BadRequest(f"thresholds 는 {MAX_THRESHOLDS}개까지입니다")
    if not thresholds or thresholds[0] <= 0:
        raise BadRequest(f"thresholds 는 1분 이상이어야 합니다: {raw_thresholds!r}")
    if thresholds[-1] > MAX_BUDGET_SEC:
        raise BadRequest(f"소요 시간 상한은 {MAX_BUDGET_SEC // 60}분을 넘을 수 없습니다")
    budget = max(thresholds)
    raw_total = request.args.get("walk_total", "").strip()
    unlimited = request.args.get("walk_unlimited") in ("1", "true", "yes")

    # 세 값은 포개진 제약이다. 바깥부터 소요 시간 상한, 하차 후 도보, 전체
    # 도보 순이고 안쪽이 바깥을 넘을 수 없다. 화면에서도 같은 순서로 묶지만,
    # 주소창으로 직접 부르면 그 제약을 건너뛰므로 여기서도 맞춘다.
    try:
        egress = min(int(request.args.get("egress", EGRESS_WALK_MAX_SEC // 60)) * 60, budget)
        walk_total = int(raw_total) * 60 if raw_total else None
    except ValueError:
        raise BadRequest("egress 와 walk_total 은 분 단위 정수여야 합니다") from None
    if egress < 0 or (walk_total is not None and walk_total < 0):
        raise BadRequest("도보 시간은 0분 이상이어야 합니다")
    if walk_total is not None:
        walk_total = min(max(walk_total, egress), budget)
    # "제한 없음" 만 예외다. 가장 가까운 역이 몇 시간 거리인 곳에서 쓰라고
    # 둔 것이라, 소요 시간 상한을 넘겨서도 걷는 것이 이 선택의 목적이다.
    if unlimited:
        walk_total = ACCESS_UNLIMITED_SEC

    lon, lat = need_point()
    return {
        "lon": lon,
        "lat": lat,
        "depart": parse_hhmm(request.args.get("depart", "09:00")),
        "calendar": request.args.get("calendar", "Weekday"),
        "thresholds": thresholds,
        "egress": egress,
        "walk_total": walk_total,
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
                    # 이름은 언어별로 보낸다. 어느 말로 보여줄지는 화면이 정한다.
                    "names": r.names,
                    "center": r.meta.get("center"),
                    "zoom": r.meta.get("zoom", 11),
                    # 권역마다 첫 출발지가 다르다. 간사이에서 신주쿠를
                    # 띄울 수는 없다.
                    "start": r.meta.get("start"),
                    "stations": int(r.supported.sum()),
                    "note": r.meta.get("note", ""),
                    # 현을 골라 만든 권역은 화면의 "현 조합" 탭에 따로 모인다.
                    "custom": bool(r.meta.get("custom")),
                    "prefectures": r.meta.get("prefectures") or [],
                }
                # 역이 많은 권역부터 보인다.
                for r in sorted(REGIONS.values(), key=lambda r: -int(r.supported.sum()))
            ],
        }
    )


# ---------------------------------------------------------------------------
# 현 조합 권역. 화면의 "현 조합" 탭에서 현을 골라 권역 하나로 빌드한다.
# 빌드는 make_region.py 를 따로 된 프로세스로 돌리고(10-40분), 끝나면 이
# 서버에 올린다. 다른 권역은 그동안에도 그대로 쓸 수 있다. 한 번에 하나만
# 돈다. 빌드와 올리기는 서버를 띄운 컴퓨터에서 온 요청만 받는다.
# ---------------------------------------------------------------------------
COMBO_JOB = {"id": None, "prefectures": [], "names": None, "k": 0,
             "n": len(make_region.STEPS), "step": "", "detail": "",
             "started": None, "ended": None, "ok": None, "error": None}
COMBO_LOCK = threading.Lock()


def _from_this_machine() -> bool:
    return request.remote_addr in ("127.0.0.1", "::1")


def _combo_running() -> bool:
    return COMBO_JOB["started"] is not None and COMBO_JOB["ended"] is None


def _combo_state(rid: str) -> str:
    """ready: 서버에 올라 있음, building: 빌드 중, built: 다 됐는데 안 올림,
    incomplete: 만들다 말았거나 실패, none: 없음."""
    if rid in REGIONS:
        return "ready"
    if _combo_running() and COMBO_JOB["id"] == rid:
        return "building"
    return make_region.state(rid)


def _combo_prefs() -> list[str]:
    raw = request.args.get("prefs") or ""
    if request.is_json:
        raw = (request.get_json(silent=True) or {}).get("prefectures") or raw
    names = raw.split(",") if isinstance(raw, str) else list(raw)
    names = [str(n) for n in names if str(n).strip()]
    if not names:
        raise BadRequest("현을 하나 이상 고르세요")
    try:
        return make_region.resolve(names)
    except ValueError as err:
        raise BadRequest(str(err)) from None


def _combo_status() -> dict:
    job = dict(COMBO_JOB)
    if job["started"]:
        job["elapsed_min"] = round(((job["ended"] or time.time()) - job["started"]) / 60, 1)
    # 단계 안에서 무엇을 하는지는 빌드 로그의 마지막 줄로 보여준다.
    if _combo_running() and job["k"] >= 1:
        log = region_mod.REGIONS_DIR / job["id"] / "build.log"
        try:
            with open(log, "rb") as f:
                f.seek(max(0, log.stat().st_size - 4096))
                tail = f.read().decode("utf-8", "replace").splitlines()
            job["detail"] = next((x.strip() for x in reversed(tail) if x.strip()), "")
        except OSError:
            pass
    return job


def _run_combo(rid: str, prefs: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "src" / "make_region.py"), *prefs, "--build", "--force"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    last = ""
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            last = line
            m = re.match(r"\[(\d+)/(\d+)\] (.+)", line)
            if m:
                COMBO_JOB.update(k=int(m[1]), n=int(m[2]), step=m[3], detail="")
            else:
                COMBO_JOB["detail"] = line
        code = proc.wait()
        if code != 0:
            raise RuntimeError(last or f"make_region.py 종료 코드 {code}")
        COMBO_JOB.update(step="서버에 올리는 중", detail="")
        REGIONS[rid] = region_mod.load(rid)
        COMBO_JOB["ok"] = True
    except Exception as err:      # noqa: BLE001 - 무엇이든 화면에 알린다
        print(f"!! 현 조합 {rid} 빌드 실패: {err}", flush=True)
        COMBO_JOB.update(ok=False, error=str(err))
    finally:
        COMBO_JOB["ended"] = time.time()


@app.get("/api/combo")
def combo():
    """현 조합 탭이 그릴 것. 지방별 현 목록, 만들어 둔 조합, 빌드 상황."""
    areas = []
    for names, members in make_region.AREAS:
        prefs = []
        for ja in members:
            info = make_region.PREFS[ja]
            prefs.append({
                "ja": ja,
                "names": {"ja": ja, "en": info["en"], "ko": info["ko"],
                          "zh-Hans": info.get("zh-Hans") or ja,
                          "zh-Hant": info.get("zh-Hant") or ja},
                "stations": info.get("stations", 0),
                "ok": "rail_bbox" in info,
                # 이어진 현만 고르게 하려고 함께 보낸다. 바다 위 경계도 이웃이라
                # 다리로 이어진 岡山-香川, 広島-愛媛 이 들어 있다.
                "neighbors": info.get("neighbors", []),
            })
        areas.append({"names": {"zh-Hans": names["ja"], "zh-Hant": names["ja"], **names},
                      "prefectures": prefs})
    made = []
    for meta_path in sorted(region_mod.REGIONS_DIR.glob(make_region.PREFIX + "*/region.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        rid = meta_path.parent.name
        made.append({"id": rid, "names": meta.get("names"),
                     "prefectures": meta.get("prefectures") or [],
                     "state": _combo_state(rid),
                     "stations": int(REGIONS[rid].supported.sum()) if rid in REGIONS else None})
    return jsonify({"can_build": _from_this_machine(), "areas": areas,
                    "regions": made, "job": _combo_status()})


@app.get("/api/combo/plan")
def combo_plan():
    """고른 현으로 만들 권역의 크기와 상태. 빌드 전에 보여준다."""
    info = make_region.plan(_combo_prefs())
    info["state"] = _combo_state(info["id"])
    return jsonify(info)


@app.get("/api/combo/status")
def combo_status():
    return jsonify(_combo_status())


@app.post("/api/combo/build")
def combo_build():
    """고른 현으로 권역을 빌드한다. 바로 돌아오고, 진행은 status 로 본다."""
    if not _from_this_machine():
        return jsonify({"error": "빌드는 서버를 띄운 컴퓨터에서만 할 수 있습니다"}), 403
    prefs = _combo_prefs()
    info = make_region.plan(prefs)
    if info["too_big"]:
        raise BadRequest(f"격자가 {info['span_km'][0]}x{info['span_km'][1]}km 라 너무 큽니다. "
                         "현을 나눠 주세요.")
    rid = info["id"]
    with COMBO_LOCK:
        if _combo_running():
            return jsonify({"error": "다른 조합을 빌드하는 중입니다", "job": _combo_status()}), 409
        if rid in REGIONS:
            return jsonify({"id": rid, "state": "ready"})
        COMBO_JOB.update(id=rid, prefectures=prefs, names=info["names"], k=0, step="준비",
                         detail="", started=time.time(), ended=None, ok=None, error=None)
        threading.Thread(target=_run_combo, args=(rid, prefs), daemon=True).start()
    return jsonify({"id": rid, "state": "building", "job": _combo_status()})


@app.post("/api/combo/delete")
def combo_delete():
    """만든 조합을 서버에서 내리고 폴더째 지운다.

    윈도우는 열어 둔 파일을 못 지운다. 권역을 REGIONS 에서 빼고 쓰레기를
    거두어야 npz 파일이 닫히므로, 몇 번 다시 해 본 뒤에도 안 되면 그대로
    알린다(대개 계산 중인 요청이 붙들고 있다).
    """
    import gc
    import shutil

    if not _from_this_machine():
        return jsonify({"error": "서버를 띄운 컴퓨터에서만 할 수 있습니다"}), 403
    rid = str((request.get_json(silent=True) or {}).get("id") or request.args.get("id") or "")
    if not re.fullmatch(make_region.PREFIX + r"[0-9_]+", rid):
        raise BadRequest(f"현 조합 권역 id 가 아닙니다: {rid!r}")
    if _combo_running() and COMBO_JOB["id"] == rid:
        raise BadRequest("지금 만드는 중인 조합입니다. 끝난 뒤에 지워 주세요.")
    base = region_mod.REGIONS_DIR / rid
    if not base.exists():
        return jsonify({"id": rid, "deleted": True})
    REGIONS.pop(rid, None)
    err = None
    for _ in range(6):
        gc.collect()
        try:
            shutil.rmtree(base)
            err = None
            break
        except OSError as e:      # 아직 열려 있는 파일이 있다
            err = e
            time.sleep(0.5)
    if err is not None:
        return jsonify({"error": f"지우지 못했습니다: {err}"}), 409
    print(f"현 조합 {rid} 를 지웠습니다", flush=True)
    return jsonify({"id": rid, "deleted": True})


@app.post("/api/combo/open")
def combo_open():
    """빌드는 끝났는데 서버에 안 올라간 조합(명령줄로 만든 것 등)을 올린다."""
    if not _from_this_machine():
        return jsonify({"error": "서버를 띄운 컴퓨터에서만 할 수 있습니다"}), 403
    rid = str((request.get_json(silent=True) or {}).get("id") or request.args.get("id") or "")
    if not re.fullmatch(make_region.PREFIX + r"[0-9_]+", rid):
        raise BadRequest(f"현 조합 권역 id 가 아닙니다: {rid!r}")
    state = _combo_state(rid)
    if state == "built":
        REGIONS[rid] = region_mod.load(rid)
        state = "ready"
    return jsonify({"id": rid, "state": state})


@app.get("/api/stations")
def stations():
    """검색창 자동완성용 역 목록."""
    return jsonify(pick_region().search_index)


@app.get("/api/prefectures")
def prefectures():
    """도도부현 이름표. 화면 언어에 맞춰 보여주려고 언어별로 들고 있다."""
    return jsonify(pick_region().pref_names)


@app.get("/api/railways")
def railways():
    """지원 노선의 선형. 지도에 겹쳐 보여주는 용도다.

    원본 선형은 7만 6천 점이라 1.6 MB 다. 지도에 그리는 데는 그만한
    해상도가 필요 없으므로 region.SHAPE_TOLERANCE_M 만큼 줄여 보낸다.
    지금 값(3 m)이면 간토가 2만 5천 점, 0.6 MB 다. 한 노선이 여러
    조각으로 끊겨 오므로 줄 수는 노선 수보다 많다.
    """
    reg = pick_region()
    return jsonify(reg.railway_shapes)


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
    lon, lat = need_point()

    # 라우터가 출발지를 잡을 때와 같은 계산을 쓴다. 따로 구현하면 화면이
    # "가까운 역 없음" 이라는데 권역은 그려지는 식으로 어긋난다.
    #
    # 다만 상한은 좁게 잡고 시작한다. 6시간을 열어두면 사람이 사는 곳에서도
    # 매번 수십만 노드를 훑어 2.3초가 걸린다. 한 시간 안에 역이 있으면 그중
    # 가장 가까운 역이 답이고, 더 멀리 볼 이유가 없다.
    g = next(iter(reg.graphs.values()))
    coords_ok = np.isfinite(reg.coords[:, 0])
    for limit in (ACCESS_GATE_SEC, ACCESS_UNLIMITED_SEC):
        secs = np.where(coords_ok, access_seconds(g, lon, lat, limit, reg.walk), np.inf)
        if np.isfinite(secs).any():
            break
    else:
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
    lon, lat = need_point()
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
                "stats": {"reached": 0, "total": int(reg.supported.sum()),
                          "farthest": [], "mode": "walk"},
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
                **station_brief(reg, i),
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
                "total": int(reg.supported.sum()),
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
    dest_lon, dest_lat = need_point("dest_")

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

    # 대중교통을 아예 타지 않고 걸어가는 쪽이 빠를 수도 있다.
    #
    # 상한은 경유 경로가 낼 수 있는 도보 합계와 맞춰야 한다. 직접 도보만
    # 좁게 자르면, 역까지 걸어갔다가 거기서 목적지까지 또 걷는 더 긴 답이
    # 이긴다. 실제로 지치부에서 도보 19분 + 도보 59분 = 78분이 나오면서
    # 직접 걷는 70분짜리가 60분 상한에 걸려 버려지는 일이 있었다.
    #
    # 무제한으로 두지는 않는다. 그러면 바다 한복판도 "걸어서 4천 분" 이 된다.
    direct_cap = access_limit + egress
    direct = direct_walk_seconds(reg, lon, lat, dest_lon, dest_lat, direct_cap)
    walkable = np.isfinite(direct) and direct <= direct_cap

    via = None
    journey: list[dict] = []
    best_total = direct if walkable else np.inf
    if np.isfinite(total).any():
        i = int(np.nanargmin(total))
        if total[i] < best_total:
            best_total = float(total[i])
            via = dict(station_brief(reg, i),
                        ride_min=round(elapsed[i] / 60),
                        walk_min=round(walk[i] / 60))
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

    # 걷는 편이 빠른 경우에도 경로는 그려 줘야 한다
    if via is None and walkable:
        journey = [
            {
                "type": "walk",
                "to": None,
                "min": round(direct / 60),
                "final": True,
                "path": direct_walk_path(reg, lon, lat, dest_lon, dest_lat, direct + 120),
            }
        ]

    if not np.isfinite(best_total):
        return jsonify({"reachable": False})
    best_total = float(best_total)

    # 이 경로가 실제로 얼마나 걷는지. 경로를 보여줄 때는 도보 상한을 넉넉히
    # 풀어 두므로(한 시간까지), 설정한 상한을 넘는 답이 나올 수 있다. 그때
    # 경로를 감추지는 않되 넘었다는 것은 말해 줘야 한다.
    walked = sum(leg["min"] for leg in journey if leg["type"] == "walk")
    walk_cap = qs["walk_total"] if qs["walk_total"] is not None else budget
    over_walk = not qs["walk_unlimited"] and walked * 60 > walk_cap

    return jsonify(
        {
            "reachable": True,
            "total_min": round(best_total / 60),
            "within_budget": best_total <= budget,
            "budget_min": round(budget / 60),
            "via": via,
            "walk_only": via is None,
            "direct_walk_min": round(direct / 60) if walkable else None,
            "walk_min": walked,
            "walk_limit_min": round(walk_cap / 60),
            "over_walk": bool(over_walk),
            "journey": journey,
            "transfers": sum(1 for leg in journey if leg["type"] == "transfer"),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5173, debug=False)
