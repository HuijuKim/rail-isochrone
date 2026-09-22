"""현을 골라 권역 하나를 만든다.

권역은 "현 목록" 으로 정의된다. 고른 현들의 역이 있는 범위로 격자를 잡고,
그 현들이 든 OSM 추출본을 적고, 바다는 해안선 방향으로 자동으로 가린다.

    python src/make_region.py 岡山 香川            # region.json 만 만든다
    python src/make_region.py 岡山 香川 --build    # 빌드까지 (10-40분)
    python src/make_region.py 오카야마 Kagawa --build

화면의 "현 조합" 탭도 이 파일로 빌드한다. 권역 id 는 현 번호로 정해진다
(岡山·香川 -> custom_33_37). 같은 조합이 두 번 생기지 않는다.

권역을 이어 붙이는 것이 아니다. 고른 현을 처음부터 한 권역으로 빌드한다.
그래서 현 경계를 넘는 노선(瀬戸大橋線 岡山-宇多津)이 끊기지 않는다.
노선 손질 사전(line-extensions 등)은 현이 겹치는 기존 권역 것을 모아 쓴다
(regional.py). 서버가 올려 둔 권역은 다시 빌드하지 못한다(윈도우가 파일을
잠근다).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ROOT / "data" / "regions"
OSM = ROOT / "data" / "osm"
PREFS = {k: v for k, v in json.loads(
    (ROOT / "data" / "prefectures.json").read_text(encoding="utf-8")).items() if k != "note"}
CODE = {ja: i + 1 for i, ja in enumerate(PREFS)}      # JIS 순서로 적혀 있다
GEOFABRIK = "https://download.geofabrik.de/asia/japan/{}-latest.osm.pbf"
PREFIX = "custom_"

# 화면에서 현을 지방별로 묶어 보여준다. 기존 권역과 같게 묶는다. 현 번호의 8지방
# 구분은 三重 를 近畿 에 넣는데, 흔히 三重 는 東海 로 치고 이 프로그램의 권역도
# 그렇다(tokai). 関西 는 2부 4현이다.
AREAS = [
    ({"ja": "北海道", "en": "Hokkaido", "ko": "홋카이도"}, ["北海道"]),
    ({"ja": "東北", "en": "Tohoku", "ko": "도호쿠"},
     ["青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県"]),
    ({"ja": "関東", "en": "Kanto", "ko": "간토"},
     ["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"]),
    ({"ja": "甲信越", "en": "Koshinetsu", "ko": "고신에쓰"}, ["新潟県", "山梨県", "長野県"]),
    ({"ja": "北陸", "en": "Hokuriku", "ko": "호쿠리쿠"}, ["富山県", "石川県", "福井県"]),
    ({"ja": "東海", "en": "Tokai", "ko": "도카이"}, ["岐阜県", "静岡県", "愛知県", "三重県"]),
    ({"ja": "関西", "en": "Kansai", "ko": "간사이"},
     ["滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"]),
    ({"ja": "中国", "en": "Chugoku", "ko": "주고쿠"},
     ["鳥取県", "島根県", "岡山県", "広島県", "山口県"]),
    ({"ja": "四国", "en": "Shikoku", "ko": "시코쿠"}, ["徳島県", "香川県", "愛媛県", "高知県"]),
    ({"ja": "九州・沖縄", "en": "Kyushu & Okinawa", "ko": "규슈·오키나와"},
     ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]),
]

MARGIN_DEG = 0.25          # 역 범위 바깥 여유. 끝 역 도보권과 해안까지 덮는다
WARN_AREA_KM2 = 150_000    # 간토(440x330km)쯤. 넘으면 보행망 빌드가 몹시 무겁다
MAX_AREA_KM2 = 250_000     # 이 이상은 받지 않는다. 메모리 16GB 에서 보행망이 안 선다

STEPS = [
    ("build_walk.py", {}),
    ("build_admin.py", {}),
    ("build_rail.py", {}),
    ("build_express.py", {"optional": True}),
    ("build_track.py", {}),
    ("build_naive.py", {}),
    ("build_admin.py", {"env": {"ADMIN_REUSE": "1"}}),
    ("build_walk.py", {"env": {"WALK_REUSE": "1"}}),
    ("build_colors.py", {"optional": True}),
]


def _short(ja: str) -> str:
    return re.sub(r"[都府県]$", "", ja)


def _forms(name: str) -> set:
    """이름을 맞춰 볼 꼴들. 東京都·東京, 오카야마현·오카야마, Okayama·okayama."""
    n = name.strip()
    return {n, n.lower(), _short(n), re.sub(r"[현부도]$", "", n)}


def resolve(names: list[str]) -> list[str]:
    """현 이름을 일본어 정식 이름으로. 일본어·한국어·영어, 都府県 을 뗀 이름도 받는다."""
    out = []
    for raw in names:
        want = _forms(raw)
        hit = [ja for ja, e in PREFS.items()
               if want & ({ja, _short(ja), e["ko"], e["en"].lower()})]
        if not hit:
            raise ValueError(f"모르는 현: {raw}")
        ja = hit[0]
        if "rail_bbox" not in PREFS[ja]:
            raise ValueError(f"{ja} 는 역 범위가 없어 고를 수 없다(홋카이도·오키나와)")
        if ja not in out:
            out.append(ja)
    return sorted(out, key=CODE.get)


def combo_id(prefs: list[str]) -> str:
    return PREFIX + "_".join(f"{CODE[p]:02d}" for p in sorted(prefs, key=CODE.get))


def _extra_areas(prefs: list[str]) -> list:
    """현이 겹치는 기존 권역이 적어 둔 예외 구역(경계 밖 종점)을 물려받는다."""
    out = []
    for meta_path in sorted(REGIONS.glob("*/region.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("custom") or not set(meta.get("prefectures") or ()) & set(prefs):
            continue
        out += [a for a in meta.get("extra_areas") or () if a not in out]
    return out


def region_json(rid: str, prefs: list[str]) -> dict:
    boxes = [PREFS[p]["rail_bbox"] for p in prefs]
    lon0 = min(b[0] for b in boxes) - MARGIN_DEG
    lat0 = min(b[1] for b in boxes) - MARGIN_DEG
    lon1 = max(b[2] for b in boxes) + MARGIN_DEG
    lat1 = max(b[3] for b in boxes) + MARGIN_DEG
    lat_ref = (lat0 + lat1) / 2
    span_x = (lon1 - lon0) * 111_320.0 * math.cos(math.radians(lat_ref))
    span_y = (lat1 - lat0) * 111_132.0
    # 현 경계 고리를 닫으려면 이웃 현이 든 추출본도 읽어야 한다. 岡山県 경계의
    # 일부(兵庫県 과의 바다 경계)는 간사이 추출본에만 있어, 주고쿠·시코쿠만 읽은
    # 岡山·香川 조합에서 岡山 경계가 안 닫혀 권역에서 통째로 빠졌다.
    extracts = sorted({PREFS[q]["extract"] for p in prefs
                       for q in [p, *PREFS[p].get("neighbors", ())]})
    hub = PREFS[max(prefs, key=lambda p: PREFS[p].get("stations", 0))]["hub"]
    ja = "・".join(_short(p) for p in prefs)
    meta = {
        "id": rid,
        "custom": True,
        "center": [hub["lon"], hub["lat"]],
        "zoom": 11,
        "names": {"ja": ja,
                  "en": " / ".join(PREFS[p]["en"] for p in prefs),
                  "ko": "·".join(PREFS[p]["ko"] for p in prefs),
                  "zh-Hans": ja, "zh-Hant": ja},
        "model": "naive",
        "timetable": None,
        "rail": {"source": "OpenStreetMap", "license": "ODbL 1.0",
                 "express_source": "ja.wikipedia.org", "express_license": "CC BY-SA 4.0"},
        "osm_extracts": extracts,
        "osm_files": [f"{e}-latest.osm.pbf" for e in extracts],
        "grid": {"lon0": round(lon0, 4), "lat0": round(lat0, 4), "lat_ref": round(lat_ref, 3),
                 "span_x": round(span_x, -3), "span_y": round(span_y, -3),
                 "ocean_seeds": "auto"},
        "note": "현을 골라 만든 권역(make_region.py). 시각표가 없어 나이브 모델로 돈다. "
                "경계는 고른 현의 경계를 해안선으로 자른 것이라, 다른 현으로 넘어가는 "
                "노선은 경계에서 끊긴다.",
        "start": {"lon": hub["lon"], "lat": hub["lat"],
                  "names": {k: hub["ja"] for k in ("ja", "en", "ko", "zh-Hans", "zh-Hant")}},
        "prefectures": prefs,
    }
    extra = _extra_areas(prefs)
    if extra:
        meta["extra_areas"] = extra
    return meta


def state(rid: str) -> str:
    """none: 없음, built: 빌드를 마침, incomplete: 만들다 말았거나 실패."""
    base = REGIONS / rid
    if not (base / "region.json").exists():
        return "none"
    try:
        stops = json.loads((base / "stops.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "incomplete"
    done = (stops.get("ids") != ["STUB"] and (base / "walk" / "sheds.npz").exists()
            and (base / "walk" / "land.npz").exists())
    return "built" if done else "incomplete"


def plan(prefs: list[str]) -> dict:
    """고른 현으로 무엇이 만들어질지. 화면이 빌드 전에 보여준다."""
    rid = combo_id(prefs)
    meta = region_json(rid, prefs)
    g = meta["grid"]
    area = g["span_x"] * g["span_y"] / 1e6
    return {
        "id": rid,
        "prefectures": prefs,
        "names": meta["names"],
        "span_km": [round(g["span_x"] / 1000), round(g["span_y"] / 1000)],
        "area_km2": round(area),
        "extracts": meta["osm_extracts"],
        "missing": [e for e in meta["osm_extracts"] if not (OSM / f"{e}-latest.osm.pbf").exists()],
        "heavy": area > WARN_AREA_KM2,
        "too_big": area > MAX_AREA_KM2,
        "state": state(rid),
    }


def create(rid: str, prefs: list[str]) -> dict:
    meta = region_json(rid, prefs)
    base = REGIONS / rid
    base.mkdir(parents=True, exist_ok=True)
    (base / "region.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8")
    # build_walk 와 build_admin 이 역 목록을 먼저 요구한다. build_naive 가 덮는다.
    (base / "stops.json").write_text(json.dumps(
        {"ids": ["STUB"], "coords": [meta["center"]]}), encoding="utf-8")
    return meta


def download(extract: str) -> None:
    """Geofabrik 추출본을 받는다. 받다 끊기면 .part 가 남고 다음에 처음부터 받는다."""
    OSM.mkdir(parents=True, exist_ok=True)
    out = OSM / f"{extract}-latest.osm.pbf"
    part = out.with_suffix(".part")
    req = urllib.request.Request(GEOFABRIK.format(extract),
                                 headers={"User-Agent": "rail-isochrone"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        got, shown = 0, -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            pct = int(got * 100 / total) if total else 0
            if pct // 5 != shown:
                shown = pct // 5
                print(f"      {extract} {got / 1e6:,.0f}/{total / 1e6:,.0f} MB", flush=True)
    part.replace(out)


def build(rid: str) -> None:
    """빌드 순서대로 끝까지 돌린다. 단계마다 "[3/9] build_rail.py" 를 찍는다.

    단계의 출력은 data/regions/<id>/build.log 에 쌓인다. 화면은 이 파일의
    마지막 줄을 진행 상황으로 보여준다.
    """
    base = REGIONS / rid
    log = base / "build.log"
    env = dict(os.environ, REGION=rid, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        for k, (script, opt) in enumerate(STEPS, 1):
            print(f"[{k}/{len(STEPS)}] {script}", flush=True)
            f.write(f"\n===== [{k}/{len(STEPS)}] {script} =====\n")
            f.flush()
            r = subprocess.run([sys.executable, str(ROOT / "src" / script)],
                               env=dict(env, **opt.get("env", {})), cwd=ROOT,
                               stdout=f, stderr=subprocess.STDOUT)
            if r.returncode != 0 and not opt.get("optional"):
                raise RuntimeError(f"{script} 가 실패했다(종료 코드 {r.returncode}). {log} 를 보세요.")
            if script == "build_admin.py" and not opt.get("env"):
                _check_rings(rid)
    print(f"끝 ({(time.time() - t0) / 60:.0f}분)", flush=True)


def _check_rings(rid: str) -> None:
    """고른 현이 모두 경계를 얻었나. build_admin 은 못 찾은 현을 알리고 넘어가는데,
    조합 권역에서 그러면 현 하나가 조용히 빠진 권역이 만들어진다."""
    base = REGIONS / rid
    prefs = json.loads((base / "region.json").read_text(encoding="utf-8"))["prefectures"]
    try:
        rings = json.loads((base / "raw" / "prefecture-rings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rings = {}
    missing = [p for p in prefs if not rings.get(p)]
    if missing:
        raise RuntimeError("현 경계를 닫지 못했다: " + ", ".join(missing)
                           + ". 이웃 현이 든 추출본이 빠졌을 수 있다.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prefectures", nargs="+", help="현 이름(일본어·한국어·영어)")
    ap.add_argument("--id", help="권역 id. 없으면 현 번호로 짓는다(custom_33_37)")
    ap.add_argument("--build", action="store_true", help="빌드까지 돌린다")
    ap.add_argument("--force", action="store_true", help="같은 id 가 있어도 덮는다")
    args = ap.parse_args()

    try:
        prefs = resolve(args.prefectures)
    except ValueError as err:
        sys.exit(str(err))
    rid = args.id or combo_id(prefs)
    if not re.fullmatch(r"[a-z0-9_]+", rid):
        sys.exit("권역 id 는 영문 소문자·숫자·_ 만 쓴다")
    info = plan(prefs)
    if info["too_big"]:
        sys.exit(f"격자가 {info['span_km'][0]}x{info['span_km'][1]}km 라 너무 크다. 현을 나누세요.")
    if (REGIONS / rid / "region.json").exists() and not args.force:
        sys.exit(f"{rid} 가 이미 있다. 덮으려면 --force")
    print(f"{rid}: {'·'.join(prefs)}  격자 {info['span_km'][0]}x{info['span_km'][1]}km  "
          f"추출본 {', '.join(info['extracts'])}", flush=True)
    if info["heavy"]:
        print("  !! 격자가 크다. 보행망 빌드에 메모리가 많이 든다.", flush=True)
    meta = create(rid, prefs)
    print(f"  {REGIONS / rid / 'region.json'}  첫 출발지 {meta['start']['names']['ja']}", flush=True)
    if not args.build:
        if info["missing"]:
            print("  추출본이 없다. --build 로 돌리면 받는다: " + ", ".join(info["missing"]))
        return
    for e in info["missing"]:
        print(f"[0/{len(STEPS)}] 추출본 받기 {e}", flush=True)
        download(e)
    try:
        build(rid)
    except RuntimeError as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
