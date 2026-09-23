"""권역 하나에 필요한 데이터 일습.

지금까지는 경로가 모듈 여섯 군데에 흩어져 있어서 권역을 늘릴 수가 없었다.
한 권역이 쓰는 것들 — 시각표 그래프, 역 목록, 보행망, 노선 선형, 범위 —
을 이 클래스가 한꺼번에 들고 있는다.

데이터는 data/regions/<id>/ 아래에 모인다. OSM 추출본만 빌드 입력이라
data/osm/ 에 공유로 둔다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import geometry as geometry_mod

ROOT = Path(__file__).resolve().parent.parent
REGIONS_DIR = ROOT / "data" / "regions"

# 화면이 고를 수 있는 언어. 역 이름을 이만큼 내보낸다.
LANGS = ("ja", "en", "ko", "zh-Hans", "zh-Hant")


def available() -> list[str]:
    """준비된 권역 id 목록."""
    if not REGIONS_DIR.exists():
        return []
    return sorted(
        d.name for d in REGIONS_DIR.iterdir()
        if (d / "region.json").exists() and (d / "stops.json").exists()
    )


@dataclass
class Region:
    id: str
    meta: dict
    dir: Path
    stops: dict
    coords: np.ndarray
    graphs: dict          # calendar -> router.Graph
    railways: dict        # railway id -> 제목/색
    walk: object | None   # walknet.WalkNet
    fine: object | None   # finegeom.FineGeometry (선을 도로에 붙일 때만)
    supported: np.ndarray  # 묶음 번호 -> 도보권이 있어 실제로 쓸 수 있는가
    railway_shapes: list   # 지도에 겹쳐 그릴 노선 선형 (단순화된 것)
    pref_names: dict       # 도도부현 일본어 이름 -> 언어별 표기
    geometry: object      # geometry.Geometry
    coverage: object | None
    coverage_geojson: dict | None
    search_index: list[dict]
    # 역 목록은 노선별로 쪼개져 있다. 신주쿠 하나가 11개 항목이다.
    # 세어 보여줄 때는 사람이 아는 "역" 단위여야 하므로 묶음 번호를 들고 있는다.
    station_group: np.ndarray   # 역 인덱스 -> 묶음 번호 (-1 이면 좌표 없음)
    n_groups: int

    @property
    def names(self) -> dict:
        """권역 이름을 언어별로. 없는 언어는 id 로 때운다."""
        listed = self.meta.get("names") or {}
        return {lang: listed.get(lang) or listed.get("ja") or self.id for lang in LANGS}

    @property
    def name(self) -> str:
        """로그나 오류 메시지용. 화면은 names 를 쓴다."""
        return self.names.get("ja", self.id)


def load(region_id: str) -> Region:
    """권역 하나를 통째로 올린다."""
    import finegeom
    import walknet
    from coverage import Coverage
    from geometry import Geometry
    from router import load_graph

    base = REGIONS_DIR / region_id
    meta = json.loads((base / "region.json").read_text(encoding="utf-8"))
    stops = json.loads((base / "stops.json").read_text(encoding="utf-8"))
    coords = np.array(stops["coords"], dtype=np.float64)

    graphs = {
        calendar: load_graph(base / f"graph-{calendar}.npz")
        for calendar in ("Weekday", "SaturdayHoliday")
        if (base / f"graph-{calendar}.npz").exists()
    }

    railways = {
        r["id"]: r
        for r in json.loads((base / "raw" / "railways.json").read_text(encoding="utf-8"))
    }
    # OSM 관계에는 일본어 이름뿐인 노선이 많다. build_geometry.py 가
    # 짝이 된 시각표 권역에서 빌려 적어 둔 이름으로 빈 언어를 메운다.
    name_path = base / "raw" / "line-names.json"
    if name_path.exists():
        lent = json.loads(name_path.read_text(encoding="utf-8"))
        for rid, names in lent.items():
            r = railways.get(rid)
            if r is None:
                continue
            title = r.setdefault("title", {})
            for lang, v in names.items():
                if lang != "ja" and not (title.get(lang) or "").strip():
                    title[lang] = v

    # operator 태그도 network 태그도 없는 노선이 있다. 그대로 두면 노선별
    # 보기 패널에서 전부 "그 밖" 으로 떨어져, 東海道本線 같은 큰 노선이 회사 없이
    # 선다. 두 단계로 메운다.
    #
    #  1. 노선 이름 안에 회사가 들어 있으면 그걸 쓴다. OSM 은 태그를
    #     비워 두고 이름에만 적어 두는 일이 많다(真岡鐵道真岡線).
    #  2. 손으로 적어 둔 data/line-operators.json 을 본다. 両毛線 처럼
    #     이름에도 회사가 없고 JR 로 시작하지도 않는 것들이다.
    #  3. 이름이 JR 로 시작하면 그 권역에서 가장 흔한 JR 회사로 본다.
    #  4. 그래도 비면 역을 가장 많이 공유하는 노선의 회사를 물려받는다.
    #     같은 회랑을 달리는 노선은 대개 같은 회사다. 절반 넘게 겹칠 때만
    #     쓴다. 환승역 하나 겹쳤다고 물려받으면 엉뚱한 회사가 붙는다.
    from collections import Counter
    from operators import operator_in_name

    for r in railways.values():
        if (r.get("operator") or "").strip():
            continue
        got = operator_in_name((r.get("title") or {}).get("ja", ""))
        if got:
            r["operator"] = got

    # 손으로 적은 것이 짐작(3)보다 먼저다. JR伊東線 은 도카이 권역에서
    # 흔한 JR도카이로 짐작되지만 JR동일본 노선이다.
    from regional import merge_books

    hand = merge_books(_line_operators(), region_id)
    for r in railways.values():
        if (r.get("operator") or "").strip():
            continue
        got = hand.get((r.get("title") or {}).get("ja", ""))
        if got:
            r["operator"] = got

    jr_seen = Counter()
    for r in railways.values():
        got = (r.get("operator") or "").strip()
        ja = (r.get("title") or {}).get("ja", "")
        if got and ja.startswith("JR"):
            jr_seen[got] += 1
    if jr_seen:
        jr_default = jr_seen.most_common(1)[0][0]
        for r in railways.values():
            ja = (r.get("title") or {}).get("ja", "")
            if ja.startswith("JR") and not (r.get("operator") or "").strip():
                r["operator"] = jr_default

    have_op = [(r, set(r.get("stations") or [])) for r in railways.values()
               if (r.get("operator") or "").strip()]
    for r in railways.values():
        if (r.get("operator") or "").strip():
            continue
        mine = set(r.get("stations") or [])
        if len(mine) < 3:
            continue
        best, best_share = None, 0.0
        for other, theirs in have_op:
            if not theirs:
                continue
            share = len(mine & theirs) / len(mine)
            if share > best_share:
                best_share, best = share, other
        if best is not None and best_share >= 0.5:
            r["operator"] = best["operator"]

    # 한국어 이름의 띄어쓰기를 맞춘다. ODPT 는 "게이요 선" 처럼 선 앞을
    # 띄우고 손으로 적은 것은 "게이요선" 이라 같은 노선이 두 가지로 보였다.
    # 선 앞의 공백만 없앤다. "산요 본선", "아비코 지선" 은 그대로 둔다.
    import re as _re2

    _ko_gap = _re2.compile(r"\s+선(?=$|[\s)\]·・])")
    for r in railways.values():
        title = r.get("title") or {}
        ko = (title.get("ko") or "").strip()
        if ko:
            title["ko"] = _ko_gap.sub("선", ko)

    # 손으로 적어 둔 이름. 빌려올 짝이 없는 노선을 여기 넣는다.
    # 권역을 가리지 않고 일본어 이름으로 찾는다.
    hand_path = ROOT / "data" / "line-names.json"
    from_hand: set[str] = set()
    if hand_path.exists():
        hand = json.loads(hand_path.read_text(encoding="utf-8"))
        for r in railways.values():
            title = r.setdefault("title", {})
            got = hand.get((title.get("ja") or "").strip())
            if not isinstance(got, dict):
                continue
            # 회사 이름이 노선 이름의 일부인 것만 아래 떼기에서 뺀다.
            # 西武園線 의 "세이부엔", 京王新線 의 "게이오신" 이 그렇다.
            # 표시가 "엔선", "신선" 이 돼 버린다.
            if got.get("keep") and (got.get("ko") or "").strip():
                from_hand.add(r["id"])
            # 이 파일이 이긴다. 여기서 고쳐 이름을 통일할 수 있어야 한다.
            # ODPT 는 같은 회사인데 浅草線 을 "도에이 아사쿠사선", 三田線 을
            # "미타선" 으로 적어 두는 식이라 손으로 맞춰야 한다.
            for lang, v in got.items():
                # keep 은 언어가 아니라 "회사 떼기에서 빼라" 는 표시다.
                if lang in ("ja", "keep") or not isinstance(v, str):
                    continue
                if v.strip():
                    title[lang] = v

    # 이름 끝에 붙은 경로 설명을 뗀다. "(오사키 → 신키바)", ": 메구로→
    # 니시타카시마다이라" 같은 것들이다. 어느 구간인지는 노선별 보기 패널이
    # span 으로 따로 적으므로 이름에 또 적으면 두 번 적힌 꼴이 된다.
    # 화살표가 든 것만 뗀다. "도쿄 사쿠라 트램 (아라카와선)" 은 남긴다.
    # 화살표뿐 아니라 "(히가시코이즈미-오타)" 처럼 하이픈으로 구간을
    # 적어 둔 것도 뗀다. "(아라카와선)" 같은 딴이름은 남긴다.
    _arrow = _re.compile(r"[→⇒]|=>|->|[^\s(（]\s*[-–—]\s*[^\s)）]")
    _tail = _re.compile(r"\s*[(（][^)）]*[)）]\s*$")
    for r in railways.values():
        title = r.get("title") or {}
        for lang in ("ko", "en", "zh-Hans", "zh-Hant"):
            v = (title.get(lang) or "").strip()
            if not v:
                continue
            m = _tail.search(v)
            if m and _arrow.search(m.group(0)):
                v = v[:m.start()].strip()
            cut = _re.split(r"\s*[:：]\s*", v, maxsplit=1)
            if len(cut) == 2 and _arrow.search(cut[1]) and cut[0].strip():
                v = cut[0].strip()
            if v:
                title[lang] = v

    # 이름 끝의 종별을 뗀다. 노선이 아니라 그 노선을 달리는 열차의
    # 등급이다. 다만 일본어 이름에서 종별이 한 칸 띄어 적혀 있을 때만
    # 뗀다. ODPT 의 常磐線快速·常磐線各駅停車 는 붙여 적고 그 자체가
    # 노선 이름이라, 떼면 둘 다 "조반선" 이 되어 가릴 수가 없다.
    _kind_ja = _re.compile(
        r"[\s・]+(普通|新快速|快速急行|通勤快速|区間快速|区間急行|通勤急行|"
        r"各駅停車|快速|準急|急行|特急)")
    _kind_ko = _re.compile(
        r"\s+(보통|신쾌속|쾌속급행|통근쾌속|구간쾌속|구간급행|통근급행|"
        r"각역정차|쾌속|준급|급행|특급)$")
    for r in railways.values():
        title = r.get("title") or {}
        if not _kind_ja.search(title.get("ja") or ""):
            continue
        ko = (title.get("ko") or "").strip()
        cut = _kind_ko.sub("", ko).strip()
        if cut and cut != ko:
            title["ko"] = cut

    # 이름 앞에 붙은 회사 이름을 뗀다. 목록은 회사로 묶어 보여주므로
    # 겹쳐 적을 이유가 없고, ODPT 는 山手線 을 "야마노테선" 으로 OSM 은
    # "JR 야마노테선" 으로 적어 두는 식이라 섞이면 지저분하다. 아래에서
    # 이름이 겹치는 노선에만 회사를 다시 붙인다.
    # 긴 이름부터 본다. "한큐" 를 먼저 떼면 "한큐전철 교토선" 이
    # "전철 교토선" 이 된다. 실제로 OSM 의 name:ko 가 그렇게 적혀 있다.
    from operators import RAIL_OPERATORS as _RO

    _LEAD_WORDS = sorted(
        {v[2] for v in _RO.values()} | {
            "JR", "도쿄 메트로", "오사카 메트로", "도에이",
            "한큐전철", "한신전철", "게이한전철", "난카이전철", "산요전철",
            "도부철도", "세이부철도", "게이세이전철", "게이오전철",
            "오다큐전철", "도큐전철", "게이큐전철", "긴키닛폰철도",
            "오사카고속전기궤도", "오사카시고속전기궤도", "오사카시영지하철",
            "교토시영지하철", "고베시영지하철", "요코하마시영지하철",
        }, key=len, reverse=True)
    # 뗀 자리에 회사 이름의 뒷동강이 남으면 잘못 뗀 것이다.
    _lead_bad = _re.compile(r"^(전철|철도|전기철도|고속철도|여객철도|지하철)")

    def _strip_lead(ko: str) -> str:
        """앞에 붙은 회사 이름을 뗀다. 띄어 적은 것도 같이 본다.

        OSM 은 "게이오 전철 게이오선" 처럼 띄어 적기도 한다. 공백을
        지운 채로 견주고, 뗄 때는 원문에서 그만큼만 잘라 낸다.
        """
        flat = ko.replace(" ", "")
        for w in _LEAD_WORDS:
            # 회사 이름 쪽도 공백을 지우고 견준다. "욧카이치 아스나로철도"
            # 처럼 띄어 적힌 회사가 안 떼어지고 있었다.
            w = w.replace(" ", "")
            if not flat.startswith(w) or len(flat) == len(w):
                continue
            n = taken = 0
            while taken < len(w) and n < len(ko):
                if ko[n] != " ":
                    taken += 1
                n += 1
            rest = ko[n:].strip()
            # 떼고 나서 남은 것이 "선" 한 글자거나 "본선" 으로 시작하면
            # 회사 이름이 아니라 노선 이름을 자른 것이다.
            # 芝山鉄道線("시바야마철도선"), 山陽本線("산요 본선")이 그렇다.
            # "1호선" 처럼 번호만 남아도 되돌린다. 지바 모노레일 1호선이
            # 그냥 "1호선" 이 됐다.
            if (len(rest) >= 3 and not _lead_bad.match(rest)
                    and not _re.match(r"(본선|지선|신선)(\s|$|[(（·])", rest)
                    and not _re.match(r"\d+호선", rest)):
                return rest
            return ko
        return ko
    for r in railways.values():
        # 사전에서 keep 을 달아 둔 이름은 깎지 않는다.
        if r["id"] in from_hand:
            continue
        title = r.get("title") or {}
        ko = (title.get("ko") or "").strip()
        if ko:
            title["ko"] = _strip_lead(ko)

    # 같은 이름을 쓰는 노선이 회사별로 둘 이상이면 회사를 앞에 붙인다.
    # 日光線 은 JR 과 東武 에 다 있는데 둘 다 "닛코선" 이면 가릴 수가 없다.
    # 한 회사 안에서 이름이 겹치는 것(계통이 여럿인 경우)은 그대로 둔다.
    #
    # 겹치는지는 꼬리표를 뗀 이름으로 본다. 西武新宿線 은 "신주쿠선",
    # 都営新宿線 은 "신주쿠선 (신주쿠 → 모토야와타)" 라 글자로는 다르지만
    # 목록에서는 둘 다 "신주쿠선" 으로 보인다.
    #
    # JR 에는 안 붙인다. JR 노선이 압도적으로 많아 다 붙이면 목록이
    # 회사 이름으로 도배된다. 겹치는 쪽(東武·西武)만 붙여도 갈린다.
    # 다만 JR 회사끼리 겹치면(東海道本線 이 JR동일본과 JR도카이에 다 있다)
    # 그때는 JR 쪽에도 붙여야 가려진다.
    from collections import defaultdict as _dd

    def _bare(ko: str) -> str:
        return ko.split("(")[0].split("（")[0].strip() or ko

    _by_ko = _dd(set)
    for r in railways.values():
        ko = ((r.get("title") or {}).get("ko") or "").strip()
        op = ((_operator_label(r) or {}).get("ko") or "").strip()
        if ko:
            _by_ko[_bare(ko)].add(op)
    for r in railways.values():
        title = r.get("title") or {}
        ko = (title.get("ko") or "").strip()
        op = ((_operator_label(r) or {}).get("ko") or "").strip()
        if not ko or not op:
            continue
        ops = _by_ko[_bare(ko)]
        if len(ops) < 2 or ko.startswith(op):
            continue
        if op.startswith("JR") and len([o for o in ops if o.startswith("JR")]) < 2:
            continue
        title["ko"] = op + " " + ko


    walk = walknet.load(base / "walk")
    fine = finegeom.load(base / "walk")
    # OSM 선로로 갈아 끼운 선형이 있으면 그것을 쓴다. mini-tokyo-3d 선형은
    # 도심에서 도식적이고, 선로를 공유하는 구간이 노선별로 갈려 있지 않아
    # 게이힌토호쿠선이 주오선 선로를 따라가는 식으로 잘못 그려진다.
    geo_path = base / "raw" / "coordinates-osm.json"
    if not geo_path.exists():
        geo_path = base / "raw" / "coordinates.json"
    # 역에서 역까지의 선로는 따로 둔다. build_track.py 가 OSM 권역에서
    # 쓰고, 시각표 권역은 build_geometry.py 가 역 id 를 바꿔 적어 둔다.
    geometry = Geometry(geo_path, stops["railway"], coords,
                        station_ids=stops["ids"],
                        segments_path=base / "raw" / "track-segments.json")

    # 역 이름도 비어 있는 언어를 사전에서 채운다. OSM 권역은 name:ko 가
    # 없는 역이 3분의 1이라 목록과 경로에 일본어가 그대로 남았다.
    st_path = ROOT / "data" / "station-names.json"
    if st_path.exists():
        book = json.loads(st_path.read_text(encoding="utf-8"))
        for i, ja in enumerate(stops.get("ja") or []):
            got = book.get((ja or "").strip())
            if not isinstance(got, dict):
                continue
            for lang, v in got.items():
                if lang == "ja" or lang not in stops:
                    continue
                if not (stops[lang][i] or "").strip():
                    stops[lang][i] = v

    index, groups, n_groups = build_search_index(
        base, stops, coords,
        {rid: ((_operator_label(r) or {}).get("ko") or "")
         for rid, r in railways.items()})
    supported = supported_mask(walk, coords, groups, n_groups)

    # 역 -> 도도부현. build_admin.py 가 OSM 행정경계로 만들어 둔다.
    pref_path = base / "prefectures.json"
    pref_data = (
        json.loads(pref_path.read_text(encoding="utf-8")) if pref_path.exists() else {}
    )
    prefectures = pref_data.get("stations", {})
    pref_names = pref_data.get("names", {})
    row_of = {sid: i for i, sid in enumerate(stops["ids"])}
    for entry in index:
        i = row_of.get(entry["id"], -1)
        g = int(groups[i]) if i >= 0 else -1
        entry["supported"] = bool(supported[g]) if g >= 0 else False
        entry["pref"] = prefectures.get(entry["id"])

    # 범위는 역 위치에서 잡는다. 보행망 전체로 잡으면 우리가 다루지 않는
    # 노선의 역들까지 선 안에 들어온다. 실제로 쓸 수 있는 역만, 노선별 중복을
    # 합쳐서 넘긴다. 중복이 섞이면 이웃 거리가 0 이 되어 반경이 무너진다.
    cover = None
    if walk is not None:
        pts, ends = coverage_seeds(base, stops, coords, groups, n_groups, supported)
        cover = Coverage(walk, base / "walk" / "land.npz",
                         stations=pts, line_ends=ends)

        # 행정 단위로 정한 권역은 경계도 그렇게 잘라야 한다. 도달 범위로만
        # 그리면 노선이 뻗은 만큼 이웃 지방으로 삐져나간다.
        rings_path = base / "raw" / "prefecture-rings.json"
        wanted = meta.get("prefectures") or []
        if cover is not None and wanted and rings_path.exists():
            data = json.loads(rings_path.read_text(encoding="utf-8"))
            rings = [r for name in wanted for r in data.get(name, [])]
            if rings:
                ok = np.isfinite(coords[:, 0])
                line_path = base / "raw" / "prefecture-outline.json"
                outline = (json.loads(line_path.read_text(encoding="utf-8"))
                           if line_path.exists() else None)
                cover.clip_to_rings(rings, coords[ok][:, :2], outline)

    return Region(
        id=region_id,
        meta=meta,
        dir=base,
        stops=stops,
        coords=coords,
        graphs=graphs,
        railways=railways,
        walk=walk,
        fine=fine,
        geometry=geometry,
        coverage=cover,
        coverage_geojson=cover.boundary_geojson() if cover is not None else None,
        search_index=index,
        station_group=groups,
        n_groups=n_groups,
        supported=supported,
        railway_shapes=railway_shapes(geometry, railways, stops, coords, groups,
                                      supported, prefectures),
        pref_names=pref_names,
    )


def build_search_index(base: Path, stops: dict, coords: np.ndarray,
                       op_of: dict | None = None):
    """검색창에 쓸 역 목록. 같은 역은 한 줄로 합친다.

    신주쿠처럼 여러 철도사가 들어오는 역은 노선 수만큼 항목이 생기는데,
    이용자 입장에서는 전부 같은 "신주쿠역" 이다. station-groups.json 의
    묶음을 기준으로 합치고, 어느 회사들이 지나는지를 함께 보여준다.
    """
    from collections import Counter

    from operators import operator_of, sort_key, title as operator_title

    groups_path = base / "raw" / "station-groups.json"
    groups = (
        json.loads(groups_path.read_text(encoding="utf-8"))
        if groups_path.exists() else []
    )

    group_of: dict[str, int] = {}
    for gi, group in enumerate(groups):
        for sub in group:
            for sid in sub:
                group_of[sid] = gi

    buckets: dict[object, list[int]] = {}
    for i, sid in enumerate(stops["ids"]):
        if not np.isfinite(coords[i, 0]):
            continue
        # 그룹에 없는 역은 역명 + 대략적인 위치로 묶는다 (100 m 안팎)
        key = group_of.get(sid)
        if key is None and "cluster" in stops:
            # 묶음 번호가 있으면 그게 곧 "한 역" 이다. 좌표를
            # 반올림해 묶으면 승강장이 갈린 역이 두 줄로 갈라진다.
            key = ("cluster", stops["cluster"][i])
        if key is None:
            key = (stops["ja"][i], round(coords[i, 0], 3), round(coords[i, 1], 3))
        buckets.setdefault(key, []).append(i)

    # 묶음이 갈려 있어도 이름이 같고 가까우면 검색에서는 한 줄이다.
    # 東京 이 두 묶음이라 목록에 두 번 섰다. 지도에 찍는 자리는 아래에서
    # 승강장마다 따로 두므로, 여기서 합쳐도 지도는 갈라진 채로 남는다.
    keys = list(buckets)
    home = {k: k for k in keys}

    def root(k):
        while home[k] != k:
            home[k] = home[home[k]]
            k = home[k]
        return k

    spots = {}
    for k in keys:
        rows = buckets[k]
        spots[k] = (stops["ja"][rows[0]],
                    float(np.mean(coords[rows, 0])),
                    float(np.mean(coords[rows, 1])))
    for a_i in range(len(keys)):
        for b_i in range(a_i + 1, len(keys)):
            ka, kb = keys[a_i], keys[b_i]
            na, xa, ya = spots[ka]
            nb, xb, yb = spots[kb]
            if not na or na != nb:
                continue
            # 도쿄역은 게이요선 승강장이 본 개찰에서 500m 쯤 떨어져 있다.
            # 이름이 같아야 하므로 넉넉히 잡아도 남의 역을 삼키지 않는다.
            if abs(xa - xb) * 91_000.0 > 700.0 or abs(ya - yb) * 111_132.0 > 700.0:
                continue
            ra, rb = root(ka), root(kb)
            if ra != rb:
                home[rb] = ra
    if any(home[k] != k for k in keys):
        joined = {}
        for k in keys:
            joined.setdefault(root(k), []).extend(buckets[k])
        buckets = joined

    out = []
    group_of_station = np.full(len(stops["ids"]), -1, dtype=np.int32)
    for gi, members in enumerate(buckets.values()):
        group_of_station[members] = gi
        # 한 그룹 안에 이름이 섞여 있으면 (신주쿠 / 신주쿠니시구치) 다수를 따른다
        members = list(members)
        names = Counter(stops["ja"][i] for i in members)
        ja = names.most_common(1)[0][0]
        lead = next(i for i in members if stops["ja"][i] == ja)

        # OSM 권역은 역 id 접두사가 전부 "OSM" 이라 회사를 알 수 없다.
        # 그 역이 속한 노선의 operator 를 쓴다.
        got = set()
        for i in members:
            lid = stops["railway"][i]
            name = (op_of or {}).get(lid)
            if name:
                got.add(name)
        if got:
            prefixes = sorted(got)
        else:
            prefixes = sorted({operator_of(stops["ids"][i]) for i in members},
                              key=sort_key)
        # 지도에 찍는 자리는 승강장마다 따로 둔다. 검색과 세는 단위는
        # 한 줄이지만, 豊島園 처럼 이름만 같고 승강장이 따로인 역을
        # 한 점으로 묶으면 지도가 거짓말을 한다. 40m 안은 같은 자리로 본다.
        spots = []
        for i in members:
            lon, lat = float(coords[i, 0]), float(coords[i, 1])
            if not any(abs(lon - u) * 91_000.0 < 40.0
                       and abs(lat - v) * 111_132.0 < 40.0 for u, v in spots):
                spots.append((lon, lat))
        out.append(
            {
                "id": stops["ids"][lead],
                "lon": float(np.mean(coords[members, 0])),
                "lat": float(np.mean(coords[members, 1])),
                "spots": [[round(u, 6), round(v, 6)] for u, v in spots],
                **{lang: (stops[lang][lead] if lang != "ja" else ja)
                   for lang in LANGS if lang in stops},
                "operators": [p if got else operator_title(p, "ko")
                          for p in prefixes],
            }
        )

    out.sort(key=lambda s: s["ja"])
    return out, group_of_station, len(buckets)


def supported_mask(walk, coords: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """역 묶음별로 "실제로 쓸 수 있는가".

    OSM 추출본이 간토뿐이라 야마나시·이즈·시즈오카의 역은 보행망에 붙지
    못한다. 그런 역은 도보권이 없어 권역에 아무것도 칠하지 못하고, 출발지로
    찍어도 계산이 안 된다. 지도에 "지원 역" 으로 점을 찍으면 거짓말이 된다.
    """
    ok = np.zeros(max(n_groups, 1), dtype=bool)
    if walk is None:
        return ok
    for i in range(len(coords)):
        g = int(groups[i])
        if g < 0 or ok[g]:
            continue
        if int(walk.station_node[i]) >= 0 and len(walk.shed(i)[0]) > 0:
            ok[g] = True
    return ok


def coverage_seeds(base: Path, stops: dict, coords: np.ndarray, groups: np.ndarray,
                   n_groups: int, supported: np.ndarray):
    """권역을 칠할 시드 점과, 각 점이 노선의 끝인지.

    끝인지는 노선의 정차역 목록에서 본다. 어떤 노선의 첫 역이거나 마지막
    역이면서, 다른 노선의 중간역이 아닌 역이 끝이다. 예전에는 이웃 역의
    방향이 한쪽으로 쏠렸는지로 짐작했는데, 그러면 망이 성겨지는 외곽의
    중간역(하스다, 가모노미야, 네부카와)까지 끝으로 잡혔다.
    """
    railways = json.loads((base / "raw" / "railways.json").read_text(encoding="utf-8"))
    index = {sid: i for i, sid in enumerate(stops["ids"])}

    ends: set[int] = set()
    middles: set[int] = set()
    for railway in railways:
        order = railway.get("stations") or []
        if len(order) < 2:
            continue
        for sid, is_end in [(order[0], True), (order[-1], True)]:
            i = index.get(sid)
            if i is not None and groups[i] >= 0:
                ends.add(int(groups[i]))
        for sid in order[1:-1]:
            i = index.get(sid)
            if i is not None and groups[i] >= 0:
                middles.add(int(groups[i]))
    line_ends = ends - middles

    sums = np.zeros((n_groups, 2))
    counts = np.zeros(n_groups)
    for i in range(len(coords)):
        g = int(groups[i])
        if g < 0 or not supported[g]:
            continue
        sums[g] += coords[i]
        counts[g] += 1

    keep = counts > 0
    pts = sums[keep] / counts[keep, None]
    flags = np.array([g in line_ends for g in np.flatnonzero(keep)], dtype=bool)
    return pts, flags


# 지도에 겹쳐 그릴 때 줄이는 정도. 25m 로 두면 야마노테선이 917점에서
# 54점으로 줄어 곡선이 2km 짜리 직선 하나가 된다. 3m 면 눈에 띄지 않고
# 전체 크기도 0.16MB 에서 0.47MB 로 느는 정도다.
SHAPE_TOLERANCE_M = 3.0
# 구간과 구간을 이을 때 이만큼 벌어지면 선을 끊는다.
#
# 300m 로 두었더니 역에서 애먼 자리를 끊었다. 구간마다 뽑히는 선형이
# 달라 같은 역에서 조금씩 어긋나는데, 세 권역을 재어 보니 그 어긋남이
# 최대 464m 이고 300m 를 넘는 자리가 18곳뿐이다. 전부 오사카/기타신치
# 처럼 300m 안의 두 역을 한 묶음으로 합친 자리다. 역 구내에서 그 정도
# 이어 붙이는 것은 눈에 띄지 않는다.
JOIN_TOLERANCE_M = 600.0
# 이음매가 이만큼 벌어지면 역을 거쳐 잇는다.
JOIN_VIA_STATION_M = 30.0
VIA_DETOUR_MAX = 1.15
# 선형 안에서 이만큼 벌어지면 거기서도 끊는다.
# geometry 가 후보를 고를 때도 같은 값을 봐야 한다. 달랐을 때 거기서
# 괜찮다고 고른 호가 여기서 토막 났다.
SPLIT_GAP_M = geometry_mod.DRAW_GAP_M


def _dp_mask(line: np.ndarray, tol_m: float) -> np.ndarray:
    """더글러스-포이커. 남길 점을 True 로."""
    keep = np.zeros(len(line), dtype=bool)
    if len(line) == 0:
        return keep
    keep[0] = keep[-1] = True
    if len(line) < 3:
        return keep
    scale = np.cos(np.radians(float(line[:, 1].mean())))
    stack = [(0, len(line) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = line[a], line[b]
        seg = line[a + 1:b]
        dx = (q[0] - p[0]) * scale * 111_320.0
        dy = (q[1] - p[1]) * 111_132.0
        length = np.hypot(dx, dy)
        sx = (seg[:, 0] - p[0]) * scale * 111_320.0
        sy = (seg[:, 1] - p[1]) * 111_132.0
        if length < 1e-9:
            dist = np.hypot(sx, sy)
        else:
            dist = np.abs(sx * dy - sy * dx) / length
        i = int(np.argmax(dist))
        if dist[i] > tol_m:
            keep[a + 1 + i] = True
            stack.append((a, a + 1 + i))
            stack.append((a + 1 + i, b))
    return keep


# 줄인 선이 꺾여 보이면 안 된다. 반경이 작은 곡선을 몇 m 허용오차로
# 줄이면 부드러운 곡선이 한 꼭짓점의 날카로운 각이 된다. 선형 자체는
# 멀쩡한데 화면에서는 선이 꺾인 것으로 보인다. 세 권역에서 120도 넘는
# 자리가 선로에는 4곳뿐인데 3m 로 줄이면 26곳이 됐다.
#
# 허용오차를 통째로 낮추면 점이 76% 는다. 꺾이는 자리만 원본을 되살린다.
SIMPLIFY_TURN_DEG = 100.0
SIMPLIFY_PASSES = 3


def _simplify(line: np.ndarray, tol_m: float) -> np.ndarray:
    """선 모양을 지키면서 점 수를 줄인다. 꺾여 보이는 자리는 되살린다."""
    if len(line) < 3:
        return line
    scale = float(np.cos(np.radians(float(line[:, 1].mean()))))
    keep = _dp_mask(line, tol_m)
    for _ in range(SIMPLIFY_PASSES):
        idx = np.flatnonzero(keep)
        add = []
        for t in range(1, len(idx) - 1):
            if geometry_mod.turn_deg(
                    line[idx[t - 1]], line[idx[t]], line[idx[t + 1]],
                    scale) <= SIMPLIFY_TURN_DEG:
                continue
            a, b = int(idx[t - 1]), int(idx[t + 1])
            if b - a < 2:
                continue
            sub = _dp_mask(line[a:b + 1], tol_m / 8.0)
            add.extend((a + np.flatnonzero(sub)).tolist())
        if not add:
            break
        keep[np.asarray(add, dtype=int)] = True
    return line[keep]


import re as _re

# 애칭이 붙은 열차 계통. 노선이 아니라 그 선로를 달리는 열차다. 지도에
# 따로 그리면 남의 선로 위에 겹쳐 그려져 어지럽기만 하다. 시각표에는
# 그대로 남으므로 경로 계산은 달라지지 않는다.
#
# えのしま 는 小田原線 과 江ノ島線 을 이어 달리는 특급이라 어느 한 노선의
# 부분집합이 아니고, 그래서 "통과 계통" 걸러내기에 안 걸렸다.
_NAMED_TRAIN = _re.compile(
    r"(?<!つくば)エクスプレス|特急|列車|"
    r"^(えのしま|ちちぶ|ひたち|スーパーはこね|こうのとり|はまかぜ|いなば|"
    r"きぬがわ|しおさい|わかしお|さざなみ|あずさ|かいじ|ときわ|"
    r"サフィール|スワローあかぎ|ラビュー|S-Train|Ｓ－Ｔｒａｉｎ|"
    r"踊り子|ミュースカイ|ひだ|しなの|南紀|ふじかわ|伊那路|しらさぎ|サンダーバード|"
    r"「?ソニック|「?にちりん|ゆふ|きらめき|みどり|ハウステンボス|有明|きりしま|"
    r"ひゅうが|スーパーおき|おき|スーパーいなば|南風|しおかぜ|うずしお|剣山|いしづち|しまんと|あしずり|むろと|宇和海|つばさ|つがる|いなほ)"
    r"(?=$|[\s(（:：・])")


_NORM_STRIP = _re.compile(
    "東日本旅客鉄道|西日本旅客鉄道|東海旅客鉄道|JR東日本|JR西日本|JR東海|ＪＲ|JR|"
    "東京地下鉄|東京メトロ|都営地下鉄|東京都交通局|横浜市営地下鉄|"
    "Osaka Metro|大阪市高速電気軌道|大阪メトロ|神戸市営地下鉄|京都市営地下鉄|"
    "近畿日本鉄道|京浜急行電鉄|京王電鉄|小田急電鉄|東京急行電鉄|東急電鉄|"
    "京成電鉄|西武鉄道|東武鉄道|相模鉄道|山陽電気鉄道|阪神電気鉄道|"
    "阪急電鉄|南海電気鉄道|京阪電気鉄道|神戸電鉄|北大阪急行電鉄|"
    "電気鉄道|急行電鉄|都市モノレール|新交通|電鉄|鉄道|株式会社")
_NORM_DROP = _re.compile(r"[（(][^）)]*[）)]")
_NORM_KIND = _re.compile(
    "各駅停車|各停|普通|新快速|快速急行|通勤快速|通勤急行|通勤準急|"
    "区間急行|区間快速|快速|特急|急行|準急|直通運転|上り|下り")
_NORM_PUNCT = _re.compile(r"[\s・･:：>=→\-–—]")


# 종별만 적혀 있거나 직통운전이라고만 적힌 이름. 노선이 아니라 운행
# 계통이다. 相鉄直通線, 小田急通勤急行, 快速急行 (神戸三宮 => 近鉄奈良),
# 東京地下鉄の直通運転 - 常磐線 같은 것들이 목록에 노선인 척 서 있었다.
_KIND_ONLY = _re.compile(
    r"^(各駅停車|各停|普通|新快速|区間快速|区間急行|通勤快速|通勤急行|"
    r"通勤準急|快速急行|快速|準急|急行|特急|直通線?|直通運転)$")


def _is_service_pattern(name: str) -> bool:
    """이름이 운행 계통인가. 회사 이름을 떼고 남은 것이 종별뿐이면 그렇다."""
    from operators import strip_operator_head

    raw = (name or "").strip()
    if "直通運転" in raw:
        return True
    s = _NORM_DROP.sub("", raw).split(":")[0].split("：")[0]
    s = strip_operator_head(_NORM_STRIP.sub("", s).strip())
    s = s.strip(" 　-–—・")
    return bool(_KIND_ONLY.match(s))


def _is_named_train(name: str) -> bool:
    """이름이 열차 이름이거나 운행 계통인가. 노선 이름이면 False."""
    name = (name or "").strip()
    return bool(_NAMED_TRAIN.search(name)) or _is_service_pattern(name)



def _norm_name(name: str) -> str:
    """권역마다 다르게 부르는 이름을 맞추기 좋게 다듬는다."""
    s = _NORM_DROP.sub("", name or "")
    s = _NORM_STRIP.sub("", s)
    s = _NORM_KIND.sub("", s)
    return _NORM_PUNCT.sub("", s)


# 지어낸 노선 색을 적어 두는 곳. 손으로 고칠 수 있게 파일로 둔다.
COLOR_DB = ROOT / "data" / "line-colors.json"
_color_table: dict[str, str] | None = None


# 위키가 색을 이름으로 적어 둔 경우. 그대로 내보내면 지도에서 안 먹는다.
_NAMED = {"red": "#ff0000", "blue": "#0000ff", "green": "#008000",
          "orange": "#ffa500", "yellow": "#ffff00", "purple": "#800080",
          "pink": "#ffc0cb", "brown": "#a52a2a", "gray": "#808080",
          "grey": "#808080", "black": "#000000", "skyblue": "#87ceeb",
          "navy": "#000080", "teal": "#008080", "olive": "#808000",
          "magenta": "#ff00ff", "cyan": "#00ffff", "lime": "#00ff00",
          "gold": "#ffd700", "silver": "#c0c0c0", "maroon": "#800000",
          "darkgreen": "#006400", "darkblue": "#00008b", "darkred": "#8b0000",
          "white": "#dddddd", "violet": "#ee82ee", "indigo": "#4b0082"}
_canon_table: dict[str, str] | None = None


def _hex(value) -> str | None:
    """색을 #rrggbb 로 맞춘다. 아니면 None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if _re.fullmatch(r"#[0-9A-Fa-f]{6}", v):
        return v
    if _re.fullmatch(r"#[0-9A-Fa-f]{3}", v):
        return "#" + "".join(c * 2 for c in v[1:])
    return _NAMED.get(v.lower())


def _canon() -> dict[str, str]:
    """다듬은 이름 -> 색 하나.

    같은 노선을 부르는 이름이 여럿이면 짧은 쪽을 고른다. 짧은 쪽이 대개
    정식 노선 이름이고(南武線 < JR南武線), 사업자·방향·종별이 덜 붙어
    있어 공식 색이 달려 있을 확률이 높다.

    다만 다듬으면 사업자가 떨어져 나가, 이름만 같은 남의 노선까지 한
    열쇠가 된다(江ノ島電鉄線 과 小田急江ノ島線, JR東西線 과 東京メトロ
    東西線, 京急大師線 과 東急大師線). 색이 갈리는 무리는 어느 쪽을
    골라도 한쪽이 틀리므로 아예 내놓지 않는다.
    """
    global _canon_table
    if _canon_table is None:
        best: dict[str, tuple[int, str]] = {}
        clash: set[str] = set()
        for key, val in _colors().items():
            color = _hex(val.get("color") if isinstance(val, dict) else val)
            if not color:
                continue
            want = _norm_name(key)
            if not want:
                continue
            if want in best and best[want][1] != color:
                clash.add(want)
            if want not in best or len(key) < best[want][0]:
                best[want] = (len(key), color)
        _canon_table = {k: v[1] for k, v in best.items() if k not in clash}
    return _canon_table


def _colors() -> dict[str, str]:
    global _color_table
    if _color_table is None:
        try:
            _color_table = json.loads(COLOR_DB.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _color_table = {}
    return _color_table


# 노선으로 잡히지만 타고 다닐 수 없는 것들. build_rail.py 가 OSM 권역에서
# 걸러 주지만, 시각표 권역은 ODPT 에서 오므로 여기서도 본다. 화물 전용선은
# 지도에 그려도 탈 수가 없고, 도식적인 선형이라 블록을 가로지르는 직선으로
# 그려져 눈에 거슬린다(東海道貨物線 이 무사시코스기에서 859m 를 가로지른다).
_SKIP: set[str] | None = None


_LINE_OPS: dict | None = None


def _line_operators() -> dict:
    """data/line-operators.json. 권역 -> 일본어 노선 이름 -> 회사."""
    global _LINE_OPS
    if _LINE_OPS is None:
        path = ROOT / "data" / "line-operators.json"
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            got = {}
        _LINE_OPS = {k: v for k, v in got.items() if isinstance(v, dict)}
    return _LINE_OPS


def _skip_line(name: str) -> bool:
    global _SKIP
    if _SKIP is None:
        path = ROOT / "data" / "excluded-lines.json"
        names: set[str] = set()
        try:
            for key, val in json.loads(path.read_text(encoding="utf-8")).items():
                if isinstance(val, list):
                    names.update(val)
        except (OSError, ValueError):
            pass
        _SKIP = names
    base = name.split("(")[0].split("（")[0].strip()
    return name in _SKIP or base in _SKIP or any(
        name.startswith(k) for k in _SKIP)


_SKIP_BY_ST: dict | None = None


def _skip_by_stations(name: str, station_names) -> bool:
    """이름이 같은 진짜 노선이 있어 이름만으로는 못 버리는 것.

    碓氷峠 철도문화마을의 관광 트롯코가 "信越本線" 이라는 이름을 달고
    있다. 진짜 信越本線 도 같은 권역에 있어 이름으로 버리면 둘 다
    없어진다. 적어 둔 역을 모두 들고 있을 때만 버린다.
    """
    global _SKIP_BY_ST
    if _SKIP_BY_ST is None:
        path = ROOT / "data" / "excluded-lines.json"
        got = {}
        try:
            for key, val in json.loads(path.read_text(encoding="utf-8")).items():
                if isinstance(val, dict):
                    got.update({k: set(v) for k, v in val.items()
                                if isinstance(v, list)})
        except (OSError, ValueError):
            pass
        _SKIP_BY_ST = got
    want = _SKIP_BY_ST.get(name.split("(")[0].split("（")[0].strip())
    return bool(want) and want <= set(station_names)


def _auto_color(name: str, rid: str) -> str | None:
    """data/line-colors.json 에 적힌 색. 없으면 None.

    OSM 관계에 colour 태그가 없는 노선이 간사이는 143개 중 80개,
    간토 OSM 판은 177개 중 72개다. 전부 회색으로 두면 나란히 달리는
    노선을 가를 수가 없다.

    먼저 data/line-colors.json 을 본다. 거기 적어 둔 것이 있으면 그것을
    쓴다. 손으로 고칠 수 있어야 해서 파일로 둔다. 없으면 이름에서 해시를
    떠서 지어낸다. 밝기와 채도는 고정해 밝은 지도에서도 어두운 지도에서도
    읽히게 한다. 노선 id 가 아니라 이름을 열쇠로 쓰는 것은, id 가 다시
    만들 때마다 바뀔 수 있기 때문이다.
    """
    # 적혀 있는 이름 그대로를 먼저 본다. build_colors.py 가 권역끼리
    # 역 목록으로 맞춰 같은 노선에 같은 색을 적어 두므로, 여기서 이름을
    # 다듬어 맞출 일이 없다.
    #
    # 다듬은 이름을 먼저 보던 때는 두 가지로 틀렸다. 江ノ島電鉄線 은
    # 電鉄 를 떼면 小田急 의 江ノ島線 과 같은 열쇠가 되어 에노덴이 오다큐
    # 파랑을 받았고, JR東西線 과 東京メトロ東西線, 京急大師線 과 東急大師線,
    # 東京メトロ南北線 과 北大阪急行南北線 이 모두 한 열쇠로 뭉쳤다.
    table = _colors()
    got = table.get(name) or table.get(rid)
    if isinstance(got, dict):
        got = got.get("color")
    if got is None:
        # 표에 없는 이름일 때만 다듬어 본다. 사업자가 갈리는 무리는
        # _canon 이 내놓지 않는다.
        got = _canon().get(_norm_name(name))
    return _hex(got)


def _operator_label(railway: dict) -> dict:
    """노선을 굴리는 회사 이름을 언어별로. 모르면 빈 사전.

    화면이 언어를 바꿔도 다시 받아오지 않아도 되도록 통째로 보낸다.
    """
    from operators import OPERATORS, rail_operator
    from operators import title as operator_title

    got = (railway.get("operator") or "").strip()
    if got:
        return rail_operator(got)
    head = str(railway.get("id", "")).split(".")[0]
    if head in OPERATORS:
        # 회사 표에 짝이 있으면 5개 언어가 다 나온다. 없으면 일본어와
        # 한국어뿐이라 나머지는 일본어로 둔다.
        return {lang: operator_title(head, lang)
                for lang in ("ja", "en", "ko", "zh-Hans", "zh-Hant")}
    return {}


def _qual_key(railway: dict) -> str:
    """사업자를 붙인 열쇠. build_colors.qual_key 와 같은 규칙이어야 한다.

    이름만으로는 갈리지 않는 노선이 있다. 간토의 新宿線 은 都営 와 西武
    둘이고, 日光線 은 JR 과 東武 둘이다. 한 칸을 나눠 쓰면 한쪽 색이
    다른 쪽을 덮는다.
    """
    ja = railway.get("title", {}).get("ja", "") or railway.get("id", "")
    op = (railway.get("operator") or "").strip()
    if not op:
        op = str(railway.get("id", "")).split(".")[0]
    return (op + "|" + ja) if op else ja


def line_color(railway: dict) -> str:
    """노선 하나의 색. 지도와 경로 패널이 같은 값을 쓰도록 여기서만 정한다.

    예전에는 지도는 이 순서를 타고 경로 패널은 raw 의 color 태그를 그대로
    썼다. 그래서 같은 노선인데 선 색과 경로 색이 달랐다.
    """
    table = _colors()
    got = table.get(_qual_key(railway))
    if isinstance(got, dict):
        got = got.get("color")
    return (_hex(got)
            or _auto_color(railway.get("title", {}).get("ja", ""),
                           railway.get("id", ""))
            or _hex(railway.get("color"))
            or _hex(railway.get("colour"))
            or "#888888")


def _dist_m(a, b) -> float:
    lat = np.radians((a[1] + b[1]) / 2)
    return float(np.hypot((b[0] - a[0]) * np.cos(lat) * 111_320.0,
                          (b[1] - a[1]) * 111_132.0))


def _path_len_m(line: np.ndarray) -> float:
    if len(line) < 2:
        return 0.0
    lat = np.radians(line[:-1, 1])
    return float(np.hypot((line[1:, 0] - line[:-1, 0]) * np.cos(lat) * 111_320.0,
                          (line[1:, 1] - line[:-1, 1]) * 111_132.0).sum())


def railway_shapes(geometry, railways: dict, stops: dict, coords: np.ndarray,
                   groups: np.ndarray, supported: np.ndarray,
                   prefectures: dict) -> list[dict]:
    """지도에 겹쳐 그릴 노선 선형.

    노선이 가진 선형을 그대로 쓰면 안 된다. 쇼난신주쿠라인이나 호쿠소선처럼
    남의 선로를 빌려 쓰는 계통은 자체 선형이 몇 점짜리 껍데기여서, 그리면
    제 역에서 15 km 떨어진 직선이 지도를 가로지른다. 대신 정차역을 차례로
    이어 구간마다 촘촘한 선형을 고르는 ride_path 를 쓴다. 이 길은 경로를
    그릴 때 이미 쓰고 있어 검증돼 있다.

    쓸 수 없는 역만 있는 노선(야마나시의 후지큐, 이즈의 이즈큐)은 뺀다.
    지원 범위를 보여주는 것이 목적인데 넣으면 거짓말이 된다.
    """
    row_of = {sid: i for i, sid in enumerate(stops["ids"])}
    lat_scale = float(np.cos(np.radians(float(np.nanmedian(coords[:, 1])))))
    sb_path = ROOT / "data" / "switchbacks.json"
    switchbacks = ({k: set(v) for k, v in json.loads(
        sb_path.read_text(encoding="utf-8")).items() if isinstance(v, list)}
        if sb_path.exists() else {})

    # 애칭이 붙은 열차 계통은 지도에 안 그린다. 다만 제 역이 다른 노선에
    # 다 들어 있을 때만이다. 그 계통만 닿는 역이 있으면 그려야 한다.
    # 역 id 는 노선마다 다르다(OSM.4.973 / OSM.JR.973). 이름으로 견준다.
    covered = set()
    all_st = {}
    for rid, railway in railways.items():
        all_st[rid] = {stops["ja"][row_of[s]] for s in (railway.get("stations") or [])
                       if s in row_of}
    for rid, mine in all_st.items():
        if not mine or not _is_named_train(railways[rid].get("title", {}).get("ja", "")):
            continue
        rest = set()
        for other, theirs in all_st.items():
            if other != rid:
                rest |= theirs
        if len(mine & rest) >= 0.95 * len(mine):
            covered.add(rid)

    out = []
    for rid, railway in railways.items():
        if rid in covered:
            continue
        if _skip_line(railway.get("title", {}).get("ja", "")):
            continue
        if _skip_by_stations(railway.get("title", {}).get("ja", ""),
                             all_st.get(rid) or set()):
            continue
        title_ja = railway.get("title", {}).get("ja", "")
        # 사전은 괄호 앞 이름으로도 찾는다. 제목에 구간이 붙은 노선이 많다
        # (富山地方鉄道本線 (電鉄富山=>稲荷町->宇奈月温泉)).
        sb_key = title_ja if title_ja in switchbacks else _re.sub(r"\s*[(（].*$", "", title_ja)
        order = railway.get("stations") or []
        rows = [row_of[s] for s in order
                if s in row_of and np.isfinite(coords[row_of[s], 0])]
        if len(rows) < 2:
            continue
        if not any(groups[i] >= 0 and supported[int(groups[i])] for i in rows):
            continue

        # 구간마다 따로 잇고, 선형이 없거나 크게 돌아가는 자리에서 끊는다.
        # 한 번에 이으면 그런 구간이 긴 직선이 되어 지도를 가로지른다.
        #
        # 노선 제 선형만 보는 쪽도 해 봤으나 더 나빴다(간토 3곳 -> 15곳).
        # 소부쾌속선이나 쇼난신주쿠라인처럼 남의 선로를 달리는 계통은
        # 제 선형이 그 구간을 안 덮기 때문이다.
        # 조각마다 어느 역에서 어느 역까지인지 함께 들고 간다. 한 노선이
        # 여러 조각으로 나오면 지도에서 끊겨 보이는데, 이름이 다 같아서
        # 어느 구간이 끊긴 것인지 가릴 수가 없었다(무사시노선이 그렇다).
        pieces, spans, path, using = [], [], [], None
        start = end = None
        for a, b in zip(rows, rows[1:]):
            got, src = geometry.segment_source(a, b, using, rid=rid)
            # 선로 구간은 노선을 고른 것이 아니라 그 구간의 선로 그 자체다.
            # 이걸 prefer 로 물려주면 다음 구간의 노선 고르기가 어긋난다.
            if src != geometry_mod.TRACK_SRC:
                using = src
            arc = np.asarray(got if got is not None
                             else [coords[a], coords[b]], dtype=np.float64)
            # 선로에서 뽑은 선형은 걸음이 커도 그게 실제 좌표다. 단나
            # 터널은 7.8km 를 곧게 뚫어 걸음도 7.8km 다. 도식적인 선형과
            # 이어 붙인 선형만 없는 구간을 현으로 때우므로, 그쪽에서만
            # 큰 걸음을 끊는다.
            made_up = src is None or "#" in src
            straight = _dist_m(coords[a], coords[b])
            ok = len(arc) >= 2 and straight > 0
            # 선로를 따라 찾아 둔 구간은 재볼 것이 없다. 길이로 걸러 내면
            # 실제로 크게 도는 선로(회차선, 삼각선)를 버리게 된다.
            if src == geometry_mod.TRACK_SRC:
                pass
            elif ok:
                arc_len = _path_len_m(arc)
                # 두 점짜리 직선은 선형이 없어 그냥 이은 것이다.
                # 크게 도는 것은 남의 선로를 빌려 온 것이다.
                if len(arc) <= 2 and straight > 400:
                    ok = False
                elif arc_len > straight * geometry_mod.MAX_DETOUR_RATIO:
                    # 고르는 쪽이 이미 이 배수를 넘는 호를 버렸다. 여기서
                    # 더 좁게 잡으면(2.0 배) 통과한 호를 또 잘라 낸다.
                    # 도부 사노선 사노시-사노가 2.0 배라 끊겼다.
                    ok = False
            if not ok:
                if len(path) >= 2:
                    pieces.append(path)
                    spans.append((start, end))
                path = []
                start = end = None
                continue
            # 앞 구간의 끝과 이 구간의 시작이 어긋나면 이어 붙이지 않는다.
            # 구간마다 붙는 선형이 달라 역에서 조금씩 벌어지는데, 그대로
            # 이으면 그 틈이 직선으로 그어진다.
            if path and _dist_m(path[-1], arc[0]) > JOIN_TOLERANCE_M:
                if len(path) >= 2:
                    pieces.append(path)
                    spans.append((start, end))
                path = []
                start = end = None
            # 앞 구간이 역을 지나쳐 나갔다가 이 구간이 같은 길을 되짚어
            # 오면 그만큼 걷어낸다. 안 걷어내면 역에서 뾰족하게 찌르고
            # 돌아오는 모양이 된다(긴자선 赤坂見附, 사이쿄선 赤羽).
            # 스위치백 역에서는 걷어내지 않는다. 養老線 은 大垣 에서 방향을
            # 바꿔 西大垣·室 쪽 열차가 같은 선로를 900m 오가는데, 그걸 걷어내
            # 선이 大垣 에 닿지 않았다. 규칙으로 가르려 하면 역을 지나쳤다
            # 돌아오는 인공 꼬리까지 되살아나(42~95개 노선) 손으로 적는다.
            if path and stops["ja"][a] in switchbacks.get(sb_key, ()):
                skip = 0
            else:
                skip = geometry_mod.unwind_retrace(path, arc) if path else 0
            # 두 구간이 역에서 안 만나면 역을 거쳐 잇는다. 곧장 이으면
            # 옆으로 튀어 지그재그가 된다. 도식적인 선형은 역까지 오지
            # 않는 일이 있다(사이쿄선 武蔵浦和 에서 121m 모자랐다).
            # 역이 두 끝 사이에 놓일 때만 끼운다. 옆으로 돌아가면 안 된다.
            if path and skip < len(arc):
                gap = _dist_m(path[-1], arc[skip])
                if gap > JOIN_VIA_STATION_M:
                    via = [float(coords[a][0]), float(coords[a][1])]
                    d1 = _dist_m(path[-1], via)
                    d2 = _dist_m(via, arc[skip])
                    # 역이 두 끝을 잇는 직선에 가까이 놓일 때만 거친다. 1.6 배까지
                    # 받았더니 선로에서 20m 비켜 적힌 ODPT 역 좌표를 거치며 Z 자로
                    # 꺾였다(常磐線 龍ケ崎市, 藤代).
                    if d1 + d2 < gap * VIA_DETOUR_MAX:
                        path.append(via)
            for q in arc[skip:]:
                pair = [float(q[0]), float(q[1])]
                if made_up and path and _dist_m(path[-1], pair) > SPLIT_GAP_M:
                    # 없는 구간을 현으로 때운 자리. 여기서 끊는다.
                    if len(path) >= 2:
                        pieces.append(path)
                        spans.append((start, end))
                    path = []
                    start = end = None
                if not path or path[-1] != pair:
                    path.append(pair)
            if start is None:
                start = a
            end = b
        if len(path) >= 2:
            pieces.append(path)
            spans.append((start, end))
        # 갔다가 되돌아왔다 다시 가는 자리를 걷어낸다. 원인은 자리마다
        # 다르지만 모양은 늘 같아서, 그리는 마지막에 한 번 훑는 것이
        # 제일 확실하다.
        keep_at = coords[[i for i in rows if np.isfinite(coords[i, 0])]]
        hold_at = coords[[i for i in rows if np.isfinite(coords[i, 0])
                          and stops["ja"][i] in switchbacks.get(sb_key, ())]]
        pieces = [geometry_mod.drop_tiny_zigzags(
                      geometry_mod.smooth_spikes(p, lat_scale, keep_at, hold_at),
                      lat_scale)
                  for p in pieces]
        if not pieces:
            # 제 선형이 아예 없는 노선(직통 계통이 남의 선로만 쓰는 경우)은
            # 예전처럼 구간마다 후보를 골라 잇는다.
            pieces, spans, path = [], [], []
            start = end = None
            for a, b in zip(rows, rows[1:]):
                got, src = geometry.segment_source(a, b)
                if got is None or len(got) < 2:
                    if len(path) >= 2:
                        pieces.append(path)
                        spans.append((start, end))
                    path = []
                    start = end = None
                    continue
                arc = np.asarray(got, dtype=np.float64)
                straight = _dist_m(coords[a], coords[b])
                # 선로를 따라 찾아 둔 구간은 길이로 거르지 않는다. 스위치백
                # (木次線 出雲坂根)이나 회차선처럼 실제로 크게 도는 선로가
                # 지도에서 통째로 끊겼다. 경로 쪽은 이미 이렇게 하고 있다.
                if (src != geometry_mod.TRACK_SRC
                        and _path_len_m(arc) > straight * geometry_mod.MAX_DETOUR_RATIO):
                    if len(path) >= 2:
                        pieces.append(path)
                        spans.append((start, end))
                    path = []
                    start = end = None
                    continue
                for q in arc:
                    pair = [float(q[0]), float(q[1])]
                    if not path or path[-1] != pair:
                        path.append(pair)
                if start is None:
                    start = a
                end = b
            if len(path) >= 2:
                pieces.append(path)
                spans.append((start, end))
        if not pieces:
            continue

        title = railway.get("title", {})
        seen = {prefectures.get(order[k]) for k in range(len(order))}
        base = {
            "id": rid,
            **{lang: title.get(lang, "") or title.get("ja", "")
               for lang in LANGS},
            # data/line-colors.json 이 가장 우선이다. 원본이 들고 있는
            # 색이 낡은 경우가 있어서다(総武本線 이 회색으로 나왔다).
            # 거기 없으면 원본(ODPT 는 color, OSM 은 colour)을 쓴다.
            # 원본 색도 #rrggbb 로 맞춰서 내보낸다. OSM 은 colour 를
            # "blue", "darkgreen" 처럼 이름으로 적어 두기도 하는데, 그대로
            # 보내면 지도에서 안 먹거나(white) 안 보인다.
            "color": line_color(railway),
            # 노선별 보기 패널이 운영사별로 묶는 데 쓴다. OSM 권역은 관계의
            # operator 태그가, 시각표 권역은 노선 id 앞머리가 회사다.
            "operator": _operator_label(railway),
            "prefs": sorted(p for p in seen if p),
        }
        for k, piece in enumerate(pieces):
            # 조각마다 id 를 달리 한다. 같은 id 를 달면 노선별 보기 패널에서
            # 조각 하나를 끕 때 그 노선의 조각이 전부 꺼진다.
            # build_rail.py · build_track.py 가 쓰는 "~k" 꼬리표를 따른다.
            row = dict(base, id=(rid if k == 0 else f"{rid}~{k}"),
                       path=_simplify(
                np.asarray(piece, dtype=np.float64),
                SHAPE_TOLERANCE_M).round(5).tolist())
            # 조각마다 어느 역에서 어느 역까지인지 적는다. 조각이 하나뿐인
            # 노선에도 적는다. 이름이 같은 노선을 한 줄로 묶을 때 그 구간이
            # 유일한 단서다. 없으면 "조각 1" 같은 알 수 없는 줄이 생긴다.
            got = spans[k] if k < len(spans) else None
            if got and got[0] is not None:
                def _nm(i, lang):
                    return stops[lang][i] or stops["ja"][i]
                row["span"] = {lang: _nm(got[0], lang) + "–" + _nm(got[1], lang)
                               for lang in LANGS if lang in stops}
            out.append(row)
    out.sort(key=lambda r: r["ja"])
    return out
