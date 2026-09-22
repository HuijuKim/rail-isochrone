"""OSM 에서 철도망을 뽑는다. 시각표가 없는 권역용이다.

간토는 공개 시각표가 있어 노선·역 목록을 거기서 받았다. 간사이 이후로는
그런 것이 없으므로 OSM 만으로 같은 모양의 자료를 만들어야 한다.

  노선     type=route 관계. 방향별·종별로 쪼개져 있어 한 노선으로 접는다.
  역       관계의 stop 멤버 노드. 없으면 선로 옆의 역 노드에서 찾는다.
  역 묶음   이름이 같고 가까운 역들. 환승이 되는 단위다.
  통과 계통  쾌속·급행·특급. 정차역 순서가 그대로 통과 패턴이 된다.

관계에는 두 세대가 섞여 있다. 새 것은 정차역을 노드 멤버로 달고 있지만
옛 것은 선로 웨이만 담는다. 京阪本線 이 후자라 처음에는 통째로 빠졌다.
선로만 있는 관계는 웨이를 이어 경로를 만들고, 그 옆 150m 안의 역 노드를
경로 위 거리 순으로 세워 정차 순서를 복원한다.

접는 기준은 이름이 아니라 정차역 겹침이다. 이름으로 접으면 같은 노선이
"JR京都線・JR宝塚線" 과 "JR宝塚線・JR京都線" 으로 갈라지고, 新快速 이
JR神戸線 의 통과 계통이 아니라 별개 노선이 되어 버린다.

신칸센은 뺀다. 간토 쪽 자료(ODPT)에 신칸센이 없어 권역끼리 어긋나기
때문이다. 구별되는 태그가 없으므로 계통 이름으로 거른다.

사용법: REGION=kansai python src/build_rail.py data/osm/kansai-latest.osm.pbf
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import osmium

ROOT = Path(__file__).resolve().parent.parent
REGION = os.environ.get("REGION", "kansai")
OUT = ROOT / "data" / "regions" / REGION / "raw"

LANGS = ("ja", "en", "ko", "zh-Hans", "zh-Hant")
RAIL_ROUTES = ("train", "railway", "subway", "light_rail", "monorail", "tram")
STATION_TAGS = ("station", "halt", "tram_stop")
# 사람을 태우지 않는 역의 usage 값. 도쿄 에는 JR동일본 승무원
# 훈련용 모의역(志茂田·大田, usage=training)이 railway=station 으로
# 들어 있어, 그대로 두면 京浜東北線 의 川崎-蒲田 사이에 끼어든다.
NOT_PASSENGER = ("tourism", "training", "military", "test", "industrial")
# 선로 웨이의 구성 노드에 붙는 정차 지점. 그 선로 위에 있는 것이
# 확실해서, 옆으로 나란히 달리는 노선의 역을 주워 담지 않는다.
STOP_POSITION = "stop_position"

# 열차 종별. 빠른 것부터 본다. 먼저 맞는 것을 종별로 삼는다.
# ライナー 는 넣지 않는다. ポートライナー・六甲ライナー 같은 신교통 노선
# 이름과 구별되지 않아 사람 나르는 궤도를 특급으로 만들어 버린다.
KINDS = [
    ("특급", r"特急|Limited Express"),
    ("통근쾌속", r"通勤快速|通勤特快|通勤急行|通勤準急|区間急行|区間快速"),
    ("쾌속", r"新快速|快速"),
    # 사업자 이름 안의 글자에 걸리면 안 된다. 北大阪急行電鉄南北線 이
    # 급행 계통으로 몰려 노선째 사라졌고(江坂 위쪽 桃山台·緑地公園·
    # 千里中央·箕面船場阪大前·箕面萱野 다섯 역), 京浜急行電鉄·富士急行線·
    # 伊豆急行線 도 같은 자리다. 뒤에 사업자·노선 글자가 오면 넘긴다.
    ("급행", r"急行(?!電鉄|鉄道|線)"),
    ("준급", r"準急"),
    ("각역정차", r"各駅停車|各停|普通|Local"),
]
KIND_RE = re.compile("|".join(p for _, p in KINDS))

# 애칭은 이름 한가운데서 걸리면 안 된다. "かもめ" 를 그냥 찾으면
# ゆりかもめ 가 신칸센으로 몰려 노선째 사라진다("とき" 는 ときわ 를
# 잡을 뻔했다). 앞뒤가 이름의 끝이거나 구분 기호일 때만 본다.
# "こまち列車" 처럼 뒤에 列車 가 붙는 표기가 많아 그것도 끝으로 친다.
_TRAIN_NAMES = (r"のぞみ|ひかり|こだま|みずほ|さくら|つばめ|はやぶさ|はやて|"
                r"こまち|やまびこ|なすの|とき|たにがわ|かがやき|はくたか|"
                r"つるぎ|あさま|かもめ")
SHINKANSEN = re.compile(
    r"新幹線|Shinkansen|"
    r"(?:^|[\s(（:：・=>＞、,])(?:" + _TRAIN_NAMES + r")(?:$|列車|[\s(（)）:：・=>＞、,])")

# 재래선 특급 애칭. 이름에 特急 이 안 적혀 종별 없는 노선으로 읽히면,
# 정차역이 많은 것부터 뼈대로 고르므로 진짜 노선을 부분 계통으로 밀어낸다.
# 踊り子(10역)가 뼈대가 되어 伊東線 이 熱海·宇佐美·伊東 를 잃고 3역만
# 남았다. OSM 에 종별 태그가 없어 이름으로 본다. 뒤에 구분 기호가 와야
# 한다. しなの鉄道線 은 しなの 가 아니다.
_LTD_EXPRESS_NAMES = (r"ひだ|しなの|南紀|ふじかわ|伊那路|踊り子|ミュースカイ|"
                      r"しらさぎ|サンダーバード|はまかぜ|こうのとり")
LTD_EXPRESS = re.compile(
    r"^(?:" + _LTD_EXPRESS_NAMES + r")(?:$|[\s(（:：・=>＞、,])")

PAREN_RE = re.compile(r"[（(][^）)]*[）)]")
DIR_RE = re.compile(r"(上り|下り|内回り|外回り|環状)")
ANGLE_RE = re.compile(r"[〈《<][^〉》>]*[〉》>]")

SAME_STATION_M = 400.0      # 이 안에 있고 이름이 같으면 한 역으로 본다
# 이름이 달라도 이만큼 붙어 있으면 한 환승역이다. 三ノ宮/神戸三宮, 大阪/
# 大阪梅田/梅田 이 그렇다. 안 합치면 같은 역이 여러 개로 세어지고 환승도
# 끊긴다. 간토에서 이 값으로 합쳤을 때 역 묶음이 1,858개로, 사람이 손수
# 묶은 ODPT 의 1,859개와 하나 차이였다.
CROSS_NAME_M = 300.0
# 이름 없는 정차 노드가 옆의 역 이름을 물려받는 거리
NAME_ADOPT_M = 300.0
TRACK_NEAR_M = 150.0        # 선로에서 이만큼 안이면 그 노선의 역으로 본다
SAME_LINE = 0.70            # 정차역이 이만큼 담기면 같은 선로의 다른 계통
DUP_LINE = 0.90             # 이만큼 같으면 중복 노선
# 이만큼 새 역을 가져오면 지선으로 보고 노선으로 세운다.
# 2 로 두었더니 고이즈미선 지선이 안 걸렸다. 竜舞 만 새 역이고 太田 는
# 이세사키선에 이미 있어서다. 1 로 해도 두 권역 합쳐 5개만 늘어난다.
MIN_BRANCH = 1
# 같은 노선으로 묶인 계통 끝에서 지선을 세울 때 역 사이 상한
BRANCH_GAP_M = 5000.0
MIN_WAYS = 5                # 선로에서 역을 찾아볼 최소 웨이 수
# 정차 순서에서 이 정도로 튀는 자리는 순서가 틀린 것으로 보고 다시 잇는다
# 5.0 으로 두었더니 남부선이 안 걸렸다. 가와사키-무카이가하라 5.3km 인데
# 보통 간격이 1.3km 라 기준이 6.5km 였다. 3.0 으로 낮추면 두 권역에서
# 노선 19개가 좋아지고 나빠지는 것은 하나도 없다(닛코선 25.5->9.3km,
# 사가미선 21.2->5.7km). 되이었을 때 가장 벌어진 자리가 실제로 좁아질
# 때만 바꾸므로 낮춰도 손해가 안 난다.
OUTLIER_MULT = 3.0          # 그 노선의 보통 간격의 몇 배부터
OUTLIER_M = 3000.0          # 그래도 이보다 가까우면 놔둔다


def _region_filter():
    """권역 밖의 역을 걸러내는 판정기.

    추출본은 권역보다 넓다. 간토는 야마나시·이즈 때문에 주부 추출본을
    함께 읽는데, 그대로 두면 나고야와 기후까지 딸려 온다.

    region.json 에 현 목록이 있고 build_admin.py 가 그 경계를 뽑아 두었으면
    현 경계로 자른다. 네모로 자르면 남부 니가타처럼 엉뚱한 곳이 딸려 온다.
    없으면 격자 네모로 자른다.
    """
    base = ROOT / "data" / "regions" / REGION
    rings_path = base / "raw" / "prefecture-rings.json"
    meta_path = base / "region.json"
    prefs = []
    if meta_path.exists():
        prefs = json.loads(meta_path.read_text(encoding="utf-8")).get("prefectures") or []
    if prefs and rings_path.exists():
        from matplotlib.path import Path as MplPath
        data = json.loads(rings_path.read_text(encoding="utf-8"))
        paths, boxes = [], []
        for name in prefs:
            for ring in data.get(name, []):
                a = np.asarray(ring, dtype=np.float64)
                if len(a) >= 4:
                    paths.append(MplPath(a))
                    boxes.append((a[:, 0].min(), a[:, 0].max(),
                                  a[:, 1].min(), a[:, 1].max()))
        if paths:
            boxes = np.array(boxes)

            def inside(lon, lat):
                near = np.flatnonzero((boxes[:, 0] <= lon) & (lon <= boxes[:, 1])
                                      & (boxes[:, 2] <= lat) & (lat <= boxes[:, 3]))
                return any(paths[i].contains_point((lon, lat)) for i in near)

            return inside, f"현 {len(prefs)}개 경계"

    box = _bbox()
    if box is None:
        return None, ""
    lon0, lat0, lon1, lat1 = box

    def inside_box(lon, lat):
        return lon0 <= lon <= lon1 and lat0 <= lat <= lat1

    return inside_box, (f"격자 {lon0:.2f}~{lon1:.2f}E, {lat0:.2f}~{lat1:.2f}N")


def _bbox():
    """권역 격자가 덮는 범위."""
    meta = ROOT / "data" / "regions" / REGION / "region.json"
    if not meta.exists():
        return None
    grid = json.loads(meta.read_text(encoding="utf-8")).get("grid")
    if not grid:
        return None
    lon0, lat0 = float(grid["lon0"]), float(grid["lat0"])
    m_lon = 111_320.0 * np.cos(np.radians(float(grid["lat_ref"])))
    return (lon0, lat0,
            lon0 + float(grid["span_x"]) / m_lon,
            lat0 + float(grid["span_y"]) / 111_132.0)


def _pbf_list():
    """어느 OSM 추출본을 읽을지. 권역 설정에 적고, 인자로 덮을 수 있다."""
    if len(sys.argv) > 1:
        return [Path(a) for a in sys.argv[1:]]
    meta = ROOT / "data" / "regions" / REGION / "region.json"
    if meta.exists():
        listed = json.loads(meta.read_text(encoding="utf-8")).get("osm_files")
        if listed:
            return [ROOT / "data" / "osm" / n for n in listed]
    return [ROOT / "data" / "osm" / (REGION + "-latest.osm.pbf")]


def kind_of(name):
    if LTD_EXPRESS.match(name.strip()):
        return "특급"
    for kind, pat in KINDS:
        if re.search(pat, name):
            return kind
    return None


def base_name(name):
    """방향과 종별을 떼어 노선 이름만 남긴다."""
    s = PAREN_RE.sub("", name)
    s = ANGLE_RE.sub("", s)
    s = DIR_RE.sub("", s)
    s = KIND_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ・･-–—>=→:：")


# 노선으로 잡히지만 타고 다닐 수 없는 것들. 화물 전용선, 폐선, 관광
# 삭도다. OSM 태그로는 가릴 수 없다. 神奈川臨海鉄道本牧線 은 화물
# 전용인데 route=train 으로 달려 있고, 여객선인 鹿島臨海鉄道大洗鹿島線 과
# 태그가 같다. 그래서 이름을 data/excluded-lines.json 에 적어 두고 본다.
_EXCLUDE: set[str] | None = None


def _excluded(name: str) -> bool:
    global _EXCLUDE
    if _EXCLUDE is None:
        path = ROOT / "data" / "excluded-lines.json"
        names: set[str] = set()
        try:
            for key, val in json.loads(path.read_text(encoding="utf-8")).items():
                if isinstance(val, list):
                    names.update(val)
        except (OSError, ValueError):
            pass
        _EXCLUDE = names
    base = PAREN_RE.sub("", name).strip()
    return name in _EXCLUDE or base in _EXCLUDE or any(
        name.startswith(k) for k in _EXCLUDE)


class Relations(osmium.SimpleHandler):
    """철도 계통 관계를 모은다. 정차역이 없으면 선로 웨이를 챙겨 둔다."""

    def __init__(self):
        super().__init__()
        self.routes = []
        self.want_nodes = set()
        self.want_ways = set()
        self.dropped = 0
        self.skipped = 0
        # OSM id 는 추출본 사이에서 전역이다. 권역이 두 추출본에 걸치면
        # 경계의 관계가 양쪽에 다 들어 있어 그대로 읽으면 두 번 잡힌다.
        self.seen = set()

    def relation(self, r):
        if r.id in self.seen:
            return
        self.seen.add(r.id)
        t = r.tags
        if t.get("type") != "route":
            return
        if (t.get("route") or "") not in RAIL_ROUTES:
            return
        name = t.get("name") or t.get("name:ja") or ""
        if not name:
            return
        if SHINKANSEN.search(name) or SHINKANSEN.search(t.get("ref", "")):
            self.dropped += 1
            return
        # 유원지 어트랙션은 태그가 일반 노선과 같다. 디즈니랜드의
        # ウエスタンリバー鉄道 는 route=train, operator=オリエンタルランド
        # 이고 실제로 762mm 증기기관차가 다니는 진짜 철도다. 다만
        # tourism=attraction 이 붙어 있어 그걸로 가른다. 같은 리조트의
        # ディズニーリゾートライン 은 요금 받는 정식 모노레일이라 이
        # 태그가 없고, 그대로 남는다.
        if t.get("tourism") == "attraction" or _excluded(name):
            self.skipped += 1
            return
        stops = [m.ref for m in r.members
                 if m.type == "n" and m.role in ("stop", "stop_entry_only",
                                                 "stop_exit_only")]
        if not stops:
            stops = [m.ref for m in r.members if m.type == "n" and m.role == "platform"]
        ways = [m.ref for m in r.members if m.type == "w"]
        if len(stops) < 2 and len(ways) < MIN_WAYS:
            return
        self.want_nodes.update(stops)
        # 선로 기하는 두 곳에 쓴다. 정차역이 없는 관계에서 정차 순서를
        # 되살릴 때와, 지도에 노선을 그릴 때다. 그래서 전부 챙긴다.
        self.want_ways.update(ways)
        self.routes.append({
            "name": name,
            "rtype": t.get("route") or "",
            "titles": {g: t.get("name:" + g, "") for g in LANGS},
            "operator": t.get("operator", "") or t.get("network", ""),
            "ref": t.get("ref", ""),
            "colour": t.get("colour", ""),
            "kind": kind_of(name),
            "stops": stops,
            "ways": ways,
        })


class Ways(osmium.SimpleHandler):
    """선로 웨이의 좌표열과 구성 노드 번호."""

    def __init__(self, want):
        super().__init__()
        self.want = want
        self.geom = {}
        self.refs = {}

    def way(self, w):
        if w.id not in self.want or w.id in self.geom:
            return
        self.refs[w.id] = [n.ref for n in w.nodes]
        try:
            self.geom[w.id] = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        except osmium.InvalidLocationError:
            # 비워 두지 말고 아예 남기지 않는다. 키가 생기면 위의 가드에
            # 걸려, 첫 추출본에서 좌표가 모자랐던 웨이를 두 번째 추출본이
            # 다시 채울 기회를 잃는다. 추출본 경계가 바로 그런 자리다.
            pass


# 승강장 노드에 붙는 꼬리. 이걸 떼지 않으면 北千住01 과 北千住 가,
# 柘植駅1番のりば 와 柘植 가 서로 다른 역이 된다. 간토에서 84개,
# 간사이에서 107개가 이렇게 갈라져 있었다.
_TAIL = re.compile(r"(?:\s*\d+\s*番(?:のりば|線)|\s*のりば|\s*ホーム)$")
_NUM = re.compile(r"[\s._-]*\d+(?:[.\-_]\d+)*$")


def clean_station_name(nm: str) -> str:
    """승강장·출입구 표기를 떼어 역 이름 하나로 만든다."""
    nm = (nm or "").strip()
    for br in ("(", "（"):
        if br in nm:
            nm = nm.split(br)[0].strip()
    nm = _TAIL.sub("", nm).strip()
    nm = _NUM.sub("", nm).strip()
    if nm.endswith("駅") and len(nm) > 1:
        nm = nm[:-1]
    return nm


class StationNodes(osmium.SimpleHandler):
    """역 노드 전부와, 관계가 가리킨 정차 노드."""

    def __init__(self, want, inside=None):
        super().__init__()
        self.want = want
        self.inside = inside
        self.pos = {}
        self.stations = {}
        self.rail = set()

    def node(self, n):
        if n.id in self.pos:
            return
        t = n.tags
        is_station = (t.get("railway") in STATION_TAGS
                      or t.get("public_transport") in ("station", STOP_POSITION))
        if not is_station and n.id not in self.want:
            return
        if self.inside is not None and not self.inside(n.location.lon,
                                                       n.location.lat):
            return
        rec = (n.location.lon, n.location.lat,
               {g: clean_station_name(t.get("name:" + g, "")) for g in LANGS},
               clean_station_name(t.get("name", "")))
        self.pos[n.id] = rec
        if is_station and (rec[2].get("ja") or rec[3]):
            self.stations[n.id] = rec
            # 빠진 역을 끼워 넣을 때 쓸 후보. 태그로 가른다. 버스터미널
            # (public_transport=station) 과 선로 위 정차 노드를 넣으면 같은
            # 역이 두 번 들어간다. 노면전차 정류장은 제 노선에서 잡히고,
            # 관광용 미니열차·공원 모노레일은 본선 옆 120m 안에 있어 그냥
            # 두면 쇼난신주쿠라인에 鉄道博物館 과 飛鳥山공원 역이 끼었다.
            if (t.get("railway") in ("station", "halt")
                    and t.get("usage") not in NOT_PASSENGER
                    and t.get("monorail") != "yes"
                    and not (rec[3] or "").endswith("信号場")):
                self.rail.add(n.id)


def stitch(ways, geom):
    """가장 긴 사슬 하나. 정차 순서를 되살릴 때 쓴다."""
    chains = stitch_all(ways, geom)
    return chains[0] if chains else []


def stitch_all(ways, geom):
    """웨이를 끝점끼리 이어 사슬들로.

    멤버 순서만 믿고 이으면 순서가 흐트러진 관계에서 경로가 지그재그가
    되고, 그 자리가 지도에서 선로를 벗어난 직선으로 보인다. 끝점을
    맞대어 사슬을 만들고 가장 긴 사슬을 쓴다. 갈라지는 지선은 버린다.
    """
    segs = [(wid, geom[wid]) for wid in ways
            if len(geom.get(wid, ())) >= 2]
    if not segs:
        return []

    def key(pt):
        return (round(pt[0], 7), round(pt[1], 7))

    at = defaultdict(list)
    for k, (_wid, pts) in enumerate(segs):
        at[key(pts[0])].append(k)
        at[key(pts[-1])].append(k)

    used = set()

    def walk(start_k, start_end):
        """한 웨이에서 시작해 끝점을 따라 갈 수 있는 데까지 간다.

        start_end 는 자유로운 쪽 끝이다. 사슬은 거기서 시작해 반대쪽으로
        자라야 하므로, 자유로운 끝이 뒤에 있을 때만 뒤집는다.
        """
        pts = list(segs[start_k][1])
        if start_end == 1:
            pts.reverse()
        used.add(start_k)
        while True:
            tip = key(pts[-1])
            nxt = next((j for j in at.get(tip, ()) if j not in used), None)
            if nxt is None:
                break
            used.add(nxt)
            q = list(segs[nxt][1])
            if key(q[-1]) == tip:
                q.reverse()
            pts.extend(q[1:])
        return pts

    chains = []
    # 끝이 하나뿐인 웨이(노선의 끝)에서 시작해야 중간부터 뻗어 나가지 않는다
    for k, (_wid, pts) in enumerate(segs):
        if k in used:
            continue
        for end, pt in ((0, pts[0]), (1, pts[-1])):
            if len(at.get(key(pt), ())) == 1:
                chains.append(walk(k, end))
                break
    for k in range(len(segs)):
        if k not in used:
            chains.append(walk(k, 1))
    chains.sort(key=len, reverse=True)
    return chains


def join_runs(runs, stations, scale):
    """토막난 정차 순서를 끝 역끼리 가까운 쪽으로 이어 붙인다.

    노드 번호로 이은 사슬은 관계에 빠진 웨이가 있으면 거기서 끊긴다.
    아가쓰마선은 시부카와 구내 선로가 조에쓰선 것이라 관계에 없어,
    시부카와만 따로 떨어진 토막이 된다. 긴 것부터 그냥 붙이면 그 역이
    맨 뒤로 가 43km 짜리 직선이 그어진다. 료모선은 아예 뒤집힌 토막이
    붙어 마에바시에서 이세사키로 거슬러 올라갔다.

    긴 토막에서 시작해, 남은 토막 중 끝 역이 가장 가까운 것을 방향까지
    맞춰 앞이나 뒤에 붙인다.
    """
    runs = [r for r in runs if r]
    if len(runs) <= 1:
        return list(runs[0]) if runs else []
    runs.sort(key=len, reverse=True)

    def gap(a, b):
        pa, pb = stations[a], stations[b]
        return float(np.hypot((pb[0] - pa[0]) * scale * 111_320.0,
                              (pb[1] - pa[1]) * 111_132.0))

    out = list(runs[0])
    rest = runs[1:]
    seen = set(out)
    while rest:
        best = None
        for i, r in enumerate(rest):
            for tail, a in ((1, out[-1]), (0, out[0])):
                for far, b in ((0, r[0]), (1, r[-1])):
                    d = gap(a, b)
                    if best is None or d < best[0]:
                        best = (d, i, tail, far)
        _d, i, tail, far = best
        r = rest.pop(i)
        if tail == 1:
            if far == 1:
                r = r[::-1]
            add = [n for n in r if n not in seen]
            seen.update(add)
            out.extend(add)
        else:
            if far == 0:
                r = r[::-1]
            add = [n for n in r if n not in seen]
            seen.update(add)
            out[:0] = add
    return out


def _max_gap_m(seq, cpos, scale):
    """이웃한 두 역 사이가 가장 벌어진 거리."""
    P = np.array([cpos[c] for c in seq], dtype=np.float64)
    return float(np.hypot(np.diff(P[:, 0]) * scale * 111_320.0,
                          np.diff(P[:, 1]) * 111_132.0).max())


def _path_m(seq, cpos, scale):
    """역을 순서대로 이은 길이(m)."""
    if len(seq) < 2:
        return 0.0
    P = np.array([cpos[c] for c in seq], dtype=np.float64)
    return float(np.hypot(np.diff(P[:, 0]) * scale * 111_320.0,
                          np.diff(P[:, 1]) * 111_132.0).sum())


def _follows_track(seq, cpos, way_ids, geom, scale):
    """역들이 가장 긴 선로 사슬 위에 한 방향 차례로 놓여 있는가.

    역의 90% 이상이 그 사슬 200m 안에 있고, 사슬 위 위치가 목록 순서대로
    늘거나 줄기만 해야 한다. 선로가 잘게 쪼개져 가장 긴 사슬이 역을 다
    못 덮으면 판단하지 않는다(False).
    """
    if len(seq) < 3 or any(c not in cpos for c in seq):
        return False
    chains = stitch_all(way_ids, geom)
    if not chains:
        return False
    C = np.asarray(chains[0], dtype=np.float64)
    cx, cy = C[:, 0] * scale * 111_320.0, C[:, 1] * 111_132.0
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
    at = []
    for c in seq:
        d = np.hypot(cx - cpos[c][0] * scale * 111_320.0,
                     cy - cpos[c][1] * 111_132.0)
        k = int(np.argmin(d))
        if d[k] <= FILL_NEAR_M:
            at.append(float(arc[k]))
    if len(at) < 0.9 * len(seq) or len(at) < 3:
        return False
    d = np.diff(at)
    return bool((d >= 0).all() or (d <= 0).all())


def repair_order(seq, cpos, scale):
    """정차 순서에서 튀는 자리를 찾아 토막을 내고 다시 잇는다.

    관계에 적힌 정차 순서가 늘 옳지는 않다. 조반선은 도쿄 구간과
    이바라키 구간이 61km 씩 떨어진 채 세 토막으로 엇갈려 있었고,
    한조몬선 직통 계통은 주오린칸 다음이 가스카베(59km)였다. 그대로
    두면 지도에 그 길이만큼 직선이 그어진다.

    다만 특급은 원래 역 사이가 멀다. 슈퍼하코네는 신주쿠-오다와라가
    68km 인데 그게 맞다. 그래서 절대 거리가 아니라 그 노선의 보통
    간격에 견주어 튀는 자리만 자른다. 자른 토막은 끝 역이 가까운
    쪽끼리 방향까지 맞춰 다시 붙이고, 그래서 가장 벌어진 자리가 실제로
    좁아질 때만 바꾼다. 순서가 이미 옳으면 아무 일도 없다.
    """
    if any(c not in cpos for c in seq):
        return seq
    # 상행과 하행이 한 줄에 섞여 들어오는 관계가 많다. 오다큐 오다와라선은
    # 96개 중 서로 다른 역이 47개였고, 그 이음매에서 신주쿠와 이세하라가
    # 붙어 48km 짜리 직선이 그어졌다. 한 방향으로 펴고 본다.
    #
    # 펴는 것은 길이와 상관없이 먼저 한다. 짧다고 그냥 돌려보내면
    # 難波線(5개 중 3개), 関西空港線, 日生線 같은 짧은 노선이
    # 왜복 중복을 그대로 달고 나간다.
    # 순환선은 끝에 첫 역을 한 번 더 적어 고리를 닫는다. 그건 중복이
    # 아니다. 지우면 山手線 고리가 한 구간 끊긴다.
    if len(seq) >= 4 and seq[0] == seq[-1] and len(set(seq)) == len(seq) - 1:
        return seq
    plain, seen = [], set()
    for c in seq:
        if c not in seen:
            seen.add(c)
            plain.append(c)
    # 처음 나온 순서로만 지우면 돌아오는 반에만 있는 역이 반환점 뒤에
    # 거꾸로 붙는다. 近鉄名古屋線 은 하행에 白子 남쪽 역이 빠져 있어
    # 豊津上野-霞ヶ浦 가 伊勢中川 뒤로 갔다. 가는 반을 뼈대로 두고 나머지를
    # 가장 덜 돌아가는 자리에 끼운다.
    turn = next((i for i, c in enumerate(seq) if c in seq[:i]), None)
    if turn is not None:
        head = []
        for c in seq[:turn]:
            if c not in head:
                head.append(c)
        rest = [c for c in plain if c not in head]
        if len(head) >= 2 and rest:

            def xy(c):
                return (cpos[c][0] * scale * 111_320.0, cpos[c][1] * 111_132.0)
            for c in rest:
                _insert_cheapest(head, c, xy)
            # 가는 반이 듬성하면 끼우는 차례에 따라 지그재그가 된다
            # (名鉄名古屋本線). 떠돌이 역이 앞에 붙은 것은 이웃과 가장 잘
            # 맞는 자리만 남겨야 풀린다(紀勢本線). 셋 중 순서대로 이은
            # 길이가 가장 짧은 것을 쓴다.
            best = _drop_strays(seq, xy, lambda c: c)
            plain = min((plain, head, best),
                        key=lambda q: _path_m(q, cpos, scale))
    if len(plain) < 4:
        return plain if len(plain) < len(seq) else seq

    P = np.array([cpos[c] for c in plain], dtype=np.float64)
    d = np.hypot(np.diff(P[:, 0]) * scale * 111_320.0,
                 np.diff(P[:, 1]) * 111_132.0)
    med = float(np.median(d))
    cut = np.flatnonzero(d > max(med * OUTLIER_MULT, OUTLIER_M))
    if not len(cut):
        # 자를 데가 없어도 한 방향으로 펐 것은 살린다. 예전에는
        # 그냥 seq 를 돌려줘서 갔다 오는 관계가 역을 두 번씩 단 채로
        # 나갔다. 그러면 지도에서 선이 갔다가 그대로 되돌아오고(大宮·
        # 練馬·春日部·鶴橋·難波 의 Z 자 꺾임), 나이브 시각표는 한
        # 노선에 열차를 두 배로 깔고, 구간 선형도 두 번씩 저장된다.
        return plain if len(plain) < len(seq) else seq

    blocks, prev = [], 0
    for k in cut:
        blocks.append(plain[prev:k + 1])
        prev = k + 1
    blocks.append(plain[prev:])
    fixed = join_runs(blocks, cpos, scale)
    if set(fixed) != set(plain):
        return plain if len(plain) < len(seq) else seq
    # 견줄 대상은 한 방향으로 펀 쪽이다. 왜복으로 적힌 목록은
    # 이웃이 전부 실제 이웃이라 최대 간격이 작게 나온다. 그걸
    # 잣대로 쓰면 올바로 고친 쪽이 진다.
    return (fixed if _max_gap_m(fixed, cpos, scale)
            < _max_gap_m(plain, cpos, scale) else plain)


def reseat_strays(seq, cpos, scale, rounds: int = 40):
    """역 하나를 빼서 제일 나은 자리에 다시 넣기를 되풀이한다.

    OSM 관계의 정차 순서가 통째로 뒤섞여 있는 일이 있다. 成田線
    (成田空港→千葉) 은 千葉 이 가운데 끼어 11 km 를 왕복하고, 四街道 와
    都賀 가 뒤집혀 있고, 成田 이 맨 끝에 가 있었다. 常磐線 은 임시역
    偕楽園 이 45개 중 맨 끝에 붙어 大津港 에서 60 km 가 됐다.

    노선은 길이라, 역을 순서대로 이은 길이가 짧을수록 옳은 순서다.
    한 역을 빼서 가장 덜 돌아가는 자리에 다시 넣고, 전체 길이가
    눈에 띄게 줄 때만 받는다. 줄지 않으면 그대로 둔다. 버리지는
    않는다. 자리만 틀렸을 뿐 진짜 역이다.

    순환선은 첫 역과 끝 역이 같아 길이로 재면 안 되므로 건드리지 않는다.
    """
    if len(seq) < 5 or seq[0] == seq[-1]:
        return seq
    if any(c not in cpos for c in seq):
        return seq

    P = np.array([cpos[c] for c in seq], dtype=np.float64)
    P[:, 0] *= scale * 111.320
    P[:, 1] *= 111.132


    def total(idx):
        Q = P[idx]
        return float(np.hypot(*(np.diff(Q, axis=0).T)).sum())

    order = list(range(len(seq)))
    base = total(order)
    # 이만큼은 줄어야 자리를 바꾼다. 재고 또 재는 것을 막는다.
    floor = max(base * 0.01, 0.3)

    for _ in range(rounds):
        best = None
        for i in range(len(order)):
            rest = order[:i] + order[i + 1:]
            R = P[rest]
            x = P[order[i]]
            # 끼워 넣을 자리마다 늘어나는 길이. 양 끝은 이웃이 하나다.
            leg = np.hypot(*(np.diff(R, axis=0).T))
            to = np.hypot(R[:, 0] - x[0], R[:, 1] - x[1])
            inner = to[:-1] + to[1:] - leg          # 사이에 끼울 때
            add = np.concatenate([[to[0]], inner, [to[-1]]])
            k = int(np.argmin(add))
            gain = base - (total(rest) + float(add[k]))
            if gain > floor and (best is None or gain > best[0]):
                best = (gain, i, k, rest)
        if best is None:
            break
        _gain, i, k, rest = best
        order = rest[:k] + [order[i]] + rest[k:]
        base = total(order)

    return [seq[i] for i in order] if order != list(range(len(seq))) else seq


def stops_on_ways(ways, refs, stations, scale):
    """선로 웨이의 구성 노드에서 정차 지점을 순서대로 뽑는다.

    멤버 순서를 그대로 믿으면 안 된다. 손이 많이 간 관계는 웨이가
    뒤섞여 들어 있어, 순서대로 읽으면 역 차례가 뒤죽박죽이 된다.
    도부 도조선은 나리마스가 세 번, 모리노미야가 두 번 나왔고, 그
    자리마다 지도에 30km 짜리 직선이 그어졌다. 노드 번호로 끝을
    맞대어 사슬을 만든 뒤 사슬을 따라 읽는다.

    선로 옆 거리로 줍는 방식과 달리 나란히 달리는 노선의 역이 섞이지
    않는다. 야마노테선과 선로를 나눠 쓰는 쇼난신주쿠라인이 原宿·代々木
    까지 서는 것으로 잘못 잡히던 문제가 여기서 갈린다.
    """
    segs = [refs[w] for w in ways if refs.get(w) and len(refs[w]) >= 2]
    if not segs:
        return []

    at = defaultdict(list)
    for k, ns in enumerate(segs):
        at[ns[0]].append(k)
        at[ns[-1]].append(k)
    used = set()

    def walk(start, end):
        ns = list(segs[start])
        if end == 1:
            ns.reverse()
        used.add(start)
        while True:
            nxt = next((j for j in at.get(ns[-1], ()) if j not in used), None)
            if nxt is None:
                return ns
            used.add(nxt)
            q = list(segs[nxt])
            if q[-1] == ns[-1]:
                q.reverse()
            ns.extend(q[1:])

    chains = []
    # 끝이 하나뿐인 웨이(노선의 끝)에서 시작해야 중간부터 뻗어 나가지 않는다
    for k, ns in enumerate(segs):
        if k in used:
            continue
        for end, n in ((0, ns[0]), (1, ns[-1])):
            if len(at.get(n, ())) == 1:
                chains.append(walk(k, end))
                break
    for k in range(len(segs)):
        if k not in used:
            chains.append(walk(k, 1))
    chains.sort(key=len, reverse=True)

    runs = []
    for ns in chains:
        run = []
        for n in ns:
            if n in stations and (not run or run[-1] != n):
                run.append(n)
        if run:
            runs.append(run)
    return join_runs(runs, stations, scale)


# 제 선로 사슬에서 이만큼 안에 있는 역까지 줍는다. 플랫폼 노드가 선로
# 중심에서 떨어져 있어 너무 좁히면 큰 역을 놓친다. 120m 로 두었더니
# 山陰本線 의 石原 이 153m 라 빠졌다. 나란히 달리는 남의 선로 역은
# 가장 가까운 선로가 제 선로인지로 거른다(fill_missing 의 owner).
FILL_NEAR_M = 200.0
# 이만큼 안에 이미 정차역이 있으면 그 역이다. 이름 없는 정차 노드는
# 이름으로 가릴 수 없어 자리로 가린다. 新川崎 와 鹿島田 이 300m 라
# 그보다 넉넉히 좁게 잡는다.
FILL_SAME_M = 200.0
# 가장 가까운 선로가 남의 것인 역을 노선 끝에 붙일 때 제 선로와의 거리
# 상한. 分岐駅 多気 가 参宮線 선로에서 38m 다.
FILL_END_M = 60.0
# 같은 역의 다른 자리가 이만큼 더 어긋나면 떠돌이로 보고 버린다.
# 紀勢本線 앞머리의 那智 는 150km 어긋났다.
STRAY_SLACK_M = 5000.0


def _name_key(rec):
    """같은 역을 가리키는 다른 이름을 하나로. 浜坂 와 浜坂駅 이 그렇다."""
    nm = (rec[2].get("ja") or rec[3] or "").strip()
    for br in ("(", "（"):
        if br in nm:
            nm = nm.split(br)[0].strip()
    return nm[:-1] if nm.endswith("駅") else nm


def _at_spot(rec, spots, scale):
    """이미 있는 정차역 자리인가."""
    if not spots:
        return False
    P = np.asarray(spots, dtype=np.float64)
    d = np.hypot((P[:, 0] - rec[0]) * scale * 111_320.0,
                 (P[:, 1] - rec[1]) * 111_132.0)
    return bool(d.min() < FILL_SAME_M)


def _insert_cheapest(out, item, xy, ends_only=False):
    """노선 길이가 가장 적게 늘어나는 자리에 item 을 끼운다.

    xy(원소) 는 미터 좌표나 None. 끼웠으면 True. 노선은 길이라 역을
    순서대로 이은 길이가 짧을수록 옳은 순서다(reseat_strays 와 같은 뜻).
    """
    p = xy(item)
    if p is None:
        return False
    p = np.asarray(p, dtype=np.float64)
    best, best_k = None, None
    for k in range(len(out) + 1):
        a = xy(out[k - 1]) if k > 0 else None
        b = xy(out[k]) if k < len(out) else None
        if a is None and b is None:
            continue
        if a is None:
            cost = float(np.hypot(*(p - b)))
        elif b is None:
            cost = float(np.hypot(*(p - a)))
        else:
            a, b = np.asarray(a), np.asarray(b)
            cost = float(np.hypot(*(p - a)) + np.hypot(*(p - b))
                         - np.hypot(*(a - b)))
        if best is None or cost < best:
            best, best_k = cost, k
    if best_k is None:
        return False
    if ends_only and 0 < best_k < len(out):
        return False
    out.insert(best_k, item)
    return True


def _drop_strays(seq, xy, key, slack_m=None):
    """같은 역이 여러 번 나오면 이웃과 가장 잘 맞는 자리 하나만 남긴다.

    선로 관계를 웨이 조각으로 이으면 먼 조각이 맨 앞에 붙어 떠돌이 역이
    생긴다. 紀勢本線 은 "下里 那智 和歌山市 …" 로 시작하고 下里·那智 가
    제자리에 또 나온다. 처음 나온 것을 남기면 떠돌이가 살고, 그 옆에
    역을 끼우면 떠돌이 덩어리가 커져 노선이 331km 에서 427km 가 됐다.
    """
    at = defaultdict(list)
    for i, n in enumerate(seq):
        at[key(n)].append(i)

    def cost(i):
        p = xy(seq[i])
        if p is None:
            return 0.0
        p = np.asarray(p, dtype=np.float64)
        a = xy(seq[i - 1]) if i > 0 else None
        b = xy(seq[i + 1]) if i + 1 < len(seq) else None
        if a is None and b is None:
            return 0.0
        if a is None:
            return float(np.hypot(*(p - b)))
        if b is None:
            return float(np.hypot(*(p - a)))
        a, b = np.asarray(a), np.asarray(b)
        return float(np.hypot(*(p - a)) + np.hypot(*(p - b)) - np.hypot(*(a - b)))

    # slack_m 을 주면 가장 잘 맞는 자리보다 그만큼 넘게 어긋나는 자리만
    # 버린다. 복선의 상행·하행이 한 목록에 섞여 온 것은 두 자리가 다
    # 맞으므로 둘 다 남는다. 하나만 남기면 두 반의 자리가 뒤섞여
    # 東海道本線 이 310km 에서 575km 가 됐다. 왕복은 repair_order 가 편다.
    drop = set()
    for k, idx in at.items():
        if k and len(idx) > 1:
            c = {i: cost(i) for i in idx}
            keep = min(idx, key=c.get)
            if slack_m is None:
                drop.update(i for i in idx if i != keep)
            else:
                drop.update(i for i in idx if c[i] > c[keep] + slack_m)
    return [n for i, n in enumerate(seq) if i not in drop]


def _nearest_track(geom, scale, pos, slack_m=15.0):
    """역 -> 그 역에서 가장 가까운 선로 웨이들. 가장 가까운 것보다
    slack_m 안쪽에 있는 것까지 준다. 선로를 공유하는 노선이 함께 걸린다."""
    from scipy.spatial import cKDTree

    wid, xy = [], []
    for w, g in geom.items():
        a = np.asarray(g, dtype=np.float64).reshape(-1, 2)
        wid.extend([w] * len(a))
        xy.append(np.c_[a[:, 0] * scale * 111_320.0, a[:, 1] * 111_132.0])
    if not xy:
        return lambda n: set()
    wid = np.asarray(wid)
    tree = cKDTree(np.concatenate(xy))
    memo = {}

    def owner(n):
        if n not in memo:
            p = (pos[n][0] * scale * 111_320.0, pos[n][1] * 111_132.0)
            d, _i = tree.query(p)
            memo[n] = set(wid[tree.query_ball_point(p, d + slack_m)].tolist())
        return memo[n]
    return owner


def fill_missing(seq, way_ids, geom, stations, pool, scale, owner=None):
    """정차 목록에 빠진 역을, 노선 길이가 가장 적게 늘어나는 자리에 끼운다.

    OSM 이 어떤 역에는 선로 위 정차 노드를 두지 않고 옆에 역 노드만
    둔다. 그러면 웨이 구성 노드만 보는 방식으로는 그 역이 통째로
    빠진다. 간사이 산인 본선의 玄武洞·養父·国府·佐津·柴山 이 그렇다.

    pool 은 후보가 될 노드다. 아무 계통도 정차역으로 부르지 않은 역
    노드만 들어 있다. 이 거름이 없으면 나란히 달리는 남의 노선 역이
    딸려 온다. 산인 본선 옆 사가노 관광철도의 トロッコ嵯峨 와 京福 의
    撮影所前 이 120m 안에 있다. owner(역) 가 주어지면 그 역에서 가장
    가까운 선로 웨이들을 돌려받아, 그중에 제 웨이가 없으면 버린다.
    어느 계통도 안 부른 남의 역이 제 선로 200m 안에 있을 때 거른다.
    JR 長島 가 近鉄長島 옆이라 近鉄名古屋線 에 딸려 왔다.

    예전에는 선로를 끝점끼리 이은 사슬 위에서 아는 역 사이에 끼웠다.
    역 구내 측선 때문에 사슬이 잘게 쪼개지면(関西本線 은 191개) 아는
    역이 없는 사슬의 역은 버려지고(新堂·佐那具·伊賀上野·島ヶ原), 방향을
    잘못 잡은 사슬은 한 무리를 노선 끝 뒤로 거꾸로 붙였다(近鉄名古屋線
    의 長島-豊津上野 가 伊勢中川 뒤로 갔다). 노선은 길이라 역을 순서대로
    이은 길이가 가장 적게 늘어나는 자리가 제자리다. 아는 역에 가까운
    후보부터 끼운다.
    """
    from scipy.spatial import cKDTree

    def xy(n):
        return (stations[n][0] * scale * 111_320.0, stations[n][1] * 111_132.0)

    pts = [np.asarray(geom[w], dtype=np.float64).reshape(-1, 2)
           for w in way_ids if len(geom.get(w, ())) >= 1]
    known = [n for n in seq if n in stations]
    if not pts or not known:
        return list(seq)
    P = np.concatenate(pts)
    tree = cKDTree(np.c_[P[:, 0] * scale * 111_320.0, P[:, 1] * 111_132.0])
    own = set(way_ids)

    have = set(seq)
    # 이름 없는 정차 노드의 빈 이름이 들어가면 안 된다. 그러면 이름
    # 없는 후보가 통째로 막힌다. 빈 이름은 자리로만 가린다.
    taken = {k for k in (_name_key(stations[n]) for n in known) if k}
    spots = [(stations[n][0], stations[n][1]) for n in known]

    K = np.array([xy(n) for n in known])
    cands = []
    for nid in pool:
        # 이름 없는 역 노드는 끼워도 목록에 빈 칸으로 설 뿐이다
        if nid in have or nid not in stations or not _name_key(stations[nid]):
            continue
        p = xy(nid)
        d_own = tree.query(p)[0]
        if d_own > FILL_NEAR_M:
            continue
        # 가장 가까운 선로가 남의 것이면 한가운데로는 안 받는다. 다만 제
        # 선로 가까이 있는 노선 끝은 받는다. 分岐駅 이 그렇다. 参宮線 은
        # 多気 에서 紀勢本線 과 갈라져, 多気 에 가장 가까운 선로가 紀勢本線
        # 이다(제 선로 38m). 東大垣(35m 차이)·長島 같은 남의 역은 노선
        # 한가운데로 들어오려 해서 걸린다.
        ends_only = owner is not None and not (owner(nid) & own)
        if ends_only and d_own > FILL_END_M:
            continue
        cands.append((float(np.hypot(K[:, 0] - p[0], K[:, 1] - p[1]).min()),
                      nid, ends_only))
    cands.sort()

    out = list(seq)

    def xy_or_none(n):
        return xy(n) if n in stations else None

    for _d, nid, ends_only in cands:
        if (nid in have or _name_key(stations[nid]) in taken
                or _at_spot(stations[nid], spots, scale)):
            continue
        if not _insert_cheapest(out, nid, xy_or_none, ends_only):
            continue
        have.add(nid)
        if _name_key(stations[nid]):
            taken.add(_name_key(stations[nid]))
        spots.append((stations[nid][0], stations[nid][1]))
    return out


def stops_along(path, stations, scale):
    """경로 옆의 역을 경로 위 거리 순으로 세운다."""
    if len(path) < 2:
        return []
    P = np.asarray(path, dtype=np.float64)
    px = P[:, 0] * scale * 111_320.0
    py = P[:, 1] * 111_132.0
    step = np.hypot(np.diff(px), np.diff(py))
    arc = np.concatenate([[0.0], np.cumsum(step)])

    picked = []
    for nid, (lon, lat, _t, _n) in stations.items():
        d = np.hypot(px - lon * scale * 111_320.0, py - lat * 111_132.0)
        k = int(np.argmin(d))
        if d[k] <= TRACK_NEAR_M:
            picked.append((float(arc[k]), nid))
    picked.sort()
    return [nid for _, nid in picked]


def cluster_nodes(pos):
    """정차 노드를 사람이 아는 "역" 단위로 묶는다."""
    ids = sorted(pos)
    lat0 = float(np.median([pos[i][1] for i in ids])) if ids else 35.0
    scale = float(np.cos(np.radians(lat0)))
    by_name = defaultdict(list)
    for i in ids:
        _lon, _lat, titles, nm = pos[i]
        by_name[titles.get("ja") or nm or ("#" + str(i))].append(i)

    of, members = {}, []
    for _, group in by_name.items():
        if len(group) == 1:
            of[group[0]] = len(members)
            members.append(list(group))
            continue
        x = np.array([pos[i][0] for i in group]) * scale * 111_320.0
        y = np.array([pos[i][1] for i in group]) * 111_132.0
        left = set(range(len(group)))
        while left:
            seed = left.pop()
            cl, frontier = {seed}, [seed]
            while frontier:
                k = frontier.pop()
                near = [j for j in list(left)
                        if np.hypot(x[j] - x[k], y[j] - y[k]) <= SAME_STATION_M]
                for j in near:
                    left.discard(j)
                    cl.add(j)
                    frontier.append(j)
            slot = len(members)
            for j in cl:
                of[group[j]] = slot
            members.append([group[j] for j in sorted(cl)])
    return of, members


def _same_station(a: set, b: set) -> bool:
    """이름으로 봐서 한 역인가. 이름이 똑같아야 한 역으로 본다.

    거리만 보고 합치면 안 된다. 新川崎 와 鹿島田 은 300m 떨어진 서로
    다른 역인데 합쳐져, 신카와사키가 데이터에서 사라지고 쇼난신주쿠라인이
    가시마다로 꺾여 들어갔다.

    글자를 나눠 가지면 합치는 것도 안 된다. 그러면 大阪 와 大阪梅田,
    梅田 와 東梅田 처럼 실제로 다른 역까지 한 점이 된다. 이름이 똑같을
    때만 합친다. 이름이 갈린 환승역(大阪/梅田)은 합치지 않아도 400m
    안이면 build_naive 가 환승 간선을 따로 놓아 준다.

    이름이 없는 정차 노드는 가릴 재료가 없으니 거리만 보고 합친다.
    """
    if not a or not b:
        return True
    return bool(a & b)


def merge_nearby(of, members, pos, routes, scale):
    """이름이 달라도 붙어 있는 묶음을 합친다.

    三ノ宮/神戸三宮, 大阪/大阪梅田/梅田 처럼 한 환승역인데 이름이 갈린
    경우가 많다. 안 합치면 같은 역이 여러 개로 세어지고 환승도 끊긴다.

    다만 같은 노선에서 잇닿은 역끼리는 합치지 않는다. 노면전차 정류장은
    300m 간격인 곳이 흔해서, 그대로 합치면 한 줄이 통째로 한 점이 된다.
    """
    n = len(members)
    cen = np.zeros((n, 2))
    label = []
    for k, ids in enumerate(members):
        cen[k] = np.mean([[pos[i][0], pos[i][1]] for i in ids], axis=0)
        names = {(pos[i][2].get("ja") or pos[i][3] or "").strip() for i in ids}
        label.append({v for v in names if v})
    x = cen[:, 0] * scale * 111_320.0
    y = cen[:, 1] * 111_132.0

    forbid = set()
    for r in routes:
        seq = []
        for nid in r["stops"]:
            c = of.get(nid)
            if c is not None and (not seq or seq[-1] != c):
                seq.append(c)
        for a, b in zip(seq, seq[1:]):
            forbid.add((a, b))
            forbid.add((b, a))

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # 금지한 쌍은 직접 잇지 않는 것만으로 모자라다. 양쪽에서 300m 안에
    # 있는 제3의 묶음을 거치면 결국 한 덩이가 된다(오사카·고베에 흔하다).
    # 뿌리 단위로 들고 다니며, 읽을 때 지금 뿌리로 고쳐 본다.
    ban = defaultdict(set)
    for u, v in forbid:
        ban[u].add(v)

    def banned(a, b):
        cur = {find(x) for x in ban.get(a, ())}
        if cur:
            ban[a] = cur
        return b in cur

    order = np.argsort(x)
    xs = x[order]
    for k in range(len(order)):
        i = int(order[k])
        hi = int(np.searchsorted(xs, xs[k] + CROSS_NAME_M, "right"))
        for m in range(k + 1, hi):
            j = int(order[m])
            if abs(y[i] - y[j]) > CROSS_NAME_M:
                continue
            if np.hypot(x[i] - x[j], y[i] - y[j]) > CROSS_NAME_M:
                continue
            a, b = find(i), find(j)
            if a == b or banned(a, b) or banned(b, a):
                continue
            if not _same_station(label[i], label[j]):
                continue
            parent[a] = b
            if a in ban:
                ban[b] |= ban.pop(a)

    slot, new_members = {}, []
    for k in range(n):
        root = find(k)
        if root not in slot:
            slot[root] = len(new_members)
            new_members.append([])
        new_members[slot[root]].extend(members[k])
    new_of = {nid: slot[find(c)] for nid, c in of.items()}
    return new_of, new_members


def overlap(small, big):
    if not small:
        return 0.0
    s = set(big)
    return sum(1 for c in small if c in s) / len(small)


# ---------------------------------------------------------------------------
# OSM 추출 결과를 저장해 두고 다시 쓴다.
#
# 이 파일은 PBF 를 세 번 훑는다. 관계, 웨이(노드 좌표 색인까지), 노드다.
# 간토는 추출본이 970 MB 라 세 번에 8~10분이 걸리는데, 그 뒤의 실제
# 계산(순서 고치기, 빠진 역 메우기, 묶기, 이름 붙이기)은 30초 남짓이다.
# 알고리즘만 고쳐 다시 돌릴 때 그 8분을 매번 다시 쓰게 된다.
#
# 그래서 훑은 결과를 그대로 저장해 둔다. 20k 노드와 36k 웨이라 50 MB 쯤이다.
#
#     REGION=<권역> RAIL_REUSE=1 python src/build_rail.py
#
# PBF 와 권역 경계, 제외 목록이 그대로일 때만 쓴다. 하나라도 바뀌면
# 저장해 둔 것을 버리고 다시 훑는다. 켤 때만 쓴다.
_CACHE_NPZ = OUT / "osm-extract.npz"
_CACHE_JSON = OUT / "osm-extract.json"


class _Bag:
    """핸들러 자리에 끼울 껍데기."""


def _cache_stamp(pbfs):
    def of(p):
        try:
            st = p.stat()
            return [p.name, st.st_size, int(st.st_mtime)]
        except OSError:
            return [str(p), 0, 0]
    extra = [ROOT / "data" / "excluded-lines.json",
             ROOT / "data" / "regions" / REGION / "region.json",
             OUT / "prefecture-rings.json"]
    return {"pbf": [of(p) for p in pbfs], "extra": [of(p) for p in extra]}


def save_osm_cache(pbfs, rel, ways, nodes):
    way_ids = sorted(ways.geom)
    xy, gptr = [], [0]
    for w in way_ids:
        a = np.asarray(ways.geom[w], dtype=np.float64).reshape(-1, 2)
        xy.append(a)
        gptr.append(gptr[-1] + len(a))
    ref_ids = sorted(ways.refs)
    rf, rptr = [], [0]
    for w in ref_ids:
        a = np.asarray(ways.refs[w], dtype=np.int64)
        rf.append(a)
        rptr.append(rptr[-1] + len(a))
    node_ids = sorted(nodes.pos)
    np.savez_compressed(
        _CACHE_NPZ,
        way_ids=np.asarray(way_ids, dtype=np.int64),
        geom_xy=(np.concatenate(xy) if xy else np.zeros((0, 2))),
        geom_ptr=np.asarray(gptr, dtype=np.int64),
        ref_ids=np.asarray(ref_ids, dtype=np.int64),
        ref_nodes=(np.concatenate(rf) if rf else np.zeros(0, dtype=np.int64)),
        ref_ptr=np.asarray(rptr, dtype=np.int64),
        node_ids=np.asarray(node_ids, dtype=np.int64),
        node_xy=np.asarray([[nodes.pos[n][0], nodes.pos[n][1]]
                            for n in node_ids], dtype=np.float64).reshape(-1, 2),
        is_station=np.asarray([n in nodes.stations for n in node_ids], dtype=bool),
        is_rail=np.asarray([n in nodes.rail for n in node_ids], dtype=bool),
    )
    _CACHE_JSON.write_text(json.dumps({
        "stamp": _cache_stamp(pbfs),
        "routes": rel.routes,
        "dropped": rel.dropped,
        "skipped": rel.skipped,
        "names": {str(n): [nodes.pos[n][2], nodes.pos[n][3]] for n in node_ids},
    }, ensure_ascii=False), encoding="utf-8")


def load_osm_cache(pbfs):
    if not (_CACHE_NPZ.exists() and _CACHE_JSON.exists()):
        return None
    try:
        meta = json.loads(_CACHE_JSON.read_text(encoding="utf-8"))
        if meta.get("stamp") != _cache_stamp(pbfs):
            print("  (저장해 둔 OSM 추출이 지금 파일과 안 맞아 다시 훑는다)",
                  flush=True)
            return None
        z = np.load(_CACHE_NPZ)
    except (OSError, ValueError) as e:
        print(f"  (저장해 둔 OSM 추출을 못 읽었다: {e})", flush=True)
        return None

    rel = _Bag()
    rel.routes = meta["routes"]
    # 종별은 이름에서 다시 매긴다. 종별 규칙만 고쳤을 때 PBF 를 다시
    # 훑지 않아도 되게.
    for r in rel.routes:
        r["kind"] = kind_of(r["name"])
    rel.dropped = meta.get("dropped", 0)
    rel.skipped = meta.get("skipped", 0)
    rel.want_nodes = {n for r in rel.routes for n in r["stops"]}
    rel.want_ways = {w for r in rel.routes for w in r["ways"]}

    ways = _Bag()
    wid, gp, gx = z["way_ids"], z["geom_ptr"], z["geom_xy"]
    ways.geom = {int(w): gx[gp[i]:gp[i + 1]] for i, w in enumerate(wid)}
    rid, rp, rn = z["ref_ids"], z["ref_ptr"], z["ref_nodes"]
    ways.refs = {int(w): rn[rp[i]:rp[i + 1]].tolist() for i, w in enumerate(rid)}

    nodes = _Bag()
    nid, nxy = z["node_ids"], z["node_xy"]
    st, ra = z["is_station"], z["is_rail"]
    names = meta["names"]
    nodes.pos, nodes.stations, nodes.rail = {}, {}, set()
    for i, n in enumerate(nid):
        n = int(n)
        titles, nm = names[str(n)]
        rec = (float(nxy[i, 0]), float(nxy[i, 1]), titles, nm)
        nodes.pos[n] = rec
        if st[i]:
            nodes.stations[n] = rec
        if ra[i]:
            nodes.rail.add(n)
    return rel, ways, nodes


def main():
    pbfs = _pbf_list()
    print("[" + REGION + "] " + ", ".join(p.name for p in pbfs) + " 읽는 중...",
          flush=True)

    cached = (load_osm_cache(pbfs)
              if os.environ.get("RAIL_REUSE") == "1" else None)
    if cached is not None:
        rel, ways, nodes = cached
        print(f"  저장해 둔 OSM 추출을 다시 쓴다 "
              f"(관계 {len(rel.routes):,}개, 웨이 {len(ways.geom):,}개, "
              f"노드 {len(nodes.pos):,}개)", flush=True)
        return _build(pbfs, rel, ways, nodes)

    rel = Relations()
    for p in pbfs:
        rel.apply_file(str(p))
    track_only = sum(1 for r in rel.routes if not r["stops"])
    print(f"  철도 계통 관계 {len(rel.routes):,}개 "
          f"(신칸센 {rel.dropped}개, 못 타는 노선 {rel.skipped}개 제외), "
          f"선로만 있는 것 {track_only:,}개", flush=True)

    ways = Ways(rel.want_ways)
    for p in pbfs:
        ways.apply_file(str(p), locations=True, idx="flex_mem")
    print(f"  선로 웨이 {len(ways.geom):,}개", flush=True)

    inside, how = _region_filter()
    nodes = StationNodes(rel.want_nodes, inside)
    for p in pbfs:
        nodes.apply_file(str(p))
    if how:
        print(f"  권역 밖은 버린다 ({how})", flush=True)
    print(f"  역 노드 {len(nodes.stations):,}개, 정차 노드 {len(nodes.pos):,}개",
          flush=True)
    try:
        save_osm_cache(pbfs, rel, ways, nodes)
    except Exception as e:        # 캐시를 못 써도 빌드는 계속한다
        print(f"  (OSM 추출을 저장하지 못했다: {e})", flush=True)
    return _build(pbfs, rel, ways, nodes)


def extend_lines(lines, members, pos):
    """data/line-extensions.json 에 적은 대로 노선 끝을 이어 붙인다.

    OSM 은 노선을 시설 기준으로 적어 둔다. 名鉄犬山線 은 新鵜沼-下小田井
    이지만 열차는 역이 아닌 枇杷島分岐点 에서 본선으로 들어가 名鉄名古屋
    까지 간다. 분기점이 역이 아니라 두 노선이 어느 역에서도 만나지 않고,
    나고야에서 이누야마까지 기후를 돌아 80분이 나왔다.
    """
    path = ROOT / "data" / "line-extensions.json"
    if not path.exists():
        return
    book = json.loads(path.read_text(encoding="utf-8")).get(REGION) or {}
    if not book:
        return
    by_name = defaultdict(list)
    for cl, ns in enumerate(members):
        if ns:
            v = pos[ns[0]]
            by_name[(v[2].get("ja") or v[3] or "").strip()].append(cl)
    for ln in lines:
        rep = ln["rep"]["name"]
        ext = book.get(rep) or book.get(base_name(rep))
        if not ext:
            continue
        for end, names in ext.items():
            cls = []
            for nm in names:
                got = by_name.get(nm) or []
                if len(got) != 1:
                    print(f"  !! 이어 붙일 역 {nm} 이 {len(got)}개라 {rep} 를 "
                          f"잇지 않는다", flush=True)
                    break
                cls.append(got[0])
            else:
                seq = ln["seq"]
                if by_name.get(end) == [seq[-1]]:
                    ln["seq"] = seq + cls
                elif by_name.get(end) == [seq[0]]:
                    ln["seq"] = cls[::-1] + seq
                else:
                    print(f"  !! {rep} 의 끝이 {end} 가 아니라 잇지 않는다",
                          flush=True)
                    continue
                print(f"  {rep}: {end} 뒤로 {'·'.join(names)} 를 이었다",
                      flush=True)


def _build(pbfs, rel, ways, nodes):

    lat0 = float(np.median([v[1] for v in nodes.pos.values()])) if nodes.pos else 35.0
    scale = float(np.cos(np.radians(lat0)))

    # 선로만 있는 관계는 선로 옆 역에서 정차 순서를 복원한다.
    # 통과 계통에는 쓸 수 없다. 선로 옆 역을 모두 주우면 통과하는 역까지
    # 정차로 만들어, 특급이 각역정차로 둔갑하고 그 긴 목록이 진짜 노선을
    # 삼켜 버린다. 종별이 적힌 관계는 정차역 멤버가 있을 때만 쓴다.
    #
    # route=railway 는 계통이 아니라 선로 그 자체다. 정차역 멤버가 몇 개
    # 붙어 있어도 그것은 정차 패턴이 아니라 누군가 넣다 만 것이다.
    # 山陰本線 은 웨이가 1,820개인데 정차 노드가 5개뿐이라, 그대로 믿으면
    # 소노베 위쪽 교토 산간이 통째로 사라진다. 이런 관계는 정차역이
    # 있어도 선로에서 되살린다.
    recovered, by_node, picked_up = 0, 0, 0
    # 빠진 역을 메울 후보. 어느 관계도 정차역으로 가리키지 않은 역
    # 노드만 남긴다. 남의 노선 역과 같은 역의 다른 노드를 빼는 거름이다.
    # 특급이 부른 역은 거르지 않는다. 특급 정차역은 그 선로의 완행역이기도
    # 해서, 거르면 高山本線 이 ひだ 가 서는 下呂·高山 등을 못 줍고 그 역들이
    # ひだ 에만 붙는다. 쾌속·급행은 그대로 거른다. 総武快速線 처럼 제 선로를
    # 가진 쾌속이 있어, 풀면 新日本橋 가 옆의 銀座線 에 붙는다.
    claimed = {_name_key(nodes.stations[n]) for r in rel.routes
               if r["kind"] != "특급"
               for n in r["stops"] if n in nodes.stations}
    pool = [n for n in nodes.rail
            if _name_key(nodes.stations[n]) not in claimed
            and "貨物" not in _name_key(nodes.stations[n])]
    print(f"  어느 계통도 안 부른 역 노드 {len(pool):,}개", flush=True)
    owner = _nearest_track(ways.geom, scale, nodes.pos)
    for r in rel.routes:
        track = r.get("rtype") == "railway"
        # 정차역을 되살릴 필요가 없는 관계라도 빠진 역은 메워야 한다.
        # 예전에는 여기서 바로 건너뛰어, 정차역이 두 개 이상인 관계는
        # fill_missing 을 아예 안 거쳤다. JR日光線 이 정차역 6개를 들고
        # 있어서 건너뛰었고, 그래서 日光 역이 끝내 안 들어왔다.
        need = track or (len(r["stops"]) < 2
                         and r["kind"] in (None, "각역정차"))
        if not need:
            if r["kind"] in (None, "각역정차") and len(r["stops"]) >= 2:
                filled = fill_missing(r["stops"], r["ways"], ways.geom,
                                      nodes.pos, pool, scale, owner)
                if len(filled) > len(r["stops"]):
                    picked_up += len(filled) - len(r["stops"])
                    r["stops"] = filled
            continue
        # 먼저 웨이의 구성 노드에서 찾는다. 확실한 대신 갖춰지지 않은
        # 노선이 있어, 그때만 선로 옆 거리로 줍는다.
        seq = stops_on_ways(r["ways"], ways.refs, nodes.stations, scale)
        seq = _drop_strays(
            seq,
            lambda n: ((nodes.pos[n][0] * scale * 111_320.0,
                        nodes.pos[n][1] * 111_132.0) if n in nodes.pos else None),
            lambda n: _name_key(nodes.pos[n]) if n in nodes.pos else "",
            slack_m=STRAY_SLACK_M)
        found = "웨이 노드"
        if len(seq) < 2 and not track:
            seq = stops_along(stitch(r["ways"], ways.geom), nodes.stations, scale)
            found = "선로 옆"
        elif len(seq) < 2:
            # 선로 관계는 간선이면 나란한 남의 노선 역까지 줍는다. 어느
            # 계통도 안 부른 역(pool)과 선로 위에서 이미 찾은 역만 본다.
            # 樽見鉄道 는 계통 관계 없이 선로 관계만 있고, 역이 전부
            # 선로 옆 railway=station 이라 大垣 하나만 잡혀 노선째 빠졌다.
            near = {n: nodes.stations[n] for n in pool}
            near.update({n: nodes.pos[n] for n in seq if n in nodes.pos})
            seq = stops_along(stitch(r["ways"], ways.geom), near, scale)
            found = "선로 옆"
        # 웨이 구성 노드만 보면 놓치는 역이 있다. OSM 이 그 역에 선로
        # 위 정차 노드를 안 두고 옆에 railway=station 노드만 둔 경우다.
        # 간사이는 그런 역이 141개다(玄武洞·養父·国府·佐津·柴山·立木 등).
        # 제 선로 사슬 가까이 있고 아직 없는 역만 끼워 넣는다. 통과 계통에는
        # 하지 않는다. 지나가는 역이 정차역이 되어 버린다.
        if len(seq) >= 2 and r["kind"] in (None, "각역정차"):
            # 좌표·이름은 pos 에서 본다. 관계가 부른 정차 노드 중에는
            # 역 태그가 없어 stations 에 없는 것이 있는데, 그것을 모르면
            # 같은 역이 이름만 같은 다른 노드로 또 들어간다.
            filled = fill_missing(seq, r["ways"], ways.geom,
                                  nodes.pos, pool, scale, owner)
            if len(filled) > len(seq):
                picked_up += len(filled) - len(seq)
                seq = filled
        if len(seq) >= 2 and len(seq) >= len(r["stops"]):
            r["stops"] = seq
            recovered += 1
            by_node += found == "웨이 노드"
    print(f"  선로 옆에서 더 주운 역 {picked_up:,}개", flush=True)
    print(f"  선로에서 정차 순서를 되살린 관계 {recovered:,}개 "
          f"(웨이 노드 {by_node:,}, 선로 옆 {recovered - by_node:,})", flush=True)

    used = {n for r in rel.routes for n in r["stops"]}
    pos = {n: nodes.pos[n] for n in used if n in nodes.pos}

    # 노선 관계가 이름 없는 stop_position 을 가리키는 경우가 있다. 그대로
    # 두면 이름 없는 역이 되어 검색도 표시도 안 된다. 바로 옆의 이름 있는
    # 역 노드에서 이름을 물려받는다. 지치부 본선이 27개가 그랬는데 전부
    # 100m 안팎에 짝이 있었다.
    blank = [n for n, v in pos.items() if not (v[2].get("ja") or v[3] or "").strip()]
    if blank and nodes.stations:
        keys = list(nodes.stations)
        sp = np.array([[nodes.stations[k][0], nodes.stations[k][1]] for k in keys])
        sx = sp[:, 0] * scale * 111_320.0
        sy = sp[:, 1] * 111_132.0
        filled = 0
        for n in blank:
            lon, lat, titles, nm = pos[n]
            d = np.hypot(sx - lon * scale * 111_320.0, sy - lat * 111_132.0)
            k = int(np.argmin(d))
            if d[k] <= NAME_ADOPT_M:
                src = nodes.stations[keys[k]]
                titles = dict(titles)
                for g in LANGS:
                    if not titles.get(g) and src[2].get(g):
                        titles[g] = src[2][g]
                if not titles.get("ja"):
                    titles["ja"] = src[3]
                pos[n] = (lon, lat, titles, src[3] or nm)
                filled += 1
        print(f"  이름 없는 정차 노드 {len(blank)}개 중 {filled}개에 "
              f"옆 역 이름을 붙였다", flush=True)
    of, members = cluster_nodes(pos)
    before = len(members)
    of, members = merge_nearby(of, members, pos, rel.routes, scale)
    print(f"  역 묶음 {len(members):,}개 "
          f"(이름이 달라도 {CROSS_NAME_M:.0f}m 안이면 합쳐 {before:,} -> {len(members):,})",
          flush=True)

    cpos = {cl: (pos[ns[0]][0], pos[ns[0]][1])
            for cl, ns in enumerate(members) if ns}
    mended = reseated = 0
    for r in rel.routes:
        seq = []
        for nid in r["stops"]:
            c = of.get(nid)
            if c is not None and (not seq or seq[-1] != c):
                seq.append(c)
        fixed = repair_order(seq, cpos, scale)
        # 역들이 제 선로 위에 차례대로 놓여 있으면 그 순서는 선로가
        # 보증한다. 길이로 옮기지 않는다. 선로가 고리처럼 도는 노선은 틀린
        # 자리가 몇 % 짧게 나와 옳은 관계 순서를 망가뜨렸다(ゆりかもめ 의
        # お台場海浜公園 가 青海 옆으로, ユーカリが丘線 의 井野 가 公園 옆으로).
        # 튀는 간격으로 가르려 하니 成田線 의 千葉 11km 왕복을 못 잡았다.
        # 출발역으로 돌아오는 닫힌 관계(순환선, ユーカリが丘線 같은 라켓
        # 모양)도 옮기지 않는다. 중복을 지운 순서가 이미 선로 순서다.
        closed = len(seq) >= 4 and seq[0] == seq[-1]
        # 라켓 모양(줄기를 지나 고리를 돌고 줄기로 돌아온다)은 고리를 처음
        # 닫는 곳까지 쓴다. 중복을 다 지우면 고리의 마지막 구간이 빠진다.
        # ユーカリが丘線 이 井野 에서 公園 으로 돌아오지 못하고 끊겼다.
        # 바로 뒤돌아서는 왕복(阪神なんば線 관계)은 라켓이 아니다.
        # 고리를 돈 뒤 남은 것이 들어온 줄기를 그대로 거꾸로 되짚어야
        # 라켓이다. 간사이의 近鉄名古屋線 은 미에 쪽만 잘려 남은 목록이
        # 우연히 닫혀, 이 조건 없이는 라켓으로 잘못 읽혔다.
        turn = next((i for i, c in enumerate(seq) if c in seq[:i]), None)
        stem = seq.index(seq[turn]) if turn is not None else 0
        racket = (closed and turn is not None and turn - stem >= 3
                  and seq[turn + 1:] == seq[:stem][::-1])
        if racket:
            seated = seq[:turn + 1]
        elif closed or _follows_track(fixed, cpos, r["ways"], ways.geom, scale):
            seated = fixed
        else:
            seated = reseat_strays(fixed, cpos, scale)
        if seated != fixed:
            reseated += 1
            fixed = seated
        mended += fixed is not seq
        r["seq"] = fixed
    print(f"  정차 순서가 튀어 다시 이은 계통 {mended:,}개"
          + (f" (제자리로 옮긴 역이 있는 계통 {reseated:,}개)" if reseated else ""),
          flush=True)
    routes = [r for r in rel.routes if len(r["seq"]) >= 2]
    # 뼈대는 각역정차 쪽에서 고른다. 통과 계통을 먼저 집으면 그 긴 회랑이
    # 뼈대가 되고 진짜 노선들이 그 밑으로 빨려 들어간다.
    routes.sort(key=lambda r: (r["kind"] not in (None, "각역정차"),
                               -len(r["seq"]), r["name"]))

    lines, patterns = [], []
    for r in routes:
        best, best_ov = None, 0.0
        for k, ln in enumerate(lines):
            ov = overlap(r["seq"], ln["seq"])
            if ov > best_ov:
                best, best_ov = k, ov
        if best is not None and best_ov >= SAME_LINE:
            if best_ov >= DUP_LINE and r["kind"] in (None, "각역정차"):
                lines[best]["rels"].append(r)
            else:
                patterns.append((best, r))
            continue
        lines.append({"seq": r["seq"], "rep": r, "rels": [r]})
    # 같은 노선으로 묶인 계통 중 가장 긴 것의 역만 쓰면 다른 계통에만 있는
    # 역이 빠진다. 近鉄山田線 은 선로 관계가 뼈대가 되며 완행 계통에만
    # 있는 종점 宇治山田 를 잃었다. 노선 끝에만 붙인다. 한가운데까지 받으면
    # 묶인 계통의 오류가 딸려 온다. きのさき 가 품은 西舞鶴·東舞鶴 이
    # 山陰本線 에, 空港線 계통의 関西空港 이 南海本線 에 끼었다.

    def cxy(c):
        return ((cpos[c][0] * scale * 111_320.0, cpos[c][1] * 111_132.0)
                if c in cpos else None)
    for ln in lines:
        for r in ln["rels"][1:]:
            for c in r["seq"]:
                if c not in ln["seq"] and _name_key(pos[members[c][0]]):
                    ln["seq"] = list(ln["seq"])
                    _insert_cheapest(ln["seq"], c, cxy, ends_only=True)
    # 끝에 못 붙은 역이 그 계통의 한쪽 끝에 이어져 있으면 지선이다.
    # 豊橋鉄道東田本線 의 運動公園前 는 井原 에서 갈라지는 한 정거장짜리
    # 지선인데, 그 계통이 본선과 90% 넘게 겹쳐 한 노선으로 묶이며 버려졌다.
    # 아래 "지선만 덮는 계통" 으로 넘겨 지선을 세운다. 계통 한가운데에만
    # 있는 역(近鉄奈良線 계통에 잘못 실린 鳥居前)이나 멀리 튀는 역
    # (きのさき 가 품은 西舞鶴, 城崎温泉 에서 60km)은 넘기지 않는다.
    for k, ln in enumerate(lines):
        have_seq = set(ln["seq"])
        for r in ln["rels"][1:]:
            seq_r = [c for c in r["seq"] if _name_key(pos[members[c][0]])]
            out_at = [i for i, c in enumerate(seq_r) if c not in have_seq]
            if not out_at or len(seq_r) < 2:
                continue
            n = len(seq_r)
            tail = out_at == list(range(n - len(out_at), n))
            head = out_at == list(range(len(out_at)))
            if not (tail or head) or len(out_at) == n:
                continue
            part = (seq_r[out_at[0] - 1:] if tail else seq_r[:out_at[-1] + 2])
            gaps = [float(np.hypot(*(np.asarray(cxy(x)) - np.asarray(cxy(y)))))
                    for x, y in zip(part, part[1:])]
            if max(gaps) <= BRANCH_GAP_M:
                patterns.append((k, r))
    # 지선만 덮는 계통은 통과 계통이 아니라 노선이다.
    #
    # 東武小泉線 은 OSM 에 館林 => 西小泉 과 館林 => 太田 두 계통으로 들어
    # 있다. 뒤의 것이 앞의 것과 71% 겹쳐 SAME_LINE 과 DUP_LINE 사이에
    # 걸리고, 그러면 "부분 계통" 으로 밀려 정차 패턴만 남는다. 그 계통에만
    # 있는 竜舞 와 太田 쪽 가지는 통째로 사라져, 지도에서 히가시코이즈미
    # 앞에서 선이 끊긴 채 끝난다. ODPT 는 이 가지를 따로 노선으로 들고
    # 있어서 시각표판에는 제대로 나온다.
    #
    # 진짜 통과 계통은 정차역이 뼈대의 부분집합이라 새로 가져오는 역이
    # 없다. 새로 가져오는 역이 있으면 그것은 지선이므로 노선으로 세운다.
    covered = {c for ln in lines for c in ln["seq"]}
    rest, promoted = [], 0
    for best, r in sorted(patterns, key=lambda z: -len(z[1]["seq"])):
        fresh = [c for c in r["seq"] if c not in covered]
        if len(fresh) >= MIN_BRANCH:
            # 계통을 통째로 세우면 줄기가 두 번 그려진다. 東武小泉線 은
            # 관림-히가시코이즈미가 겹쳐 선이 두 겹으로 보였다. 새로
            # 가져오는 역이 든 토막만 남기고, 줄기에 붙도록 양쪽으로 한
            # 역씩만 더 붙인다. 그러면 지선만 남는다.
            at = [i for i, c in enumerate(r["seq"]) if c in set(fresh)]
            lo = max(min(at) - 1, 0)
            hi = min(max(at) + 2, len(r["seq"]))
            branch = r["seq"][lo:hi]
            if len(branch) < 2:
                rest.append((best, r))
                continue
            lines.append({"seq": branch, "rep": r, "rels": [r]})
            covered.update(branch)
            promoted += 1
        else:
            rest.append((best, r))
    patterns = rest
    print(f"  뼈대 노선 {len(lines):,}개 (지선이라 따로 세운 계통 {promoted}개), "
          f"통과·부분 계통 {len(patterns):,}개", flush=True)

    extend_lines(lines, members, pos)

    railways, stations, express = [], [], []
    used_ids, seen_sid = set(), set()

    for k, ln in enumerate(lines):
        rep = ln["rep"]
        base = base_name(rep["name"]) or rep["name"]
        slug = re.sub(r"[^0-9A-Za-z]+", "", rep["operator"])[:20] or "OSM"
        tail = re.sub(r"[^0-9A-Za-z]+", "", base)
        lid = slug + "." + (tail if tail else str(k))
        while lid in used_ids:
            lid += "_"
        used_ids.add(lid)

        # 이 노선이 실제로 서는 노드를 쓴다. 묶음의 첫 노드를 쓰면
        # 이름만 같고 승강장이 다른 역에서 남의 자리로 간다. 豊島園 은
        # 세이부와 오에도가 111m 떨어져 있는데, 세이부선이 오에도
        # 승강장으로 가 선이 그리로 끌려갔다.
        #
        # 묶음 자체는 그대로 둔다. 검색과 세는 단위는 여전히 한 역이고,
        # 지도에 찍는 자리만 노선별로 갈라진다.
        own = {}
        for r in ln["rels"]:
            for nid in r["stops"]:
                c = of.get(nid)
                if c is not None:
                    own.setdefault(c, nid)

        order = []
        for cl in ln["seq"]:
            sid = lid + "." + str(cl)
            if sid not in seen_sid:
                seen_sid.add(sid)
                lon, lat, titles, nm = pos[own.get(cl) or members[cl][0]]
                titles = dict(titles)
                if not titles.get("ja"):
                    titles["ja"] = nm
                stations.append({"id": sid, "title": titles, "cluster": cl,
                                 "coord": [lon, lat], "railway": lid})
            order.append(sid)

        titles = {g: "" for g in LANGS}
        for r in ln["rels"]:
            for g in LANGS:
                if not titles[g] and r["titles"].get(g):
                    titles[g] = r["titles"][g]
        titles["ja"] = titles["ja"] or base

        railways.append({"id": lid, "title": titles, "stations": order,
                         "operator": rep["operator"], "colour": rep["colour"],
                         "clusters": ln["seq"]})
        ln["lid"] = lid

    for k, r in patterns:
        express.append({"railway": lines[k]["lid"], "kind": r["kind"] or "부분",
                        "name": r["name"], "clusters": r["seq"]})

    by_cluster = defaultdict(list)
    for s in stations:
        by_cluster[s["cluster"]].append(s["id"])
    groups = [ids for _, ids in sorted(by_cluster.items())]

    OUT.mkdir(parents=True, exist_ok=True)
    # 지도에 그릴 선형. 역과 역을 직선으로 이으면 실제 선로와 어긋난다.
    # mini-tokyo-3d 의 coordinates.json 과 같은 모양으로 맞춘다.
    shapes = []
    for ln in lines:
        # 사슬마다 따로 담는다. 하나로 이으면 사슬 사이가 직선으로 이어져
        # 지도에서 선로를 크게 벗어난다.
        for k, path in enumerate(stitch_all(ln["rep"]["ways"], ways.geom)):
            if len(path) < 2:
                continue
            shapes.append({"id": ln["lid"] if k == 0 else f"{ln['lid']}~{k}",
                           "sublines": [{"type": "main",
                                         "coords": [[round(x, 6), round(y, 6)]
                                                    for x, y in path]}]})
    body = json.dumps({"railways": shapes, "airways": []}, ensure_ascii=False)
    (OUT / "coordinates.json").write_text(body, encoding="utf-8")
    # 같은 것을 한 벌 더 남긴다. build_track.py 가 선로를 따라 다시 그린
    # 뒤 coordinates.json 을 덮어쓰는데, 거기서 못 이은 구간에는 아무
    # 선형도 남지 않는다. 관계에서 뽑은 이 선형을 후보로 두면 그런
    # 자리에서 선이 끊기는 대신 거친 선형이라도 이어진다.
    (OUT / "coordinates-rel.json").write_text(body, encoding="utf-8")

    for name, data in (("railways.json", railways),
                       ("stations.json", stations),
                       ("station-groups.json", [[g] for g in groups]),
                       ("express.json", express)):
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    kinds = defaultdict(int)
    for e in express:
        kinds[e["kind"]] += 1
    n_exp = len({e["railway"] for e in express if e["kind"] != "부분"})
    print(f"\n  노선 {len(railways):,}개  역 {len(stations):,}개  "
          f"역 묶음 {len(groups):,}개")
    print("  계통: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print(f"  통과 계통이 있는 노선 {n_exp:,}개 "
          f"({n_exp / max(len(railways), 1) * 100:.0f}%)")
    print(f"  노선 선형 {len(shapes):,}개")
    print("  저장 -> " + str(OUT))


if __name__ == "__main__":
    main()
