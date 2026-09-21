"""통과 계통의 정차 패턴을 일본어 위키백과에서 읽어 노선에 붙인다.

OSM 에도 통과 계통이 있지만 간사이 주요 노선의 14% 뿐이다. 京阪本線,
近鉄大阪線, 南海高野線처럼 그 지역 이동의 뼈대가 되는 노선에 특급·급행이
없으면 도달 범위가 크게 어긋난다. 간토 시각표로 재 보면 통과 계통은
도달 범위의 4분의 1을 좌우한다.

위키백과 '駅一覧' 표는 종별을 열로, 역을 행으로 놓고 ● 와 ｜ 로 정차와
통과를 적어 둔다. 사람 보라고 만든 표지만 규칙이 일정해 기계로 읽힌다.
정차역 목록 자체는 사실이라 저작권 대상이 아니지만 출처는 남긴다.
(일본어 위키백과, CC BY-SA 4.0)

노선을 잇는 기준은 이름이 아니라 역 이름 겹침이다. 위키의 '近鉄大阪線'
과 OSM 의 '近畿日本鉄道大阪線' 은 이름이 다르지만 역 목록은 거의 같다.

결과는 raw/express-wiki.json 에 따로 담는다. build_rail.py 가 만드는
express.json 에 섞으면 노선을 다시 뽑을 때마다 날아간다.

사용법:
  REGION=kansai python src/build_express.py 京阪本線 近鉄大阪線 ...
  REGION=kansai python src/build_express.py          (전에 받아 둔 것으로 다시 붙이기)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rail import kind_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGION = os.environ.get("REGION", "kansai")
RAW = ROOT / "data" / "regions" / REGION / "raw"
CACHE = RAW / "wiki-stops.json"
OUT = RAW / "express-wiki.json"

UA = ("rail-isochrone/0.1 (rail research; "
      "https://github.com/HuijuKim/rail-isochrone)")

# 표에서 찾을 종별. 긴 것부터 봐야 '通勤快急' 이 '快急' 으로 잘리지 않는다.
TYPES = ["快速特急", "通勤快急", "通勤特快", "通勤快速", "通勤急行", "通勤準急",
         "区間急行", "区間快速", "区間準急", "新快速", "快速急行", "直通特急",
         "特急", "急行", "準急", "快速"]
# ▲ 는 일부 열차만 서는 것이라 정차로 본다.
STOP_MARKS = set("●○▲△◎☆")
PASS_MARKS = set("｜|↑↓∥∨∧ーレ")

STATION_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]")
VERT_RE = re.compile(r"\{\{縦書き\|([^|}]+)")
SECTION_RE = re.compile(r"^==+\s*駅一覧\s*==+", re.M)

# 사업자 접두어. OSM 은 '近鉄丹波橋', 위키는 '丹波橋' 처럼 엇갈린다.
PREFIX = re.compile(r"^(JR|ＪＲ|京阪|近鉄|阪急|阪神|南海|山陽|神鉄|叡電|京福)")
MATCH_MIN = 0.55        # 이만큼 겹쳐야 같은 노선으로 본다
MISS_MAX = 0.30         # 정차역의 이만큼까지는 못 찾아도 받아들인다


def wikitext(title: str) -> str | None:
    url = ("https://ja.wikipedia.org/w/api.php?action=parse&prop=wikitext"
           "&format=json&formatversion=2&page=" + urllib.parse.quote(title))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return d["parse"]["wikitext"] if "parse" in d else None


def cells(row: str) -> list[str]:
    """표 한 줄을 칸으로 쪼갠다. 스타일 속성은 떼어낸다."""
    out = []
    for raw in re.split(r"\n[!|]|\|\|", "\n" + row.strip()):
        raw = raw.strip()
        if not raw:
            continue
        if re.match(r'^[a-z-]+\s*=\s*"', raw):      # style="..."|내용
            parts = re.split(r'"\s*\|', raw, maxsplit=1)
            raw = parts[1] if len(parts) > 1 else ""
        out.append(raw.strip())
    return out


def header_types(row: str) -> list[str]:
    found = []
    for c in cells(row):
        m = VERT_RE.search(c)
        text = re.sub(r"<[^>]+>|\[\[|\]\]|\{\{|\}\}", "", m.group(1) if m else c).strip()
        for t in TYPES:
            if text.startswith(t):
                found.append(t)
                break
    return found


def station_of(cell: str) -> str | None:
    m = STATION_RE.search(cell)
    if not m:
        return None
    name = (m.group(2) or m.group(1)).strip()
    return name if name.endswith("駅") else None


def parse(title: str) -> dict | None:
    wt = wikitext(title)
    if not wt:
        return None
    # 본문의 [[#駅一覧]] 링크가 아니라 절 제목을 찾아야 한다. 링크를 집으면
    # '日中の運行パターン' 같은 엉뚱한 표가 걸린다.
    m = SECTION_RE.search(wt)
    if not m:
        return None
    j = wt.find("{|", m.end())
    if j < 0:
        return None
    k = wt.find("\n|}", j)
    rows = wt[j:k if k > 0 else j + 60000].split("\n|-")

    types: list[str] = []
    for row in rows[:4]:
        t = header_types(row)
        if len(t) >= 2:
            types = t
            break
    if not types:
        return None

    # 열마다 따로 담는다. 딕셔너리로 담으면 같은 종별 이름으로 시작하는
    # 열이 둘 있을 때(特急 와 特急〈ひたち〉 처럼) 한 칸으로 합쳐져,
    # 한쪽의 통과역이 다른 쪽의 정차역으로 기록된다.
    stops = [[] for _ in types]
    order = []
    for row in rows:
        cs = cells(row)
        name = next((station_of(c) for c in cs if station_of(c)), None)
        if not name:
            continue
        # 그 노선에 없는 종별 칸은 &nbsp; 로 비어 있다. 표시 개수로 맞추면
        # 한 칸 비는 행이 통째로 버려지므로, 표시가 가장 많이 담기는 창에
        # 맞춘다.
        is_mark = [len(c) == 1 and (c in STOP_MARKS or c in PASS_MARKS) for c in cs]
        if sum(is_mark) < len(types) - 2:
            continue
        width = len(types)
        best, best_n = None, -1
        for start in range(max(len(cs) - width + 1, 1)):
            n = sum(is_mark[start:start + width])
            if n > best_n:
                best_n, best = n, start
        window = cs[best:best + width] if best is not None else []
        if len(window) < width:
            continue
        order.append(name)
        for k, mark in enumerate(window):
            if mark in STOP_MARKS:
                stops[k].append(name)
    return {"title": title, "types": types, "all_stations": order, "stops": stops}


def norm(name: str) -> str:
    s = re.sub(r"駅$", "", name.strip())
    s = re.sub(r"[（(].*?[）)]", "", s)
    return s.replace("ケ", "ヶ").replace(" ", "")


def keys(name: str) -> set[str]:
    a = norm(name)
    return {a, PREFIX.sub("", a)} - {""}


def line_key(name: str) -> tuple[str, str]:
    """노선 이름을 견주기 좋게. (다듬은 것, 회사까지 뗀 것)"""
    from operators import strip_operator_head

    s = re.sub(r"[（(].*?[）)]", "", name or "")
    s = s.split(":")[0].split("：")[0]
    s = re.sub(r"[\s・･>=→\-–—]", "", s)
    return s, strip_operator_head(s)


def name_score(wiki_title: str, osm_ja: str) -> int:
    """이름이 얼마나 같은가. 3 이 가장 같다.

    위키 표가 노선 일부만 들고 있으면 역 겹침으로는 진짜 노선과 그
    노선을 품은 직통 계통이 완전히 동점이 된다. 위키 相鉄本線 표는
    西谷 이후 11개뿐이라 相鉄本線 과 "新宿 => 海老名" 이 둘 다 겹침
    100%, 자카드 61% 였다. 그때는 이름으로 가른다.
    """
    a, sa = line_key(wiki_title)
    b, sb = line_key(osm_ja)
    if not a or not b:
        return 0
    if a == b:
        return 3
    if sa and sa == sb:
        return 2
    if a in b or (sa and sa in sb):
        return 1
    return 0


def merge(wiki: dict) -> list[dict]:
    railways = json.loads((RAW / "railways.json").read_text(encoding="utf-8"))
    stations = {s["id"]: s
                for s in json.loads((RAW / "stations.json").read_text(encoding="utf-8"))}

    index = []
    for r in railways:
        names = defaultdict(set)
        for sid in r["stations"]:
            s = stations.get(sid)
            if s:
                for k in keys(s["title"].get("ja", "")):
                    names[k].add(s["cluster"])
        index.append((r, names))

    added, report = [], []
    for title, data in wiki.items():
        want = set()
        for n in data["all_stations"]:
            want |= keys(n)
        # 겹침만 보면 그 노선을 통째로 품은 직통 계통이 진짜 노선과 똑같이
        # 1.0 이 되고, 먼저 만난 쪽이 이긴다. 相鉄本線 의 표가
        # "新宿 => 海老名" 에, 東急東横線 의 표가 "東京地下鉄の直通運転 -
        # 東急東横線" 에, 近鉄京都線 의 표가 "近畿日本鉄道京都線 普通" 에
        # 붙었다. 그러면 진짜 노선은 통과 계통을 못 받아 각역정차만 깔린다.
        # 여분 역이 적은 쪽을 고르도록 자카드로 동점을 가른다.
        best, best_score = None, (0.0, 0, 0.0)
        for r, names in index:
            hit = sum(1 for k in want if k in names)
            cover = hit / max(len(want), 1)
            tight = hit / max(len(want) + len(names) - hit, 1)
            key = (cover, name_score(title, r["title"].get("ja", "")), tight)
            if key > best_score:
                best_score, best = key, (r, names)
        if best is None or best_score[0] < MATCH_MIN:
            report.append((title, None, best_score[0], 0, len(data["types"])))
            continue
        r, names = best

        n_added = 0
        # 받아 둔 캐시는 stops 를 종별 이름 딕셔너리로 담고 있다. 열
        # 순서 목록으로 펴서 새 꼴과 같이 다룬다.
        by_col = data["stops"]
        if isinstance(by_col, dict):
            by_col = [by_col.get(t, []) for t in data["types"]]
        for kind, stop_list in zip(data["types"], by_col):
            if len(stop_list) < 2 or len(stop_list) >= len(data["all_stations"]):
                continue
            seq, miss = [], 0
            for n in stop_list:
                cl = next((sorted(names[k])[0] for k in keys(n) if k in names), None)
                if cl is None:
                    miss += 1
                elif not seq or seq[-1] != cl:
                    seq.append(cl)
            if len(seq) >= 2 and miss <= len(stop_list) * MISS_MAX:
                # 뼈대 노선 순서대로 세운다. 위키 표와 방향이 다를 수 있다.
                pos = {c: i for i, c in enumerate(r["clusters"])}
                seq.sort(key=lambda c: pos.get(c, 10 ** 6))
                # 종별 이름은 build_rail 과 같은 말을 쓴다. 여기서
                # 表 머리글(特急)을 그대로 두면 build_naive 의 중복
                # 제거 키 (railway, kind) 가 OSM 쪽(특급)과 영영 안
                # 맞아, 이미 있는 통과 계통 위에 한 벌이 더 깔린다.
                added.append({"railway": r["id"],
                              "kind": kind_of(kind) or kind,
                              "name": title + " " + kind, "clusters": seq,
                              "source": "ja.wikipedia"})
                n_added += 1
        report.append((title, r["title"].get("ja", ""), best_score[0], n_added,
                       len(data["types"])))

    print(f"{'위키 노선':<16}{'붙인 OSM 노선':<32}{'겹침':>6}{'추가':>6}{'종별':>5}")
    for title, osm, score, n, total in report:
        print(f"{title:<16}{(osm or '맞는 노선 없음')[:30]:<32}"
              f"{score * 100:>5.0f}%{n:>6}{total:>5}")
    return added


def main() -> None:
    titles = sys.argv[1:]
    wiki = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    for title in titles:
        try:
            r = parse(title)
        except Exception as e:
            print(f"{title}: 실패 {type(e).__name__} {e}", flush=True)
            continue
        if not r:
            print(f"{title}: 駅一覧 표를 못 읽음", flush=True)
            continue
        wiki[title] = r
        # stops 는 종별 이름이 아니라 열 번호로 찾는 목록이다.
        print(f"{title}: 역 {len(r['all_stations'])}개, 종별 "
              + ", ".join(f"{t} {len(c)}"
                          for t, c in zip(r["types"], r["stops"])), flush=True)
        time.sleep(0.5)

    if titles:
        RAW.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(wiki, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n위키 표 {len(wiki)}개 -> {CACHE}\n")

    if not wiki:
        sys.exit("받아 둔 위키 표가 없습니다. 노선 이름을 인자로 주세요.")

    added = merge(wiki)
    OUT.write_text(json.dumps(added, ensure_ascii=False), encoding="utf-8")
    lines = len({e["railway"] for e in added})
    print(f"\n통과 계통 {len(added)}개 ({lines}개 노선) -> {OUT}")


if __name__ == "__main__":
    main()
