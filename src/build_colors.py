"""노선 색을 모아 data/line-colors.json 에 적는다.

OSM 관계에는 colour 태그가 없는 노선이 많다. 간사이는 143개 중 80개,
간토 OSM 판은 178개 중 73개가 그렇다. 전부 회색으로 두면 나란히 달리는
노선을 가를 수가 없고, 아무 색이나 지어 주면 실제 노선 색과 달라진다.

세 군데에서 차례로 찾는다.

  0. 손으로 고친 것      source 가 '손으로' 로 시작하면 건드리지 않는다.
                        원본이 낡았을 때 쓴다(지바 지역은 2020년에 노선색을
                        새로 정했는데 ODPT 값이 그 전 것이다).
  1. 일본어 위키백과      {{Infobox 鉄道路線}} 의 '路線色' 항목. 출처를 하나로
                        두어야 한 노선이 두 이름으로 불려도 색이 안 갈린다.
  2. 이미 가진 색        위키에 없으면 시각표 권역(ODPT)이 들고 있는 값.
  3. 지어내기            위 둘에서 못 찾으면 이름에서 해시를 떠서 만든다.
                        늘 같은 색이 나오고, 파일에 적히므로 나중에
                        손으로 고칠 수 있다.

사용법: python src/build_colors.py            (없는 것만 채운다)
        python src/build_colors.py --refresh  (지어낸 것을 다시 찾아본다)
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ROOT / "data" / "regions"
OUT = ROOT / "data" / "line-colors.json"
CACHE = ROOT / "data" / "line-colors-wiki.json"

UA = ("rail-isochrone/0.1 (rail colour lookup; "
      "https://github.com/HuijuKim/rail-isochrone)")
API = "https://ja.wikipedia.org/w/api.php"

COLOR_RE = re.compile(r"(?:路線色|ラインカラー|ラインの色)\s*=\s*([^\n|}]+)")
HEX_RE = re.compile(r"#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})\b")
# 위키가 색 이름으로 적어 둔 경우. 일본 노선 표기에 나오는 것만 담았다.
NAMED = {
    "red": "#ff0000", "blue": "#0000ff", "green": "#008000",
    "orange": "#ffa500", "yellow": "#ffff00", "purple": "#800080",
    "pink": "#ffc0cb", "brown": "#a52a2a", "gray": "#808080",
    "grey": "#808080", "black": "#000000", "skyblue": "#87ceeb",
    "navy": "#000080", "teal": "#008080", "olive": "#808000",
    "magenta": "#ff00ff", "cyan": "#00ffff", "lime": "#00ff00",
    "gold": "#ffd700", "silver": "#c0c0c0", "maroon": "#800000",
}

# 이름을 맞출 때 떼어낼 표기
STRIP = re.compile(
    "東日本旅客鉄道|西日本旅客鉄道|東海旅客鉄道|JR東日本|JR西日本|JR東海|ＪＲ|JR|"
    "東京地下鉄|東京メトロ|都営地下鉄|東京都交通局|横浜市営地下鉄|"
    "Osaka Metro|大阪市高速電気軌道|大阪メトロ|神戸市営地下鉄|京都市営地下鉄|"
    "近畿日本鉄道|京浜急行電鉄|京王電鉄|小田急電鉄|東京急行電鉄|東急電鉄|"
    "京成電鉄|西武鉄道|東武鉄道|相模鉄道|山陽電気鉄道|阪神電気鉄道|"
    "阪急電鉄|南海電気鉄道|京阪電気鉄道|神戸電鉄|北大阪急行電鉄|"
    "電気鉄道|急行電鉄|都市モノレール|新交通|電鉄|鉄道|株式会社")
PAREN = re.compile(r"[（(][^）)]*[）)]")
KIND = re.compile(
    "各駅停車|各停|普通|新快速|快速急行|通勤快速|通勤急行|通勤準急|"
    "区間急行|区間快速|快速|特急|急行|準急|直通運転|上り|下り")
PUNCT = re.compile(r"[\s・･:：>=→\-–—]")
# 위키 제목을 맞출 때 쓰는 줄임말. OSM 은 정식 사명을, 위키는 줄임말을 쓴다.
SHORT = [("近畿日本鉄道", "近鉄"), ("京浜急行電鉄", "京急"), ("東京急行電鉄", "東急"),
         ("阪神電気鉄道", "阪神"), ("阪急電鉄", "阪急"), ("南海電気鉄道", "南海"),
         ("京阪電気鉄道", "京阪"), ("山陽電気鉄道", "山陽"), ("西日本鉄道", "西鉄")]


def norm(name: str) -> str:
    """이름을 맞추기 좋게 다듬는다."""
    s = PAREN.sub("", name or "")
    s = STRIP.sub("", s)
    s = KIND.sub("", s)
    return PUNCT.sub("", s)


def made_up(name: str) -> str:
    """못 찾았을 때 쓸 색. 늘 같은 값이 나온다."""
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360.0, 0.42, 0.62)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def wikitext(title: str, cache: dict) -> str:
    if title in cache:
        return cache[title]
    url = (API + "?action=query&prop=revisions&rvprop=content&rvslots=main"
           "&format=json&redirects=1&titles=" + urllib.parse.quote(title))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        page = next(iter(data["query"]["pages"].values()))
        text = (page.get("revisions", [{}])[0]
                .get("slots", {}).get("main", {}).get("*", ""))
    except Exception:
        text = ""
    cache[title] = text
    time.sleep(0.4)
    return text


def color_in(text: str) -> str | None:
    """위키 본문에서 노선 색을 꺼낸다. 틀 첫머리만 본다."""
    m = COLOR_RE.search(text[:12000])
    if not m:
        return None
    raw = m.group(1).strip().strip("{}| ")
    hit = HEX_RE.search(raw)
    if hit:
        v = hit.group(1)
        if len(v) == 3:
            v = "".join(c * 2 for c in v)
        return "#" + v.lower()
    return NAMED.get(raw.lower())


def candidates(name: str) -> list[str]:
    """위키에서 찾아볼 제목들. 먼저 맞는 것을 쓴다."""
    base = PAREN.sub("", name or "").strip()
    base = KIND.sub("", base).strip()
    out = [base]
    for full, short in SHORT:
        if full in base:
            out.append(base.replace(full, short))
    out.append(re.sub(r"^(JR|ＪＲ)", "", base))
    out.append(STRIP.sub("", base))
    # 직통 계통은 이름이 길다. "東京地下鉄の直通運転 - 東急東横線 : 横浜→渋谷"
    # 처럼 여러 토막이 붙어 있어 그대로는 위키에 없다. 구분 기호로 잘라
    # 조각마다 찾아본다. "線" 으로 끝나는 조각이 노선 이름일 확률이 높다.
    parts = [t.strip() for t in re.split(r"[-–—:：・>＞→=]+", base) if t.strip()]
    out += [t for t in parts if t.endswith("線")]
    out += [STRIP.sub("", t) for t in parts if t.endswith("線")]
    out += [t for t in parts if not t.endswith("線")]
    seen, keep = set(), []
    for t in out:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            keep.append(t)
    return keep


# 역 목록이 이만큼 겹치면 같은 노선으로 본다. 권역마다 이름이 달라서
# 이름으로는 맞출 수가 없다. 한쪽 기준이 아니라 합집합에 견준다.
# 짧은 쪽 기준으로 보면 総武快速線 이 総武本線 에 담겨 한 무리가 되는데,
# 둘은 공식 색이 다른 서로 다른 노선이다.
SHARE_MIN = 0.70
SHARE_STATIONS = 3

# 사업자를 가리는 말. 이름만 같고 회사가 다른 노선을 가른다.
# 南北線(도쿄메트로 / 기타오사카급행), 東西線(JR / 도쿄메트로),
# 大師線(게이큐 / 도큐), 奈良線(JR / 긴테쓰) 이 그렇다.
OPERATOR = re.compile(
    "東日本旅客鉄道|西日本旅客鉄道|東海旅客鉄道|JR東日本|JR西日本|JR東海|ＪＲ|JR|"
    "東京地下鉄|東京メトロ|都営地下鉄|東京都交通局|横浜市営地下鉄|"
    "Osaka Metro|大阪市高速電気軌道|大阪メトロ|神戸市営地下鉄|京都市営地下鉄|"
    "近畿日本鉄道|近鉄|京浜急行電鉄|京急|京王電鉄|京王|小田急電鉄|小田急|"
    "東京急行電鉄|東急電鉄|東急|京成電鉄|京成|西武鉄道|西武|東武鉄道|東武|"
    "相模鉄道|相鉄|山陽電気鉄道|阪神電気鉄道|阪神|阪急電鉄|阪急|"
    "南海電気鉄道|南海|京阪電気鉄道|京阪|神戸電鉄|北大阪急行電鉄|北大阪急行|"
    "江ノ島電鉄|江ノ電|湘南モノレール|鹿島臨海鉄道|信楽高原鐵道")


def operator_of(name: str) -> str:
    """이름에 든 사업자. 없으면 빈 문자열."""
    m = OPERATOR.search(name or "")
    return m.group(0) if m else ""


def same_operator(a: str, b: str) -> bool:
    """두 이름이 같은 회사를 가리키는가. 한쪽이라도 안 적혀 있으면 모른다."""
    oa, ob = operator_of(a), operator_of(b)
    if not oa or not ob:
        return True
    return oa == ob or oa in ob or ob in oa


def can_borrow(target: str, source: str) -> bool:
    """target 이 source 의 색을 빌려도 되는가.

    회사가 적힌 이름이 무명 이름의 색을 받으면 안 된다. 표의 南北線 은
    도쿄메트로 것인데, 北大阪急行電鉄南北線 이 電鉄 를 떼면 같은 열쇠가
    되어 그 에메랄드색을 받았다. 반대 방향(東上線 이 東武東上線 의 색을
    받는 것)은 맞으므로 그대로 둔다.
    """
    ot, os_ = operator_of(target), operator_of(source)
    if ot and not os_:
        return False
    return same_operator(target, source)


def qual_key(railway) -> str:
    """사업자를 붙인 열쇠.

    이름만으로는 갈리지 않는 노선이 있다. 간토에는 新宿線 이 둘(都営·西武),
    日光線 이 둘(JR·東武) 있어서 한 칸을 나눠 쓰면 한쪽 색이 다른 쪽을
    덮는다. ODPT 는 id 앞머리가, OSM 은 operator 태그가 회사를 가리킨다.
    """
    ja = railway.get("title", {}).get("ja", "") or railway.get("id", "")
    op = (railway.get("operator") or "").strip()
    if not op:
        op = str(railway.get("id", "")).split(".")[0]
    return (op + "|" + ja) if op else ja


def _lines_by_station():
    """권역별 노선 목록. (권역, 열쇠, 이름, 역 이름 집합)."""
    rows = []
    for d in sorted(REGIONS.iterdir()):
        rp = d / "raw" / "railways.json"
        sp = d / "raw" / "stations.json"
        if not (rp.exists() and sp.exists()):
            continue
        st = {x["id"]: x["title"].get("ja", "")
              for x in json.loads(sp.read_text(encoding="utf-8"))}
        for r in json.loads(rp.read_text(encoding="utf-8")):
            names = {st[x] for x in (r.get("stations") or []) if st.get(x)}
            if len(names) >= SHARE_STATIONS:
                rows.append((d.name, qual_key(r),
                             r.get("title", {}).get("ja", "") or r["id"], names))
    return rows


def sync_regions(table):
    """권역이 달라도 같은 노선이면 같은 색을 쓰게 맞춘다.

    이름을 다듬어 맞추면 안 된다. 두 가지로 다 틀린다. 江ノ島電鉄線 은
    電鉄 를 떼면 小田急 의 江ノ島線 과 같은 열쇠가 되어 에노덴이 오다큐
    파랑을 받았고, 総武快速線 과 横須賀・総武快速線 은 다듬어도 서로
    달라 같은 노선인데 색이 갈렸다. 이름이 아니라 역 목록으로 맞춘다.

    맞춘 무리 안에서는 손으로 적은 색, 그 다음 위키 색을 따른다.
    """
    rows = _lines_by_station()
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i][0] == rows[j][0]:
                continue        # 같은 권역 안의 계통끼리는 합치지 않는다
            if not same_operator(rows[i][2], rows[j][2]):
                continue
            a, b = rows[i][3], rows[j][3]
            share = len(a & b)
            if share >= SHARE_STATIONS and share >= SHARE_MIN * len(a | b):
                pa, pb = find(i), find(j)
                if pa != pb:
                    parent[pb] = pa

    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append((row[1], row[2]))

    def entry(key_name):
        key, name = key_name
        return table.get(key) or table.get(name) or {}

    def rank(key_name):
        src = str(entry(key_name).get("source", ""))
        n = len(key_name[1])
        if src.startswith("손으로"):
            return (0, n)
        if src.startswith("ja.wikipedia"):
            return (1, n)
        if src == "지어냄":
            return (3, n)
        return (2, n)

    fixed = 0
    for members in groups.values():
        members = sorted(set(members))
        if len(members) < 2:
            continue
        lead = min(members, key=rank)
        color = entry(lead).get("color")
        if not color:
            continue
        for key, name in members:
            have = entry((key, name))
            if (key, name) == lead or have.get("color") == color:
                continue
            # 손으로 적은 것만 지킨다. 위키끼리 값이 갈리는 무리가 있어
            # (鹿島線 과 JR鹿島線, 日光線 과 東武日光線) 둘 다 지키면
            # 권역마다 색이 달라지는 그 일이 그대로 남는다. 무리의
            # 대표를 정해 나머지를 맞춘다. 사업자를 붙인 열쇠로 적어,
            # 이름이 같은 남의 노선까지 덮지 않게 한다.
            if str(have.get("source", "")).startswith("손으로"):
                continue
            table[key] = {"color": color, "source": "권역 맞춤:" + lead[1]}
            fixed += 1
    print(f"  권역끼리 역 목록으로 맞춘 노선 무리 "
          f"{sum(1 for v in groups.values() if len(set(v)) > 1)}개, "
          f"색을 맞춘 노선 {fixed}개")


def main() -> None:
    refresh = "--refresh" in sys.argv
    table = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    # 낡은 꼴(문자열)도 받아 준다
    for k, v in list(table.items()):
        if isinstance(v, str):
            table[k] = {"color": v, "source": "지어냄"}

    lines, official, exact = {}, {}, {}
    for d in sorted(REGIONS.iterdir()):
        path = d / "raw" / "railways.json"
        if not path.exists():
            continue
        for r in json.loads(path.read_text(encoding="utf-8")):
            ja = r.get("title", {}).get("ja", "") or r["id"]
            got = r.get("color") or r.get("colour")
            if got:
                exact.setdefault(ja, (got, ja))
                official.setdefault(norm(ja), (got, ja))
            lines[ja] = r["id"]
    print(f"노선 {len(lines)}개 (원본에 색이 있는 것 {len(official)}개)", flush=True)

    from_have = from_wiki = invented = kept = 0
    for ja in sorted(lines):
        have = table.get(ja)
        if have and str(have.get("source", "")).startswith("손으로"):
            kept += 1
            continue
        if have and (not refresh or have.get("source") != "지어냄"):
            kept += 1
            continue

        # 위키백과를 먼저 본다. 원본(ODPT)은 mini-tokyo-3d 가 직접
        # 관리하는 파일이라 운영사 공식값이 아닌 것이 섞여 있다
        # (ニューシャトル 이 #FFFF00, 간토철도 두 노선이 #0000FF 처럼
        # 순색이다). 무엇보다 한 노선을 두 이름으로 부를 때 한쪽은 ODPT,
        # 한쪽은 위키를 타면 색이 갈린다. 출처를 하나로 두면 없어진다.
        # 손으로 적어 둔 것은 위에서 이미 건너뛴다.
        found = None
        for title in candidates(ja):
            got = color_in(wikitext(title, cache))
            if got:
                found = (got, title)
                break
        if found:
            table[ja] = {"color": found[0], "source": "ja.wikipedia:" + found[1]}
            from_wiki += 1
            continue

        # 위키에 없으면 원본이 들고 있는 색. 이름이 똑같은 것을 먼저 본다.
        # 다듬은 이름으로만 맞추면 江ノ島電鉄線 이 電鉄 를 떼면서 小田急 의
        # 江ノ島線 과 같은 열쇠가 되어 에노덴에 오다큐 색이 붙는다.
        key = norm(ja)
        hit = exact.get(ja)
        if hit is None:
            got = official.get(key)
            if got and can_borrow(ja, got[1]):
                hit = got
        if hit is None:
            # 한쪽이 다른 쪽을 품는 경우(東上線 과 東武東上線). 회사가
            # 다르면 보지 않는다. 北大阪急行電鉄南北線 이 電鉄 를 떼면
            # 東京メトロ南北線 과 같은 열쇠가 되어 그 색을 받았다.
            for k, v in official.items():
                if k and (k in key or key in k) and can_borrow(ja, v[1]):
                    hit = v
                    break
        if hit:
            table[ja] = {"color": hit[0], "source": "ODPT:" + hit[1]}
            from_have += 1
        else:
            table[ja] = {"color": made_up(ja), "source": "지어냄"}
            invented += 1

    sync_regions(table)

    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    OUT.write_text(json.dumps(dict(sorted(table.items())), ensure_ascii=False,
                              indent=1), encoding="utf-8")
    print(f"  그대로 둔 것 {kept}개, 원본 색을 쓴 것 {from_have}개, "
          f"위키에서 찾은 것 {from_wiki}개, 지어낸 것 {invented}개")
    print(f"  노선 색 {len(table)}개 -> {OUT} "
          f"({OUT.stat().st_size / 1024:.0f} KB)")
    left = sorted(k for k, v in table.items() if v.get("source") == "지어냄")
    if left:
        print(f"\n  아직 못 찾은 {len(left)}개 (손으로 고칠 수 있다)")
        for k in left[:20]:
            print(f"    {k}")


if __name__ == "__main__":
    main()
