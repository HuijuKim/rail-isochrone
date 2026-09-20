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

# 열차 종별. 빠른 것부터 본다. 먼저 맞는 것을 종별로 삼는다.
# ライナー 는 넣지 않는다. ポートライナー・六甲ライナー 같은 신교통 노선
# 이름과 구별되지 않아 사람 나르는 궤도를 특급으로 만들어 버린다.
KINDS = [
    ("특급", r"特急|Limited Express"),
    ("통근쾌속", r"通勤快速|通勤特快|通勤急行|通勤準急|区間急行|区間快速"),
    ("쾌속", r"新快速|快速"),
    ("급행", r"急行"),
    ("준급", r"準急"),
    ("각역정차", r"各駅停車|各停|普通|Local"),
]
KIND_RE = re.compile("|".join(p for _, p in KINDS))

SHINKANSEN = re.compile(
    r"新幹線|Shinkansen|のぞみ|ひかり|こだま|みずほ|さくら|つばめ|"
    r"はやぶさ|はやて|こまち|やまびこ|なすの|とき|たにがわ|"
    r"かがやき|はくたか|つるぎ|あさま|かもめ")

PAREN_RE = re.compile(r"[（(][^）)]*[）)]")
DIR_RE = re.compile(r"(上り|下り|内回り|外回り|環状)")
ANGLE_RE = re.compile(r"[〈《<][^〉》>]*[〉》>]")

SAME_STATION_M = 400.0      # 이 안에 있고 이름이 같으면 한 역으로 본다
TRACK_NEAR_M = 150.0        # 선로에서 이만큼 안이면 그 노선의 역으로 본다
SAME_LINE = 0.70            # 정차역이 이만큼 담기면 같은 선로의 다른 계통
DUP_LINE = 0.90             # 이만큼 같으면 중복 노선
MIN_WAYS = 5                # 선로에서 역을 찾아볼 최소 웨이 수


def kind_of(name):
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


class Relations(osmium.SimpleHandler):
    """철도 계통 관계를 모은다. 정차역이 없으면 선로 웨이를 챙겨 둔다."""

    def __init__(self):
        super().__init__()
        self.routes = []
        self.want_nodes = set()
        self.want_ways = set()
        self.dropped = 0

    def relation(self, r):
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
        stops = [m.ref for m in r.members
                 if m.type == "n" and m.role in ("stop", "stop_entry_only",
                                                 "stop_exit_only")]
        if not stops:
            stops = [m.ref for m in r.members if m.type == "n" and m.role == "platform"]
        ways = [m.ref for m in r.members if m.type == "w"]
        if len(stops) < 2 and len(ways) < MIN_WAYS:
            return
        self.want_nodes.update(stops)
        if len(stops) < 2:
            self.want_ways.update(ways)
        self.routes.append({
            "name": name,
            "titles": {g: t.get("name:" + g, "") for g in LANGS},
            "operator": t.get("operator", "") or t.get("network", ""),
            "ref": t.get("ref", ""),
            "colour": t.get("colour", ""),
            "kind": kind_of(name),
            "stops": stops,
            "ways": ways if len(stops) < 2 else [],
        })


class Ways(osmium.SimpleHandler):
    """선로 웨이의 좌표열."""

    def __init__(self, want):
        super().__init__()
        self.want = want
        self.geom = {}

    def way(self, w):
        if w.id not in self.want:
            return
        try:
            self.geom[w.id] = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        except osmium.InvalidLocationError:
            pass


class StationNodes(osmium.SimpleHandler):
    """역 노드 전부와, 관계가 가리킨 정차 노드."""

    def __init__(self, want):
        super().__init__()
        self.want = want
        self.pos = {}
        self.stations = {}

    def node(self, n):
        t = n.tags
        is_station = (t.get("railway") in STATION_TAGS
                      or t.get("public_transport") == "station")
        if not is_station and n.id not in self.want:
            return
        rec = (n.location.lon, n.location.lat,
               {g: t.get("name:" + g, "") for g in LANGS}, t.get("name", ""))
        self.pos[n.id] = rec
        if is_station and (rec[2].get("ja") or rec[3]):
            self.stations[n.id] = rec


def stitch(ways, geom):
    """웨이를 이어 하나의 경로로. 멤버 순서를 믿되 방향만 맞춘다."""
    path = []
    for wid in ways:
        pts = geom.get(wid)
        if not pts or len(pts) < 2:
            continue
        if path:
            a = (pts[0][0] - path[-1][0]) ** 2 + (pts[0][1] - path[-1][1]) ** 2
            b = (pts[-1][0] - path[-1][0]) ** 2 + (pts[-1][1] - path[-1][1]) ** 2
            if b < a:
                pts = pts[::-1]
        path.extend(pts)
    return path


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


def overlap(small, big):
    if not small:
        return 0.0
    s = set(big)
    return sum(1 for c in small if c in s) / len(small)


def main():
    pbf = sys.argv[1] if len(sys.argv) > 1 else str(
        ROOT / "data" / "osm" / (REGION + "-latest.osm.pbf"))
    print("[" + REGION + "] " + Path(pbf).name + " 읽는 중...", flush=True)

    rel = Relations()
    rel.apply_file(pbf)
    track_only = sum(1 for r in rel.routes if not r["stops"])
    print(f"  철도 계통 관계 {len(rel.routes):,}개 (신칸센 {rel.dropped}개 제외), "
          f"선로만 있는 것 {track_only:,}개", flush=True)

    ways = Ways(rel.want_ways)
    ways.apply_file(pbf, locations=True, idx="flex_mem")
    print(f"  선로 웨이 {len(ways.geom):,}개", flush=True)

    nodes = StationNodes(rel.want_nodes)
    nodes.apply_file(pbf)
    print(f"  역 노드 {len(nodes.stations):,}개, 정차 노드 {len(nodes.pos):,}개",
          flush=True)

    lat0 = float(np.median([v[1] for v in nodes.pos.values()])) if nodes.pos else 35.0
    scale = float(np.cos(np.radians(lat0)))

    # 선로만 있는 관계는 선로 옆 역에서 정차 순서를 복원한다.
    # 통과 계통에는 쓸 수 없다. 선로 옆 역을 모두 주우면 통과하는 역까지
    # 정차로 만들어, 특급이 각역정차로 둔갑하고 그 긴 목록이 진짜 노선을
    # 삼켜 버린다. 종별이 적힌 관계는 정차역 멤버가 있을 때만 쓴다.
    recovered = 0
    for r in rel.routes:
        if len(r["stops"]) >= 2 or r["kind"] not in (None, "각역정차"):
            continue
        seq = stops_along(stitch(r["ways"], ways.geom), nodes.stations, scale)
        if len(seq) >= 2:
            r["stops"] = seq
            recovered += 1
    print(f"  선로에서 정차 순서를 되살린 관계 {recovered:,}개", flush=True)

    used = {n for r in rel.routes for n in r["stops"]}
    pos = {n: nodes.pos[n] for n in used if n in nodes.pos}
    of, members = cluster_nodes(pos)
    print(f"  역 묶음 {len(members):,}개", flush=True)

    for r in rel.routes:
        seq = []
        for nid in r["stops"]:
            c = of.get(nid)
            if c is not None and (not seq or seq[-1] != c):
                seq.append(c)
        r["seq"] = seq
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
    print(f"  뼈대 노선 {len(lines):,}개, 통과·부분 계통 {len(patterns):,}개", flush=True)

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

        order = []
        for cl in ln["seq"]:
            sid = lid + "." + str(cl)
            if sid not in seen_sid:
                seen_sid.add(sid)
                lon, lat, titles, nm = pos[members[cl][0]]
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
    print("  저장 -> " + str(OUT))


if __name__ == "__main__":
    main()
