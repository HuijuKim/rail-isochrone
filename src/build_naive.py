"""시각표가 없는 권역을 위해 가상 시각표를 짓는다.

간사이·주부에는 공개 시각표가 없다. 그렇다고 라우터를 따로 만들면 화면과
서버를 통째로 두 벌 유지해야 한다. 대신 배차를 추정해 그 간격으로 열차를
깔아 기존 그래프 형식으로 저장한다. 그 뒤로는 간토와 똑같이 돌아간다.

덤이 둘 있다. 출발 시각이 의미를 갖게 되고, 대기 시간을 "배차의 절반"
같은 어림값이 아니라 RAPTOR 가 실제 열차 시각으로 계산한다.

추정하는 것은 둘뿐이고 전부 좌표에서 나온다.

  주행 시간   72.8초 + 거리 x 0.04436     역당 정차 73초, 평균 81km/h
  배차        구간 주변 역 밀도로 매긴 등급

간토 시각표로 맞춘 값이다. 등급은 노선이 아니라 구간마다 매긴다. 노선
단위로 매기면 東海道本線 처럼 米原에서 神戸까지 한 관계에 들어 있는
노선에서 도심과 시골이 한 값으로 뭉개진다.

통과 계통(쾌속·급행·특급)은 같은 역 줄 위를 건너뛰며 달리는 별도 운행으로
깐다. 그래야 라우터가 "쾌속을 기다릴지 각역정차를 탈지" 를 시각으로 푼다.

사용법: REGION=kansai python src/build_naive.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REGION = os.environ.get("REGION", "kansai")
BASE = ROOT / "data" / "regions" / REGION
RAW = BASE / "raw"

LANGS = ("ja", "en", "ko", "zh-Hans", "zh-Hant")

# 간토 시각표로 맞춘 주행식. 상수항이 정차와 가감속을, 기울기가 순항
# 속도를 담는다. 통과 열차는 최고속도가 높고 감속이 적어 따로 맞춘다.
#   각역정차  62.1초 + 74.8km/h        통과  106.0초 + 84.8km/h
RIDE_LOCAL = (62.1, 1.0 / (74.8 / 3.6))
RIDE_FAST = (106.0, 1.0 / (84.8 / 3.6))
# 선로 기하가 없는 구간만 직선거리에 이 배수를 곱해 어림한다. 기하가
# 있으면 실제 선로를 따라 잰다. 오사카-교토는 직선 41.9km 에 실제 선로
# 42.8km 라 배수가 1.02 인데, 1.15 를 곱하면 48km 가 되어 7분이 붙는다.
DETOUR = 1.15

# 등급별 대표 배차 (분). 간토 155개 노선의 실측 중앙값이다.
REP = [5.0, 8.8, 17.8, 39.3]
GRADE_NAMES = ["도심", "장거리통근", "적당한로컬", "한적한로컬"]
HEADWAY_CAP = 60.0          # 한 시간 넘는 배차는 여기서 자른다
EXP_MULT = 2.0              # 통과 계통은 각역정차보다 드물다

SERVICE_FROM, SERVICE_TO = 5 * 3600, 24 * 3600

# 같은 역 구내 환승(플랫폼 이동). 걸어서 갈아타는 이웃 역의 값은
# transfers.py 가 거리에서 매긴다.
TRANSFER_SAME = 180

SUBWAY = re.compile(r"地下鉄|Subway|メトロ|市営|Osaka Metro")
# 이름에 종별이 박힌 노선. 新快速 처럼 통과 계통이 그대로 뼈대가 된 경우다.
# 사업자 이름 안의 글자에 걸리면 안 된다. 北大阪急行電鉄南北線 과
# 京浜急行電鉄大師線 이 각역정차인데 급행으로 분류돼 배차가 두 배가
# 됐다. 急行 뒤에 사업자·노선 글자가 오면 넘긴다(富士急行線·伊豆急行線
# 도 같은 자리다). 快速 은 사업자 이름에 안 들어가므로 그대로 둔다.
# 総武快速線 처럼 이름이 快速線 으로 끝나는 진짜 쾌속이 있어서다.
#
# ライナー 는 뺐다. 세 권역에서 이 글자가 든 노선은 ポートライナー,
# 六甲ライナー, 日暮里・舎人ライナー 뿐이고 전부 각역정차하는 신교통
# 노선 이름이다. 진짜 홈라이너 계통은 express.json 이 종별로 들고 온다.
FAST_NAME = re.compile(r"新快速|快速|特急|急行(?!電鉄|鉄道|線)|準急")


def load_raw():
    railways = json.loads((RAW / "railways.json").read_text(encoding="utf-8"))
    stations = json.loads((RAW / "stations.json").read_text(encoding="utf-8"))
    # OSM 에서 뽑은 것과 위키백과에서 읽은 것을 합친다. 후자를 따로 두는
    # 것은 build_rail.py 가 express.json 을 새로 쓰기 때문이다. 한 파일에
    # 섞으면 노선을 다시 뽑을 때마다 날아간다.
    express = json.loads((RAW / "express.json").read_text(encoding="utf-8"))
    wiki_path = RAW / "express-wiki.json"
    if wiki_path.exists():
        wiki = json.loads(wiki_path.read_text(encoding="utf-8"))
        have = {(e["railway"], e["kind"]) for e in express}
        express += [e for e in wiki if (e["railway"], e["kind"]) not in have]
    return railways, stations, express


def cluster_table(stations):
    pos, title = {}, {}
    for s in stations:
        c = s["cluster"]
        if c not in pos:
            pos[c] = (float(s["coord"][0]), float(s["coord"][1]))
            title[c] = {g: s["title"].get(g, "") for g in LANGS}
    return pos, title


def _line_headways():
    """data/line-headways.json. 권역 -> 노선 일본어 이름 -> 한낮 한 방향 편수.

    규칙(역 밀도·간격)으로 매긴 배차가 실제와 크게 어긋나는 노선만 적는다.
    출처를 함께 적는다. min_per_hour 는 하한, max_per_hour 는 상한이다.
    """
    path = ROOT / "data" / "line-headways.json"
    if not path.exists():
        return {}
    got = json.loads(path.read_text(encoding="utf-8")).get(REGION) or {}
    return {k: v for k, v in got.items() if isinstance(v, dict)}


def grade_segments(railways, pos, scale):
    """구간마다 등급을 매겨 배차를 정한다. 좌표만 쓴다."""
    cl = sorted(pos)
    gx = np.array([pos[c][0] for c in cl]) * scale * 111.320
    gy = np.array([pos[c][1] for c in cl]) * 111.132

    def dens_at(lon, lat):
        d = np.hypot(gx - lon * scale * 111.320, gy - lat * 111.132)
        return int((d <= 5.0).sum())

    def km(a, b):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        return float(np.hypot((x2 - x1) * scale * 111.320, (y2 - y1) * 111.132))

    hand = _line_headways()
    seg_head, seg_km, line_grade = {}, {}, {}
    for r in railways:
        cs = [c for c in r["clusters"] if c in pos]
        if len(cs) < 2:
            continue
        title = r["title"].get("ja", "")
        spec = hand.get(title) or {}
        sub = bool(SUBWAY.search(title + " " + (r.get("operator") or "")))
        fast = bool(FAST_NAME.search(title))
        grades = []
        for a, b in zip(cs, cs[1:]):
            (x1, y1), (x2, y2) = pos[a], pos[b]
            d5 = dens_at((x1 + x2) / 2, (y1 + y2) / 2)
            L = km(a, b)
            if fast:
                # 통과 계통은 역을 건너뛰므로 역간 거리가 노선의 성격을
                # 말해 주지 않는다. 그 자리의 역 밀도만 본다.
                g = 0 if d5 >= 40 else 1 if d5 >= 15 else 2 if d5 >= 8 else 3
            elif sub or (L < 2.0 and d5 >= 40):
                g = 0
            elif d5 < 8:
                # 예전에는 역 간격 4km 넘는 구간도 여기로 보냈다. 역이
                # 듬성한 간선(시즈오카의 東海道本線)이 시골 노선이 됐다.
                # 빼도 간토 실측과의 소요 시간 비는 0.99 -> 0.98 로 같다.
                g = 3
            elif L < 3.0 and d5 >= 15:
                g = 1
            else:
                g = 2
            grades.append(g)
            h = min(REP[g] * (EXP_MULT if fast else 1.0), HEADWAY_CAP)
            # 실제 운행량으로 확인한 노선은 그 값으로 누르거나 올린다.
            # 하한은 이보다 잦게 매긴 구간(도심 쪽)을 건드리지 않는다.
            if spec.get("min_per_hour"):
                h = min(h, 60.0 / spec["min_per_hour"])
            if spec.get("max_per_hour"):
                h = max(h, 60.0 / spec["max_per_hour"])
            key = (a, b)
            if h < seg_head.get(key, np.inf):
                seg_head[key] = seg_head[(b, a)] = h
            d = L * DETOUR
            seg_km[key] = seg_km[(b, a)] = d
        line_grade[r["id"]] = int(np.median(grades))
    return seg_head, seg_km, line_grade, km


def runs_of(values):
    """같은 값이 이어지는 구간을 (시작, 끝, 값) 으로 자른다."""
    out, i = [], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        out.append((i, j + 1, values[i]))
        i = j + 1
    return out


def phase_of(key, headway_sec):
    """계통마다 다른 출발 위상. 안 주면 환승이 비현실적으로 잘 맞는다."""
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h % max(int(headway_sec), 1))


def track_lengths(rows, row_of, railways, pos, scale, km):
    """역 사이 거리를 실제 선로를 따라 잰다.

    직선거리에 일정한 배수를 곱하면 곧은 회랑에서 크게 부풀어 오른다.
    선형이 없는 구간만 어림으로 메운다.
    """
    from geometry import Geometry

    coords = np.array([pos[c] for _rid, c in rows], dtype=np.float64)
    geo = Geometry(RAW / "coordinates.json", [rid for rid, _c in rows], coords)

    out, traced = {}, 0
    for r in railways:
        cs = [c for c in r["clusters"] if c in pos]
        for a, b in zip(cs, cs[1:]):
            key = (a, b)
            if key in out:
                continue
            straight = km(a, b)
            path = geo.ride_path([row_of[(r["id"], a)], row_of[(r["id"], b)]],
                                 r["id"])
            d = straight * DETOUR
            if len(path) >= 2:
                p = np.asarray(path, dtype=np.float64)
                dx = (p[1:, 0] - p[:-1, 0]) * scale * 111.320
                dy = (p[1:, 1] - p[:-1, 1]) * 111.132
                along = float(np.hypot(dx, dy).sum())
                # 엉뚱한 선형을 고르면 크게 돌아간다. 직선의 두 배를
                # 넘으면 믿지 않는다.
                if straight * 0.9 <= along <= straight * 2.0:
                    d, traced = along, traced + 1
            out[key] = out[(b, a)] = d
    print(f"  선로를 따라 잰 구간 {traced:,}개 / {len(out) // 2:,}개", flush=True)
    return out


def build(railways, express, pos, seg_head, seg_km, km, scale):
    """역 줄과 정차 이벤트를 만든다."""
    # 역 줄은 (노선, 역묶음) 단위. 간토 자료도 이렇게 쪼개져 있다.
    rows, row_of = [], {}
    for r in railways:
        for c in r["clusters"]:
            if c not in pos:
                continue
            key = (r["id"], c)
            if key not in row_of:
                row_of[key] = len(rows)
                rows.append(key)

    seg_km = track_lengths(rows, row_of, railways, pos, scale, km)

    ev_stop, ev_arr, ev_dep, trip_start = [], [], [], []

    def lay(line_key, rid, cs, headway_min, spans_km, fast=False):
        """cs 를 순서대로 도는 운행을 배차 간격으로 깐다."""
        ride_a, ride_b = RIDE_FAST if fast else RIDE_LOCAL
        step = max(int(round(headway_min * 60)), 60)
        offset = phase_of(line_key, step)
        # 왕복 모두 깐다. 한 방향만 깔면 되돌아오는 경로가 없어진다.
        # 첫 차는 운행 시작 시각에 이미 노선 전체를 달리고 있어야 한다.
        # 출발역에서 05:00 에 처음 떠나게 두면 긴 노선의 먼 끝은 한낮에야
        # 첫 차가 온다. 日豊本線(小倉-鹿児島中央 462km)은 宮崎 에 11시 반에
        # 첫 차가 와서 宮崎-都城 이 271분으로 나왔다.
        whole = sum(int(round(ride_a + ride_b * km * 1000.0)) for km in spans_km)
        lead = -(-whole // step) * step
        for seq, legs in ((cs, spans_km), (cs[::-1], spans_km[::-1])):
            t0 = SERVICE_FROM + offset - lead
            while t0 <= SERVICE_TO:
                trip_start.append(len(ev_stop))
                t = t0
                for i, c in enumerate(seq):
                    ev_stop.append(row_of[(rid, c)])
                    ev_arr.append(t)
                    ev_dep.append(t)
                    if i < len(legs):
                        t += int(round(ride_a + ride_b * legs[i] * 1000.0))
                t0 += step
            # 반대 방향은 위상을 절반 어긋나게 둔다
            offset = (offset + step // 2) % step

    hand = _line_headways()
    capped = set()
    for r in railways:
        cs = [c for c in r["clusters"] if c in pos]
        if len(cs) < 2:
            continue
        legs = [seg_km.get((a, b), km(a, b) * DETOUR) for a, b in zip(cs, cs[1:])]
        heads = [seg_head.get((a, b), HEADWAY_CAP) for a, b in zip(cs, cs[1:])]
        # 구간 배차는 노선끼리 나눠 쓰므로(가장 잦은 값), 상한은 여기서 노선마다
        # 다시 건다. 上飯田線 은 같은 구간을 지나는 小牧線 의 배차를 받아 상한을
        # 넘었다. 상한을 둔 노선은 한적한 노선이라 통과 계통도 깔지 않는다.
        # 瀬戸線 은 한낮에 普通 만 다니는데 급행 계통이 얹혀 1.5배가 됐다.
        spec = hand.get(r["title"].get("ja", "")) or {}
        if spec.get("max_per_hour"):
            heads = [max(h, 60.0 / spec["max_per_hour"]) for h in heads]
            capped.add(r["id"])

        # 노선 전체는 가장 드문 구간에 맞춰 깐다. 그보다 잦은 구간에는
        # 짧게 도는 운행을 덧댄다. 빈도는 더해지므로 덧대는 간격은
        # 1/h_dense - 1/h_full 의 역수다.
        fast = bool(FAST_NAME.search(r["title"].get("ja", "")))
        full = max(heads)
        lay(r["id"] + "|full", r["id"], cs, full, legs, fast=fast)
        for i, j, h in runs_of(heads):
            if h >= full - 1e-6:
                continue
            extra = 1.0 / max(1.0 / h - 1.0 / full, 1e-6)
            if extra > HEADWAY_CAP:
                continue
            lay(f"{r['id']}|{i}", r["id"], cs[i:j + 1], extra, legs[i:j], fast=fast)

    # 통과 계통. 같은 역 줄 위를 건너뛰며 달린다.
    # 한 역이 두 번 실린 관계에서는 첫 자리를 쓴다. 마지막 자리를 쓰면
    # 쾌속 구간이 노선 절반을 가로지른다. 関西本線 木津->加茂 가 실제
    # 5.19km 인데 21홉 94km, 4분이 68분으로 잡혔다.
    order_of = {}
    for r in railways:
        first = {}
        for k, c in enumerate(r["clusters"]):
            first.setdefault(c, k)
        order_of[r["id"]] = first
    clusters_of = {r["id"]: r["clusters"] for r in railways}
    n_exp = 0
    for n, e in enumerate(express):
        if e["kind"] == "부분":
            continue
        rid = e["railway"]
        if rid not in order_of or rid in capped:
            continue
        cs = [c for c in e["clusters"] if c in pos and (rid, c) in row_of]
        if len(cs) < 2:
            continue
        ord_, full_cs = order_of[rid], clusters_of[rid]
        legs, unders = [], []
        for a, b in zip(cs, cs[1:]):
            if a in ord_ and b in ord_:
                i, j = sorted((ord_[a], ord_[b]))
                span = list(zip(full_cs[i:j], full_cs[i + 1:j + 1]))
                legs.append(sum(seg_km.get(p, km(*p) * DETOUR) for p in span))
                unders += [seg_head[p] for p in span if p in seg_head]
            else:
                legs.append(km(a, b) * DETOUR)
        base = float(np.median(unders)) if unders else 20.0
        lay(f"{rid}|exp{n}", rid, cs, min(base * EXP_MULT, HEADWAY_CAP), legs, fast=True)
        n_exp += 1

    trip_start.append(len(ev_stop))
    return rows, row_of, ev_stop, ev_arr, ev_dep, trip_start, n_exp


def transfers(rows, pos, scale, title):
    """같은 역 구내와, 걸어서 갈아타는 이웃 역 사이."""

    def title_of(c):
        return (title.get(c) or {}).get("ja", "")

    by_cluster = defaultdict(list)
    for i, (_rid, c) in enumerate(rows):
        by_cluster[c].append(i)

    edges = []
    for _c, ids in by_cluster.items():
        for a in ids:
            for b in ids:
                if a != b:
                    edges.append((a, b, TRANSFER_SAME))

    # 이름이 달라 안 묶인 이웃 역은 걸어서 갈아탄다. 거리에 따라 값을
    # 매긴다. transfers.py 에 규칙을 두어 시각표 권역과 같은 잣대를 쓴다.
    from router import WALK_SPEED
    import transfers as xfer

    cl = sorted(by_cluster)
    pairs = xfer.near_pairs([pos[c][0] for c in cl], [pos[c][1] for c in cl],
                            WALK_SPEED)
    near = 0
    for i, j, cost in pairs:
        # 같은 이름이면 위에서 이미 한 묶음이다
        if title_of(cl[i]) == title_of(cl[j]):
            continue
        near += 1
        for a in by_cluster[cl[i]]:
            for b in by_cluster[cl[j]]:
                edges.append((a, b, cost))
                edges.append((b, a, cost))
    return edges, near


def main():
    railways, stations, express = load_raw()
    pos, title = cluster_table(stations)
    lat0 = float(np.median([p[1] for p in pos.values()]))
    scale = float(np.cos(np.radians(lat0)))
    print(f"[{REGION}] 노선 {len(railways):,}개, 역 묶음 {len(pos):,}개", flush=True)

    seg_head, seg_km, line_grade, km = grade_segments(railways, pos, scale)
    counts = defaultdict(int)
    for g in line_grade.values():
        counts[g] += 1
    print("  등급: " + ", ".join(f"{GRADE_NAMES[g]} {counts[g]}" for g in sorted(counts)),
          flush=True)

    rows, row_of, ev_stop, ev_arr, ev_dep, trip_start, n_exp = build(
        railways, express, pos, seg_head, seg_km, km, scale)
    print(f"  역 줄 {len(rows):,}개, 운행 {len(trip_start) - 1:,}건, "
          f"정차 이벤트 {len(ev_stop):,}개 (통과 계통 {n_exp}개)", flush=True)

    edges, near = transfers(rows, pos, scale, title)
    print(f"  환승 간선 {len(edges):,}개 (이름이 다른 이웃 역 쌍 {near:,}개)", flush=True)

    tr = np.array(edges, dtype=np.int64)
    tr = tr[np.lexsort((tr[:, 1], tr[:, 0]))]
    tr_ptr = np.searchsorted(tr[:, 0], np.arange(len(rows) + 1))

    # 지도에 찍는 자리는 노선마다 제 승강장이다. 묶음 좌표를 쓰면
    # 이름만 같고 승강장이 다른 역에서 선이 남의 자리로 끌려간다.
    # 묶음은 그대로 두어 검색과 세는 단위는 한 역로 남긴다.
    at = {(st["railway"], st["cluster"]): st["coord"] for st in stations
          if "railway" in st and "cluster" in st}
    coords = np.array([at.get((rid, c)) or pos[c] for rid, c in rows],
                      dtype=np.float64)
    ids = [f"{rid}.{c}" for rid, c in rows]

    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "stops.json").write_text(json.dumps({
        "ids": ids,
        "coords": coords.tolist(),
        **{g: [title[c].get(g, "") for _rid, c in rows] for g in LANGS},
        "railway": [rid for rid, _c in rows],
        # 검색과 세기는 이 묶음 번호로 합친다
        "cluster": [int(c) for _rid, c in rows],
    }, ensure_ascii=False), encoding="utf-8")

    # 평일과 휴일을 가를 근거가 없다. 같은 것을 두 벌 쓴다.
    for calendar in ("Weekday", "SaturdayHoliday"):
        np.savez_compressed(
            BASE / f"graph-{calendar}.npz",
            coords=coords,
            ev_stop=np.array(ev_stop, dtype=np.int32),
            ev_arr=np.array(ev_arr, dtype=np.int32),
            ev_dep=np.array(ev_dep, dtype=np.int32),
            trip_start=np.array(trip_start, dtype=np.int32),
            tr_to=tr[:, 1].astype(np.int32),
            tr_cost=tr[:, 2].astype(np.int32),
            tr_ptr=tr_ptr.astype(np.int64),
        )
        size = (BASE / f"graph-{calendar}.npz").stat().st_size
        print(f"  graph-{calendar}.npz {size / 1e6:.1f} MB", flush=True)

    # 역 묶음 파일은 역 줄 id 로 다시 적는다. 검색과 세기가 이걸 쓴다.
    by_cluster = defaultdict(list)
    for i, (rid, c) in enumerate(rows):
        by_cluster[c].append(ids[i])
    (RAW / "station-groups.json").write_text(
        json.dumps([[g] for _c, g in sorted(by_cluster.items())], ensure_ascii=False),
        encoding="utf-8")
    print(f"  역 묶음 파일 {len(by_cluster):,}개")
    print("  저장 -> " + str(BASE))


if __name__ == "__main__":
    main()
