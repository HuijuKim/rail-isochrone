"""역과 역 사이를 실제 선로를 따라 잇는다.

지금까지는 노선 선형을 route 관계가 가리킨 웨이에서 받아왔다. 그런데
관계는 사람이 손으로 채우는 것이라 빠진 데가 많다. 京成本線 은 우에노로
들어가는 지하 구간 웨이가 관계에 없어서 선이 터널 입구에서 끊긴다.
지하철은 전반적으로 더 성기다.

선로 자체는 OSM 에 거의 다 들어 있다. 닛포리에서 우에노 사이 2km 상자
안에만 철도 웨이가 411개, 그중 터널이 135개다. 그래서 관계를 보지 않고
선로를 그래프로 만들어 역과 역 사이 최단 경로를 찾는다. 그 경로가 곧
그 구간의 선형이다.

다만 전체 선로를 한 덩어리로 두면 안 된다. 지하철은 환승역에서 선로가
이어져 있어, 한조몬선이 스이텐구마에에서 기요스미시라카와로 갈 때
오에도선 쪽으로 크게 돌아 버린다. 선로 웨이에는 대개 노선 이름이 붙어
있으므로, 먼저 제 노선 선로 안에서 찾고 거기서 못 찾을 때만 전체에서
찾는다.

사용법: REGION=kanto_osm python src/build_track.py
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
import sys
from pathlib import Path

import numpy as np
import osmium

sys.path.insert(0, str(Path(__file__).resolve().parent))
import osmcache  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGION = os.environ.get("REGION", "kanto_osm")
BASE = ROOT / "data" / "regions" / REGION
RAW = BASE / "raw"

RAIL_KINDS = ("rail", "subway", "light_rail", "monorail", "narrow_gauge",
              "tram", "funicular")
# 역을 선로에 붙일 때 이만큼까지 본다. 플랫폼 노드가 선로에서 떨어져 있다.
SNAP_M = 400.0
SNAP_TRIES = 12             # 가까운 선로 노드 몇 개까지 시도할지
# 찾은 경로가 직선거리의 이 배를 넘으면 엉뚱한 선로로 샌 것으로 본다.
# geometry.py 가 2.5 배를 넘는 호를 아예 버리므로 그보다 넉넉하게 잡으면
# 애써 찾은 경로가 그쪽에서 버려지고 더 거친 선형이 대신 뽑힌다.
MAX_RATIO = 2.4
MIN_CAP_M = 2500.0
# 스위치백은 되짚어 올라가므로 선로가 직선의 몇 배가 된다. 木次線 出雲坂根 은
# 3단이라 직선 1.9km 에 선로 6.4km 다. 사전(data/switchbacks.json)에 적힌 역이
# 걸린 구간만 한도를 늘린다.
SWITCHBACK_RATIO = 6.0
SWITCHBACK_MIN_M = 8000.0


def _switchbacks() -> dict:
    path = ROOT / "data" / "switchbacks.json"
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: set(v) for k, v in got.items() if isinstance(v, list)}

# 웨이 이름에서 떼어낼 사업자·종별 표기. 半蔵門線 과 東京メトロ半蔵門線 을
# 같은 것으로 보기 위한 것이다.
STRIP = re.compile(
    "東京地下鉄|東京メトロ|都営地下鉄|東京都交通局|横浜市営地下鉄|"
    "Osaka Metro|大阪市高速電気軌道|大阪メトロ|神戸市営地下鉄|"
    "京都市営地下鉄|東日本旅客鉄道|西日本旅客鉄道|東海旅客鉄道|"
    "JR東日本|JR西日本|JR東海|ＪＲ|JR|電鉄|鉄道|株式会社|"
    "各駅停車|各停|普通|新快速|快速急行|通勤快速|通勤急行|通勤準急|"
    "区間急行|区間快速|快速|特急|急行|準急|上り|下り|内回り|外回り")
DROP = re.compile("[（(\\[][^）)\\]]*[）)\\]]")
PUNCT = re.compile("[\\s・･:：>=→\\-–—]")


def norm_line(name):
    """노선 이름을 견주기 좋게 다듬는다."""
    s = STRIP.sub("", name or "")
    s = DROP.sub("", s)
    return PUNCT.sub("", s)


def line_keys(name):
    """웨이 이름에서 뽑을 수 있는 열쇠 전부.

    OSM 은 "東北本線（埼京線）" 처럼 괄호 안에 실제 계통을 적어 둔다.
    괄호를 통째로 버리면 埼京線 이 제 선로를 하나도 못 찾는다.
    """
    out = []
    k = norm_line(name)
    if k:
        out.append(k)
    for m in re.findall(r"[（(\[]([^）)\]]+)[）)\]]", name or ""):
        k2 = norm_line(m)
        if k2 and k2 not in out:
            out.append(k2)
    return out


class Tracks(osmium.SimpleHandler):
    """철도 웨이의 노드 순서, 좌표, 이름."""

    def __init__(self):
        super().__init__()
        self.ways = []
        self.names = []
        self.attrs = []
        self.pos = {}

    def way(self, w):
        if w.tags.get("railway") not in RAIL_KINDS:
            return
        refs = []
        for n in w.nodes:
            if not n.location.valid():
                continue
            refs.append(n.ref)
            if n.ref not in self.pos:
                self.pos[n.ref] = (n.lon, n.lat)
        if len(refs) >= 2:
            self.ways.append(refs)
            self.names.append(w.tags.get("name:ja") or w.tags.get("name") or "")
            self.attrs.append(_way_attrs(w.tags))


def _way_attrs(tags):
    """주행 속도를 가늠할 선로 등급. 없는 태그는 None."""
    def num(v):
        m = re.match(r"\s*(\d+(?:\.\d+)?)", v or "")
        return float(m.group(1)) if m else None
    tracks = num(tags.get("tracks"))
    elec = tags.get("electrified")
    return (tags.get("railway"), tags.get("usage"), tags.get("service"),
            num(tags.get("maxspeed")),
            None if elec is None else elec != "no",
            None if tracks is None else tracks >= 2)


def _load_tracks(pbfs):
    """저장해 둔 선로 훑기. 도장이 안 맞으면 None."""
    npz_path, json_path = osmcache.files("track", pbfs)
    meta = (osmcache.read_meta(json_path, pbfs, Tracks, _way_attrs)
            if npz_path.exists() else None)
    if meta is None:
        return None
    try:
        z = np.load(npz_path)
    except (OSError, ValueError) as e:
        print(f"  (저장해 둔 선로 훑기를 못 읽었다: {e})", flush=True)
        return None
    tr = _Bag()
    refs, ptr = z["refs"], z["ptr"]
    tr.ways = [refs[ptr[i]:ptr[i + 1]].tolist() for i in range(len(ptr) - 1)]
    tr.names = meta["names"]
    tr.attrs = [tuple(a) for a in meta["attrs"]]
    ids, xy = z["node_ids"], z["node_xy"]
    tr.pos = {int(n): (float(xy[i, 0]), float(xy[i, 1])) for i, n in enumerate(ids)}
    return tr


def _save_tracks(pbfs, tr) -> None:
    npz_path, json_path = osmcache.files("track", pbfs)
    ptr = np.cumsum([0] + [len(w) for w in tr.ways])
    ids = sorted(tr.pos)
    np.savez_compressed(
        npz_path,
        refs=(np.concatenate([np.asarray(w, dtype=np.int64) for w in tr.ways])
              if tr.ways else np.zeros(0, dtype=np.int64)),
        ptr=np.asarray(ptr, dtype=np.int64),
        node_ids=np.asarray(ids, dtype=np.int64),
        node_xy=np.asarray([tr.pos[n] for n in ids], dtype=np.float64).reshape(-1, 2),
    )
    json_path.write_text(json.dumps({
        "stamp": osmcache.stamp(pbfs, Tracks, _way_attrs),
        "names": tr.names,
        "attrs": tr.attrs,
    }, ensure_ascii=False), encoding="utf-8")


class _Bag:
    """핸들러 자리에 끼울 껍데기."""


def build_edges(ways, pos, scale):
    """선로 간선 목록. 간선마다 어느 웨이에서 왔는지 남긴다."""
    ids = sorted(pos)
    slot = {n: i for i, n in enumerate(ids)}
    xy = np.array([pos[n] for n in ids], dtype=np.float64)
    x = xy[:, 0] * scale * 111_320.0
    y = xy[:, 1] * 111_132.0

    rows, cols, owner = [], [], []
    for k, refs in enumerate(ways):
        a = [slot[n] for n in refs if n in slot]
        rows.extend(a[:-1])
        cols.extend(a[1:])
        owner.extend([k] * max(len(a) - 1, 0))
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    owner = np.asarray(owner, dtype=np.int64)
    w = np.hypot(x[rows] - x[cols], y[rows] - y[cols])
    keep = w > 0
    return (rows[keep], cols[keep], w[keep], owner[keep]), xy, x, y


def to_csr(edges, n):
    """간선 목록을 양방향 희소 행렬로."""
    from scipy.sparse import coo_matrix

    rows, cols, w, _ = edges
    if len(rows) == 0:
        return coo_matrix((np.zeros(0), (np.zeros(0, np.int64),
                                         np.zeros(0, np.int64))),
                          shape=(n, n)).tocsr()
    return coo_matrix((np.concatenate([w, w]),
                       (np.concatenate([rows, cols]),
                        np.concatenate([cols, rows]))),
                      shape=(n, n)).tocsr()


# 남의 노선 선로를 탈 때 매기는 비용 배수. 1 이면 예전처럼 아무 선로나
# 최단으로 탄다. 너무 크면 직통운전 구간을 못 잇는다.
OTHER_TRACK_COST = 4.0

# 한 구간이 왜 그 길로 갔는지 볼 때 쓴다.
#   REGION=kanto_osm PROBE="역이름>역이름,..." python src/build_track.py
_PROBE = os.environ.get("PROBE", "")
_PROBE_PAIRS: set = set()


def _length_m(arc, scale):
    a = np.asarray(arc, dtype=np.float64)
    if len(a) < 2:
        return 0.0
    return float(np.hypot(np.diff(a[:, 0]) * scale * 111_320.0,
                          np.diff(a[:, 1]) * 111_132.0).sum())


def route(graph, dijkstra, srcs, targets, cap, xy):
    """여러 출발 후보에서 여러 도착 후보로.

    (좌표열, 출발 노드, 도착 노드) 를 돌려준다. 노드 번호가 필요한 것은
    다음 구간이 같은 노드에서 출발해야 하기 때문이다. 구간마다 역 옆
    선로 노드를 따로 고르면, 한 역에서 앞 구간의 끝과 뒤 구간의 시작이
    평균 166m 어긋나 지도에 삼각형이 그려진다.
    """
    if not srcs or not targets:
        return None
    dm, pm = dijkstra(graph, indices=srcs, limit=cap, return_predecessors=True)
    dm = np.atleast_2d(dm)
    pm = np.atleast_2d(pm)
    # srcs·targets 는 역에서 가까운 순서로 온다. 그 순서를 먼저 본다.
    #
    # 예전에는 닿는 후보 중 그래프 거리가 짧은 쪽을 골랐는데, 그러면
    # 선로를 따라가다가 역 반경(400m) 안에 들자마자 멈췄다. 종착역에서
    # 그게 그대로 보인다. 도큐 고도모노쿠니선은 선로가 역까지 닿아
    # 있는데도 그린 선이 346m 앞에서 끝났다.
    best, best_key = None, None
    for k, src in enumerate(srcs):
        for ti, tgt in enumerate(targets):
            if not np.isfinite(dm[k, tgt]):
                continue
            key = (ti, k, float(dm[k, tgt]))
            if best_key is not None and key >= best_key:
                break
            seq, cur = [], int(tgt)
            while cur >= 0:
                seq.append(cur)
                if cur == src:
                    break
                cur = int(pm[k, cur])
            if len(seq) >= 2 and seq[-1] == src:
                best, best_key = seq[::-1], key
            break
    if best is None:
        return None
    return xy[best], int(best[0]), int(best[-1])


SWITCHBACKS = _switchbacks()


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_rail import _pbf_list
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    railways = json.loads((RAW / "railways.json").read_text(encoding="utf-8"))
    stations = json.loads((RAW / "stations.json").read_text(encoding="utf-8"))
    cpos, cname, spos = {}, {}, {}
    for s in stations:
        # 좌표가 없는 역이 있다. ODPT 는 권역 밖 역도 목록에 담아 둔다
        # (조반선의 이와키·하라노마치 등 27개).
        if not s.get("coord"):
            continue
        # 묶음이 없으면 역 하나가 곧 묶음이다.
        c = s.get("cluster", s["id"])
        cpos.setdefault(c, (float(s["coord"][0]), float(s["coord"][1])))
        cname.setdefault(c, (s.get("title") or {}).get("ja", ""))
        spos[s["id"]] = (float(s["coord"][0]), float(s["coord"][1]))
    scale = float(np.cos(np.radians(
        float(np.median([p[1] for p in cpos.values()])))))

    pbfs = _pbf_list()
    print("[" + REGION + "] 선로 읽는 중: "
          + ", ".join(p.name for p in pbfs), flush=True)
    tr = None if os.environ.get("TRACK_RESCAN") == "1" else _load_tracks(pbfs)
    if tr is not None:
        print(f"  저장해 둔 선로 훑기를 다시 쓴다", flush=True)
    else:
        tr = Tracks()
        for p in pbfs:
            tr.apply_file(str(p), locations=True, idx="flex_mem")
        try:
            _save_tracks(pbfs, tr)
        except Exception as e:      # 캐시를 못 써도 빌드는 계속한다
            print(f"  (선로 훑기를 저장하지 못했다: {e})", flush=True)
    print(f"  철도 웨이 {len(tr.ways):,}개, 선로 노드 {len(tr.pos):,}개", flush=True)

    edges, xy, x, y = build_edges(tr.ways, tr.pos, scale)
    n_nodes = len(xy)
    whole = to_csr(edges, n_nodes)
    tree = cKDTree(np.stack([x, y], axis=1))
    print(f"  그래프 노드 {n_nodes:,}개, 간선 {len(edges[0]):,}개", flush=True)

    way_key = [line_keys(nm) for nm in tr.names]
    named = sum(1 for k in way_key if k)
    print(f"  이름이 붙은 선로 웨이 {named:,}개 "
          f"({named / max(len(way_key), 1) * 100:.0f}%)", flush=True)

    def snap_at(p):
        lon, lat = p
        d, i = tree.query([lon * scale * 111_320.0, lat * 111_132.0],
                          k=SNAP_TRIES, distance_upper_bound=SNAP_M)
        return [int(ii) for dd, ii in zip(np.atleast_1d(d), np.atleast_1d(i))
                if np.isfinite(dd) and ii < n_nodes]

    def snap(c):
        return snap_at(cpos[c])

    snapped = {c: snap(c) for c in cpos}
    # 역 줄마다 따로도 붙인다. 묶음 하나로 뭉개면 다층 역에서 남의 노선
    # 선로에 붙는다. 武蔵浦和 는 무사시노선이 지상, 사이쿄선이 고가라
    # 묶음 좌표로 찾으면 후보가 전부 무사시노선 선로였다.
    snapped_sid = {sid: snap_at(p) for sid, p in spos.items()}
    if os.environ.get("PROBE_WAYS"):
        want = set(os.environ["PROBE_WAYS"].split(","))
        owner_of = {}
        for e0, e3 in zip(edges[0], edges[3]):
            owner_of.setdefault(int(e0), int(e3))
        for c, nm in cname.items():
            if nm not in want:
                continue
            seen = []
            for i in (snapped.get(c) or []):
                w = owner_of.get(int(i))
                if w is not None and tr.names[w] and tr.names[w] not in seen:
                    seen.append(tr.names[w])
            print(f"    [ways] {nm}: " + " | ".join(seen[:8]), flush=True)
    lost = sum(1 for v in snapped.values() if not v)
    print(f"  선로에 못 붙인 역 {lost:,}개 / {len(snapped):,}개", flush=True)

    # 구간마다 찾은 선로를 그대로 남긴다. 이어 붙인 뒤 그릴 때 역을
    # 다시 투영해 자르면, 역 근처에서 선이 선로를 벗어나 모서리를
    # 질러간다. 京橋·天王寺·新今宮·白鷺 에서 눈에 보이던 것이 그것이다.
    # 찾을 때 이미 역에서 역까지의 선로를 알고 있으니 버리지 않는다.
    #
    # 열쇠에 노선을 넣는다. 역 쌓만 보면 같은 역 쌓을 지나는 다른
    # 노선끼리 서로 덮어쓴다. 日暮里-西日暮里 를 京浜東北線·山手線·
    # 日暮里舅人ライナー 셋이 쓰는데, kanto_osm 에서 그런 역 쌓이 436개다.
    segments = {}
    attrs_out = {}
    # 구간이 지나간 선로 웨이의 등급을 길이 비율로 모은다. build_naive 가
    # 구간마다 주행 속도를 달리 매기는 데 쓴다. 간선과 로컬선의 속도 차가
    # 권역 사이 차이보다 크다(북도호쿠 완행 東北本線 79km/h, 秋田内陸線 39).
    node_of = {(round(float(p[0]), 6), round(float(p[1]), 6)): i
               for i, p in enumerate(xy)}
    er, ec, _ew, eo = edges
    way_of = {}
    for u, v, k in zip(er.tolist(), ec.tolist(), eo.tolist()):
        way_of.setdefault((u, v), k)
        way_of.setdefault((v, u), k)

    def seg_attrs(seg):
        tot = 0.0
        acc = {"main": 0.0, "main_n": 0.0, "vmax": 0.0, "vmax_n": 0.0,
               "elec": 0.0, "elec_n": 0.0, "double": 0.0, "double_n": 0.0}
        kinds = defaultdict(float)
        for p, q in zip(seg, seg[1:]):
            u, v = node_of.get(tuple(p)), node_of.get(tuple(q))
            k = way_of.get((u, v)) if u is not None and v is not None else None
            L = _length_m([p, q], scale)
            if k is None or L <= 0:
                continue
            rail, usage, _svc, vmax, elec, dbl = tr.attrs[k]
            tot += L
            kinds[rail] += L
            if usage:
                acc["main_n"] += L
                acc["main"] += L * (usage == "main")
            if vmax:
                acc["vmax_n"] += L
                acc["vmax"] += L * vmax
            if elec is not None:
                acc["elec_n"] += L
                acc["elec"] += L * elec
            if dbl is not None:
                acc["double_n"] += L
                acc["double"] += L * dbl
        if tot <= 0:
            return None
        out = {"len": round(tot), "kind": max(kinds, key=kinds.get)}
        for key in ("main", "elec", "double"):
            if acc[key + "_n"] > tot * 0.5:
                out[key] = round(acc[key] / acc[key + "_n"], 2)
        if acc["vmax_n"] > tot * 0.5:
            out["vmax"] = round(acc["vmax"] / acc["vmax_n"])
        return out
    shapes, done, failed, own_hit = [], 0, 0, 0
    fail_len = []
    for r in railways:
        # 역 id 와 묶음 번호를 함께 들고 간다. 구간 열쇠는 역 id 로
        # 적어야 그리는 쪽과 시각표 쪽이 그대로 찾을 수 있다.
        # 묶음 번호가 없는 권역(ODPT 로 만든 시간표 권역)은 역 하나를
        # 묶음 하나로 본다. 구간 열쇠가 역 id 라 그대로 들어맞는다.
        st_ids = r.get("stations") or []
        cl = r.get("clusters") or st_ids
        pair_ids = [(sid, c) for sid, c in zip(st_ids, cl) if c in cpos]
        cs = [c for _sid, c in pair_ids]
        sids = [sid for sid, _c in pair_ids]
        if len(cs) < 2:
            continue

        # 제 노선 선로만 모은 그래프와, 남의 선로에 벌점을 준 그래프.
        #
        # 예전에는 제 선로에서 못 찾으면 전체 그래프로 통째로 넘어갔다.
        # 그 순간 이름을 아예 안 보게 되어, 옆에 붙은 남의 지선을 그냥
        # 타고 갔다. 사이쿄선이 武蔵浦和 에서 무사시노선 니시우라와
        # 지선으로 빠진 것이 그 자리다(직선 1.2km 를 1.5km 로 돌았다).
        #
        # 벌점을 주면 필요할 때만 남의 선로로 넘어간다. 직통운전은
        # 실제로 남의 선로를 달리므로 막아서는 안 된다.
        key = norm_line(r["title"].get("ja", ""))
        title_ja = r["title"].get("ja", "")
        # 이 노선의 스위치백 역. 아래 구간 반복문이 sb 를 역 id 로 쓰므로
        # 이름을 겹치지 않게 둔다.
        sb_at = SWITCHBACKS.get(title_ja) or SWITCHBACKS.get(
            re.sub(r"\s*[(（].*$", "", title_ja)) or set()
        sub = pen = own_nodes = None
        if key:
            mask = np.array([any(k == key or k in key or key in k for k in ks)
                             for ks in way_key])
            pick = mask[edges[3]]
            if int(pick.sum()) >= 2:
                sub = to_csr((edges[0][pick], edges[1][pick],
                              edges[2][pick], edges[3][pick]), n_nodes)
            if int(pick.sum()):
                pen = to_csr((edges[0], edges[1],
                              edges[2] * np.where(pick, 1.0, OTHER_TRACK_COST),
                              edges[3]), n_nodes)
                # 제 노선 선로 위의 노드. 역을 선로에 붙일 때 먼저 본다.
                own_nodes = np.zeros(n_nodes, dtype=bool)
                own_nodes[edges[0][pick]] = True
                own_nodes[edges[1][pick]] = True

        if _PROBE:
            for spec in _PROBE.split(","):
                if ">" not in spec:
                    continue
                x, y = [t.strip() for t in spec.split(">", 1)]
                for (ca, cb), (s1, s2) in zip(zip(cs, cs[1:]), zip(sids, sids[1:])):
                    if cname.get(ca) == x and cname.get(cb) == y:
                        _PROBE_PAIRS.add(f"{s1}|{s2}")

        def cand(c, sid=None):
            """역 옆 선로 노드. 제 노선 선로 위의 것을 앞으로 보낸다.

            다층 역에서는 역 노드에 제일 가까운 선로가 남의 노선일 수
            있다. 武蔵浦和 는 무사시노선이 지상, 사이쿄선이 고가다.
            그대로 두면 사이쿄선이 무사시노선 선로에서 출발해 서쪽으로
            332m 불룩하게 돌아 나온다.
            """
            v = (snapped_sid.get(sid) or []) if sid else []
            v = v or (snapped.get(c) or [])
            if own_nodes is None or not v:
                return v
            mine = [i for i in v if own_nodes[i]]
            return mine + [i for i in v if not own_nodes[i]] if mine else v

        pieces, path = [], []

        def flush():
            if len(path) >= 2:
                pieces.append(list(path))
            path.clear()

        prev_node = last_node = None
        for (a, b), (sa, sb) in zip(zip(cs, cs[1:]), zip(sids, sids[1:])):
            tgt = cand(b, sb)                   # 도착 역 옆 선로 노드 후보
            straight = float(np.hypot(
                (cpos[b][0] - cpos[a][0]) * scale * 111_320.0,
                (cpos[b][1] - cpos[a][1]) * 111_132.0))
            cap = max(straight * MAX_RATIO, MIN_CAP_M)
            if sb_at and (cname.get(a) in sb_at or cname.get(b) in sb_at):
                cap = max(straight * SWITCHBACK_RATIO, SWITCHBACK_MIN_M)

            def find(srcs):
                if sub is not None:
                    got = route(sub, dijkstra, srcs, tgt, cap, xy)
                    if got is not None:
                        return got, True
                if pen is not None:
                    # 벌점 그래프에서는 비용이 길이가 아니다. 한도를
                    # 벌점만큼 늘려 잡고, 실제 길이로 다시 거른다.
                    got = route(pen, dijkstra, srcs, tgt,
                                cap * OTHER_TRACK_COST, xy)
                    if got is not None and _length_m(got[0], scale) <= cap:
                        return got, False
                return route(whole, dijkstra, srcs, tgt, cap, xy), False

            # 앞 구간이 끝난 노드에서 이어 간다. 거기서 길이 없을 때만
            # 역 옆 후보를 다시 연다.
            best, own = (None, False)
            if prev_node is not None:
                best, own = find([prev_node])
            if best is None:
                best, own = find(cand(a, sa))
            if _PROBE and f"{sa}|{sb}" in _PROBE_PAIRS:
                where = "제선로" if own else ("벌점" if pen is not None else "전체")
                ln = _length_m(best[0], scale) if best else 0
                print(f"    [probe] {sa}->{sb}: {where} 길이 {ln:.0f}m "
                      f"직선 {straight:.0f}m cap {cap:.0f}m "
                      f"sub={'있음' if sub is not None else '없음'}", flush=True)
            started = None
            if best is not None:
                own_hit += own
                best, started, prev_node = best
            else:
                prev_node = None

            # 앞 구간이 끝난 노드에서 못 이어 다른 자리에서 새로 시작했으면
            # 거기서 선을 끊는다. 그대로 이어 붙이면 그 사이가 직선 하나로
            # 그어진다. 스냅 후보는 역에서 400m 까지라 두 자리가 800m 까지
            # 벌어질 수 있다. 銀座線 末広町 에서 186m, 都営浅草線 戸越 에서
            # 173m 짜리 현이 그렇게 생겼다.
            if best is not None and path and started is not None and started != last_node:
                flush()
            last_node = prev_node

            if best is None:
                # 못 이은 자리에서는 선을 끊는다. 직선으로 때워 한 줄에
                # 섞으면 지도에 긴 현이 그어진다.
                failed += 1
                fail_len.append(straight)
                flush()
                continue
            done += 1
            seg = []
            for q in best:
                pair = [round(float(q[0]), 6), round(float(q[1]), 6)]
                if not seg or seg[-1] != pair:
                    seg.append(pair)
            if len(seg) >= 2:
                segments[f"{r['id']}|{sa}|{sb}"] = seg
                got_attrs = seg_attrs(seg)
                if got_attrs:
                    attrs_out[f"{r['id']}|{sa}|{sb}"] = got_attrs
            for pair in seg:
                if not path or path[-1] != pair:
                    path.append(pair)
        flush()
        for k, piece in enumerate(pieces):
            shapes.append({"id": r["id"] if k == 0 else f"{r['id']}~{k}",
                           "sublines": [{"type": "main", "coords": piece}]})

    # 선로를 못 따라간 구간에는 선형이 없다. build_rail.py 가 노선
    # 관계에서 뽑아 둔 선형을 다른 id 로 함께 내보내, geometry.py 가
    # 구간마다 "역에 닿는 쪽" 을 고르게 한다. 라우팅이 되는 자리에는
    # 선로 선형이, 안 되는 자리에는 관계 선형이 뽑힌다.
    rel_path = RAW / "coordinates-rel.json"
    fallback = 0
    if rel_path.exists():
        rel = json.loads(rel_path.read_text(encoding="utf-8"))
        for e in rel.get("railways", []):
            shapes.append({"id": e["id"] + "#rel", "sublines": e["sublines"]})
            fallback += 1
    else:
        print(f"  !! 관계 선형이 없습니다: {rel_path}", flush=True)

    # 시각표 권역의 coordinates.json 은 ODPT 원본이라 덮으면 안 된다.
    # 구간 선형만 만들고 싶을 때 쓴다.
    #     REGION=kanto TRACK_ONLY=1 python src/build_track.py
    out = RAW / "coordinates.json"
    if os.environ.get("TRACK_ONLY") == "1":
        print("  (노선 선형은 건드리지 않는다)", flush=True)
    else:
        out.write_text(json.dumps({"railways": shapes, "airways": []},
                                  ensure_ascii=False), encoding="utf-8")
    # 따로 둔다. 같은 파일에 두면 build_rail.py 가 다시 돌 때 덮어써,
    # 오류 하나 없이 선형이 예전 방식(모서리 질러가기)으로 돌아간다.
    seg_out = RAW / "track-segments.json"
    seg_out.write_text(json.dumps(segments, ensure_ascii=False),
                       encoding="utf-8")
    (RAW / "track-attrs.json").write_text(json.dumps(attrs_out, ensure_ascii=False),
                                          encoding="utf-8")
    total = done + failed
    print("")
    print(f"  구간 {total:,}개 중 선로를 따라 이은 것 {done:,}개 "
          f"({done / max(total, 1) * 100:.1f}%), 못 이어 끊은 것 {failed:,}개")
    print(f"  그중 제 노선 선로 안에서 찾은 것 {own_hit:,}개 "
          f"({own_hit / max(done, 1) * 100:.0f}%)")
    print(f"  관계에서 뽑은 선형 {fallback:,}개를 후보로 함께 담는다")
    print(f"  역에서 역까지 따로 남긴 구간 선형 {len(segments):,}개 "
          f"-> {seg_out.name} ({seg_out.stat().st_size / 1e6:.1f} MB)")
    if os.environ.get("TRACK_ONLY") != "1":
        print(f"  노선 선형 {len(shapes):,}개 -> {out} "
              f"({out.stat().st_size / 1e6:.1f} MB)")
    if fail_len:
        d = np.array(fail_len)
        print(f"  못 이은 구간의 역간 직선거리: 중앙 {np.median(d):.0f}m, "
              f"최대 {d.max():.0f}m")


if __name__ == "__main__":
    main()
