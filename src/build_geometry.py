"""시각표 권역의 노선 선형을 OSM 선로로 갈아 끼운다.

mini-tokyo-3d 의 선형은 3D 시각화용이라 도심에서 도식적이고, 선로를
공유하는 구간은 노선별이 아니라 'Base.TabataShinagawa' 같은 공용
선형 하나로 들어 있다. 그러면 게이힌토호쿠선을 그릴 때 "자기 노선의
선형" 이라는 단서가 사라지고, 간다에서 아키하바라 사이에서 더 짧은
호인 주오선 선로를 따라가 버린다.

OSM 에는 노선마다 실제 선로가 따로 있다. 다만 노선 id 가 달라서 그냥
가져다 쓰면 "자기 노선" 판정이 통째로 깨진다. 그래서 역 좌표 겹침으로
ODPT 노선과 OSM 노선을 맞춘 뒤, ODPT id 를 달아 내보낸다.

다만 OSM 이 언제나 나은 것은 아니다. 京成本線 은 우에노로 들어가는
지하 구간이 OSM 에 없어서 선이 터널 입구에서 끊긴다. 그래서 노선마다
둘을 실제로 재어 나은 쪽을 고른다. 잣대는 셋이다.

  어긋남   선의 끝과 역 사이 거리
  모자람   호가 두 역 직선거리에 못 미치는 만큼
  구멍     중간에 크게 벌어진 걸음

고르지 않은 쪽도 다른 id 로 남긴다. 직통 구간처럼 자기 노선 선형이
닿지 않는 자리에서 후보로 쓰이기 때문이다.

사용법: REGION=kanto SOURCE=kanto_osm python src/build_geometry.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REGION = os.environ.get("REGION", "kanto")
SOURCE = os.environ.get("SOURCE", "kanto_osm")
BASE = ROOT / "data" / "regions" / REGION
SRC = ROOT / "data" / "regions" / SOURCE
OUT = BASE / "raw" / "coordinates-osm.json"

NEAR_M = 300.0          # 역이 이만큼 안에 있으면 같은 역으로 본다
MATCH_MIN = 0.60        # 노선의 역이 이만큼 담겨야 같은 노선으로 본다


def score(geo, pairs, coords, scale):
    """노선 하나의 선형이 얼마나 잘 맞는지. 낮을수록 좋다."""
    total, n = 0.0, 0
    for a, b in pairs:
        P = np.array(geo.ride_path([a, b]), dtype=float)
        straight = float(np.hypot((coords[b, 0] - coords[a, 0]) * scale * 111_320.0,
                                  (coords[b, 1] - coords[a, 1]) * 111_132.0))
        if straight < 30:
            continue
        n += 1
        if len(P) < 2:
            total += straight
            continue
        st = np.hypot((P[1:, 0] - P[:-1, 0]) * scale * 111_320.0,
                      (P[1:, 1] - P[:-1, 1]) * 111_132.0)
        head = float(np.hypot((P[0, 0] - coords[a, 0]) * scale * 111_320.0,
                              (P[0, 1] - coords[a, 1]) * 111_132.0))
        tail = float(np.hypot((P[-1, 0] - coords[b, 0]) * scale * 111_320.0,
                              (P[-1, 1] - coords[b, 1]) * 111_132.0))
        arc = float(st.sum())
        gap = float(st.max())
        total += (max(head, tail)
                  + max(0.0, straight * 0.98 - arc)
                  + max(0.0, gap - max(200.0, straight * 0.30)))
    return total / max(n, 1)


def pick_better(swapped, old, stops, coords, scale):
    """노선마다 OSM 선형과 원래 선형을 실제로 재어 나은 쪽을 고른다."""
    from geometry import Geometry
    import tempfile

    ids = {s: i for i, s in enumerate(stops["ids"])}
    by_rail = defaultdict(list)
    for i, rid in enumerate(stops["railway"]):
        by_rail[rid].append(i)

    def build(entries):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            json.dump({"railways": entries, "airways": []}, f, ensure_ascii=False)
            path = f.name
        try:
            return Geometry(path, stops["railway"], coords)
        finally:
            Path(path).unlink(missing_ok=True)

    geo_osm = build([{"id": rid, "sublines": e["sublines"]}
                     for rid, e in swapped.items()])
    geo_old = build(old["railways"])

    # 노선별 인접 역 쌍. 순서는 stops.json 의 등장 순서를 따른다.
    railways = json.loads((BASE / "raw" / "railways.json").read_text(encoding="utf-8"))
    order_of = {r["id"]: [ids[s] for s in (r.get("stations") or []) if s in ids]
                for r in railways}

    won = {}
    for rid in swapped:
        order = order_of.get(rid) or sorted(by_rail.get(rid, []))
        pairs = [(a, b) for a, b in zip(order, order[1:])
                 if np.isfinite(coords[a, 0]) and np.isfinite(coords[b, 0])]
        if not pairs:
            won[rid] = "old"
            continue
        won[rid] = "osm" if score(geo_osm, pairs, coords, scale) <             score(geo_old, pairs, coords, scale) else "old"
    return won


def main() -> None:
    stops = json.loads((BASE / "stops.json").read_text(encoding="utf-8"))
    coords = np.array(stops["coords"], dtype=np.float64)
    scale = float(np.cos(np.radians(float(np.nanmedian(coords[:, 1])))))

    by_railway = defaultdict(list)
    for i, rid in enumerate(stops["railway"]):
        if np.isfinite(coords[i, 0]):
            by_railway[rid].append((float(coords[i, 0]), float(coords[i, 1])))

    src_rw = json.loads((SRC / "raw" / "railways.json").read_text(encoding="utf-8"))
    src_st = {s["id"]: s
              for s in json.loads((SRC / "raw" / "stations.json").read_text(encoding="utf-8"))}
    src_geo = json.loads((SRC / "raw" / "coordinates.json").read_text(encoding="utf-8"))
    # 한 노선이 여러 사슬로 나뉘어 있을 수 있다. 전부 들고 있어야 노선의
    # 일부만 덮는 일이 없다.
    shape_of = defaultdict(list)
    for e in src_geo["railways"]:
        shape_of[e["id"].split("~")[0]].append(e)

    lines = []
    for r in src_rw:
        pts = [src_st[s]["coord"] for s in r["stations"] if s in src_st]
        if len(pts) >= 2 and r["id"] in shape_of:
            P = np.array(pts, dtype=float)
            lines.append((r["id"], P[:, 0] * scale * 111_320.0, P[:, 1] * 111_132.0,
                          r["title"].get("ja", "")))
    print(f"[{REGION}] 노선 {len(by_railway)}개, {SOURCE} 후보 {len(lines)}개", flush=True)

    swapped, kept, report = {}, [], []
    matched = {}          # 이쪽 노선 id -> 짝이 된 OSM 노선 id
    _mine_rw = json.loads((BASE / "raw" / "railways.json").read_text(encoding="utf-8"))
    by_title = {r["id"]: r.get("title", {}) for r in _mine_rw}
    # 이름을 빌려줄지 가릴 때 쓸 역 이름 집합
    _mine_st = {s["id"]: s["title"].get("ja", "") for s in
                json.loads((BASE / "raw" / "stations.json").read_text(encoding="utf-8"))}
    name_stations = {r["id"]: {_mine_st[x] for x in (r.get("stations") or [])
                               if _mine_st.get(x)} for r in _mine_rw}
    osm_stations = {r["id"]: {src_st[x]["title"].get("ja", "")
                              for x in (r.get("stations") or []) if x in src_st}
                    for r in src_rw}
    for rid, pts in sorted(by_railway.items()):
        P = np.array(pts, dtype=float)
        px = P[:, 0] * scale * 111_320.0
        py = P[:, 1] * 111_132.0
        best, best_score, best_name = None, 0.0, ""
        for sid, sx, sy, name in lines:
            d = np.hypot(px[:, None] - sx[None, :], py[:, None] - sy[None, :])
            score = float((d.min(axis=1) <= NEAR_M).mean())
            if score > best_score:
                best_score, best, best_name = score, sid, name
        if best is not None and best_score >= MATCH_MIN:
            swapped[rid] = shape_of[best]
            matched[rid] = best
        else:
            kept.append(rid)
        report.append((rid, best_name, best_score, best in swapped.values() if best else False))

    # 역에서 역까지의 선로도 옮긴다. build_track.py 가 OSM 권역에서
    # "OSM 노선|OSM 역|OSM 역" 로 적어 둔 것을, 이쪽 노선과 역 id 로
    # 바꿔 단다. 이게 있으면 그리는 쪽이 선형 위에 역을 다시 투영하지
    # 않아도 된다. 투영은 역 근처에서 모서리를 질러간다.
    from scipy.spatial import cKDTree

    seg_src = SRC / "raw" / "track-segments.json"
    src_seg = (json.loads(seg_src.read_text(encoding="utf-8"))
               if seg_src.exists() else {})
    # OSM 역 id -> 좌표
    src_pos = {sid: st["coord"] for sid, st in src_st.items()}
    keys = list(src_pos)
    if keys and src_seg:
        SP = np.array([src_pos[k] for k in keys], dtype=float)
        stree = cKDTree(np.stack([SP[:, 0] * scale * 111_320.0,
                                  SP[:, 1] * 111_132.0], axis=1))
        # 역 행이 아니라 묶음 번호로 맞춘다. 같은 자리에 여러 노선의
        # 역 행이 겹쳐 있어서, 가장 가까운 행을 집으면 엉땡한 노선의
        # 행이 나온다. 日比谷線 八丁堀 이 JR京葉線 의 행에 맞춰져 22개
        # 구간 중 6개만 올겨졌고, 그래서 그 자리가 끊겼다.
        near = {}
        for i, sid in enumerate(stops["ids"]):
            if not np.isfinite(coords[i, 0]):
                continue
            d, k = stree.query([coords[i, 0] * scale * 111_320.0,
                                coords[i, 1] * 111_132.0])
            if d <= NEAR_M:
                near[sid] = src_st[keys[int(k)]].get("cluster")
    else:
        near = {}

    # 역 묶음 쌍 -> 그 사이를 이어 둔 선로. 어느 노선이 들고 있든 담는다.
    #
    # 예전에는 노선 짝을 하나 정해 두고 그 노선 안에서만 찾았다. 그래서
    # 総武快速線 은 역 18개가 전부 OSM 에 있는데도 구간이 0개였다. 짝으로
    # 정해진 노선이 그 역 쌍을 인접으로 들고 있지 않아서다. 역 쌍이 같으면
    # 그 사이 선로는 같으므로, 짝 노선을 먼저 보고 없으면 아무 노선에서나
    # 가져온다.
    by_pair = {}
    for key, arc in src_seg.items():
        bits = key.split("|")
        if len(bits) != 3:
            continue
        orid, sa, sb = bits
        ca = sa.rsplit(".", 1)[-1]
        cb = sb.rsplit(".", 1)[-1]
        by_pair.setdefault((ca, cb), []).append((orid, arc))

    row_of = {sid: i for i, sid in enumerate(stops["ids"])}

    def gap_m(p, q):
        return float(np.hypot((p[0] - q[0]) * scale * 111_320.0, (p[1] - q[1]) * 111_132.0))

    def arc_len(arc):
        return sum(gap_m(p, q) for p, q in zip(arc, arc[1:]))

    def u_turn(arc):
        """선형 안에 되꺾이는 자리(150도 넘게 도는 곳)가 있는가. 아주 짧은
        걸음은 방향이 흔들리므로 3m 넘는 걸음만 본다."""
        steps = [(p, q) for p, q in zip(arc, arc[1:]) if gap_m(p, q) > 3.0]
        for (p0, p1), (q0, q1) in zip(steps, steps[1:]):
            ax, ay = (p1[0] - p0[0]) * scale, p1[1] - p0[1]
            bx, by = (q1[0] - q0[0]) * scale, q1[1] - q0[1]
            na, nb = np.hypot(ax, ay), np.hypot(bx, by)
            if na and nb and (ax * bx + ay * by) / (na * nb) < -0.866:
                return True
        return False

    # 스위치백 역. 이 역에 닿는 구간은 되꺾이는 것이 실제 선로다.
    sb_path = ROOT / "data" / "switchbacks.json"
    switchback = {n for k, v in (json.loads(sb_path.read_text(encoding="utf-8"))
                                 if sb_path.exists() else {}).items()
                  if isinstance(v, list) for n in v}

    def find_arc(osm_rid, ca, cb, pa=None, pb=None, turns_ok=False):
        """역 쌍 사이 선로. 두 방향에 적힌 것을 다 후보로 놓고 고른다.

        예전에는 한 방향에서 짝 노선을 못 찾으면 그 방향의 첫 선형을 바로
        가져왔다. 常磐線快速 의 新橋-東京 은 新橋->東京 방향으로 横須賀線 만
        있어(지하 승강장에서 끝난다), 도쿄역 앞에서 150m 를 가로질러 꺾였다.
        그렇다고 짝 노선을 무조건 먼저 쓰면 ODPT 가 도쿄역을 지상 한 점으로
        두는 탓에 横須賀線 이 같은 식으로 꺾이고, 짝 노선 선형에 되꺾임이
        든 곳(総武線 四街道-物井)도 있다.

        되꺾이며 가장 짧은 후보보다 10% 넘게 긴 것은 다른 후보가 있으면 빼고,
        양 끝이 이쪽 역 좌표에 가장 가까운 것을 고른다. 거리가 10m 안에서
        같으면 짝 노선, 그다음 짧은 것이다. 길이를 함께 보는 것은 선로가
        역 앞에서 잠깐 지그재그 하는 것까지 되꺾임으로 걸리기 때문이다
        (小田急小田原線 海老名-厚木 은 끝이 8m·11m 로 맞는데 한 점이 튀었다).
        """
        cands = []
        for a_, b_, flip in ((ca, cb, False), (cb, ca, True)):
            for orid, arc in by_pair.get((str(a_), str(b_))) or ():
                cands.append((orid, arc[::-1] if flip else arc))
        if not cands:
            return None
        if pa is None or pb is None:
            return next((arc for orid, arc in cands if orid == osm_rid), cands[0][1])
        shortest = min(arc_len(arc) for _, arc in cands)
        ok = cands if turns_ok else [
            c for c in cands
            if not (u_turn(c[1]) and arc_len(c[1]) > 1.1 * shortest)] or cands
        return min(ok, key=lambda c: (round((gap_m(c[1][0], pa) + gap_m(c[1][-1], pb)) / 10),
                                      c[0] != osm_rid, arc_len(c[1])))[1]

    segments = {}
    borrowed = 0
    for r in json.loads((BASE / "raw" / "railways.json").read_text(encoding="utf-8")):
        osm_rid = matched.get(r["id"])
        order = r.get("stations") or []
        for a, b in zip(order, order[1:]):
            ca, cb = near.get(a), near.get(b)
            if ca is None or cb is None:
                continue
            ia, ib = row_of.get(a), row_of.get(b)
            pa = coords[ia] if ia is not None else None
            pb = coords[ib] if ib is not None else None
            # 스위치백 역에 닿는 구간은 되꺾임을 걸러내지 않는다
            names = {stops["ja"][i] for i in (ia, ib) if i is not None}
            arc = find_arc(osm_rid, ca, cb, pa, pb, turns_ok=bool(names & switchback))
            if arc is not None:
                segments[f"{r['id']}|{a}|{b}"] = arc
                if not osm_rid:
                    borrowed += 1

    # 노선 이름을 언어별로 옮겨 둔다. OSM 관계에는 name:ko 가
    # 19%, name:zh 가 3% 뿐인데, 이쪽(ODPT)은 100% 다 갖추고 있다.
    # 짝이 된 노선의 이름을 빌려쓴다. 빈 언어만 채우고 있는 것은
    # 건들지 않는다.
    # 이름은 선형보다 깐깐하게 맞춘다. 선형은 "대충 같은 회랑" 이면 쓸 만하지만
    # 이름은 노선의 정체라, 조금만 어긋나도 東武日光線 이 "닛코선" 이 된다.
    # 실제로 きぬがわ列車 와 東武日光線 이 JR日光線 의 이름을 받아 갔다.
    # 역 목록이 합집합 대비 80% 넘게 겹칠 때만 빌려준다.
    lent = {}
    for rid, osm_rid in matched.items():
        mine_st = {x for x in (name_stations.get(rid) or set())}
        theirs = {x for x in (osm_stations.get(osm_rid) or set())}
        if not mine_st or not theirs:
            continue
        share = len(mine_st & theirs)
        if share < 3 or share < 0.80 * len(mine_st | theirs):
            continue
        mine = by_title.get(rid) or {}
        got = lent.setdefault(osm_rid, {})
        for lang in ("ja", "en", "ko", "zh-Hans", "zh-Hant"):
            v = (mine.get(lang) or "").strip()
            if v and lang not in got:
                got[lang] = v
    name_out = SRC / "raw" / "line-names.json"
    name_out.write_text(json.dumps(lent, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    old = json.loads((BASE / "raw" / "coordinates.json").read_text(encoding="utf-8"))

    # OSM 은 실측이라 선로 위에 있다. 노선 단위로 어느 쪽이 나은지 고르는
    # 대신, 맞춘 것은 모두 OSM 을 ODPT id 로 내보낸다. 구간마다 어느 쪽을
    # 쓸지는 geometry.py 가 "역에 닿는가" 를 보고 정한다. OSM 에 없는
    # 지하 구간 같은 자리에서는 원래 선형이 그대로 뽑힌다.
    out = []
    for rid, entries in swapped.items():
        for k, e in enumerate(entries):
            out.append({"id": rid if k == 0 else f"{rid}~{k}",
                        "sublines": e["sublines"]})
    have = set(swapped)
    for entry in old["railways"]:
        keep = dict(entry)
        if entry["id"] in have:
            keep["id"] = entry["id"] + "#mt3d"
        out.append(keep)

    OUT.write_text(json.dumps({"railways": out, "airways": old.get("airways", [])},
                              ensure_ascii=False), encoding="utf-8")
    seg_out = BASE / "raw" / "track-segments.json"
    seg_out.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")

    print(f"  OSM 선로를 단 노선 {len(swapped)}개, 못 맞춘 노선 {len(kept)}개")
    print(f"  원래 선형 {len(old['railways'])}개는 후보로 남긴다")
    print(f"  역에서 역까지의 선로 {len(segments):,}개를 옮겨 {seg_out.name} 에 썼다 "
          f"({SOURCE} 에 {len(src_seg):,}개)")
    print(f"  노선 이름을 언어별로 {len(lent):,}개 옮겨 {name_out.name} 에 썼다")
    print(f"  저장 -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    low = sorted((s, r, n) for r, n, s, _ in report if s < MATCH_MIN)
    if low:
        print(f"\n  못 맞춘 노선 {len(low)}개 (겹침 낮은 순)")
        for s, r, n in low[:14]:
            print(f"    {r:<34} 겹침 {s * 100:>3.0f}%  가장 가까운 것: {n[:24]}")


if __name__ == "__main__":
    main()
