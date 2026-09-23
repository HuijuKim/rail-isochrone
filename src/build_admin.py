"""역마다 어느 도도부현에 있는지 붙인다.

mini-tokyo-3d 의 역 데이터에는 현 정보가 없다. OSM 추출본에는 행정경계가
들어 있으므로(admin_level=4 가 도도부현) 거기서 다각형을 받아 역 좌표를
넣어 본다.

다각형은 실행 중에 쓸 일이 없다. 역 -> 현 이름만 남기고 버린다.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REGION = os.environ.get("REGION", "kanto")
BASE = DATA / "regions" / REGION
# 읽을 추출본은 build_walk 와 같은 목록을 쓴다. 야마나시·시즈오카 경계는
# 주부 추출본에만 있다.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import osmcache  # noqa: E402
from build_walk import PBFS  # noqa: E402

# 도도부현
ADMIN_LEVEL = "4"


def _admin_files(pbf):
    return osmcache.files("adminway", [pbf])


def _scan_admin(pbf):
    """추출본 하나의 경계 조각. 관계에서 way 의 임자를, way 에서 좌표를 받는다."""
    import osmium

    labels: dict[str, dict[str, str]] = {}
    want: dict[int, set[str]] = {}
    for rel in osmium.FileProcessor(str(pbf)).with_filter(
            osmium.filter.EntityFilter(osmium.osm.RELATION)):
        tags = rel.tags
        if tags.get("boundary") != "administrative":
            continue
        if tags.get("admin_level") != ADMIN_LEVEL:
            continue
        name = tags.get("name") or tags.get("name:ja")
        if not name:
            continue
        labels[name] = {
            "ja": name,
            "en": tags.get("name:en") or name,
            "ko": tags.get("name:ko") or name,
            # 간체가 없으면 name:zh 를 쓴다. 거기에 "东京都/東京都" 처럼
            # 두 표기가 빗금으로 붙어 오는 경우가 있어 앞쪽만 취한다.
            "zh-Hans": (tags.get("name:zh-Hans")
                        or (tags.get("name:zh") or name).split("/")[0]),
            "zh-Hant": (tags.get("name:zh-Hant")
                        or (tags.get("name:zh") or name).split("/")[-1]),
        }
        for member in rel.members:
            if member.type == "w":
                want.setdefault(member.ref, set()).add(name)

    ids, arrs = [], []
    for way in osmium.FileProcessor(str(pbf)).with_locations().with_filter(
            osmium.filter.EntityFilter(osmium.osm.WAY)):
        if way.id not in want:
            continue
        pts = [(n.location.lon, n.location.lat) for n in way.nodes
               if n.location.valid()]
        if len(pts) < 2:
            continue
        ids.append(way.id)
        arrs.append(np.asarray(pts, dtype=np.float64))
    return labels, want, ids, arrs


def _save_admin(pbf, labels, want, ids, arrs) -> None:
    npz_path, json_path = _admin_files(pbf)
    ptr = np.cumsum([0] + [len(a) for a in arrs])
    np.savez_compressed(
        npz_path,
        ids=np.asarray(ids, dtype=np.int64),
        pts=(np.concatenate(arrs) if arrs else np.zeros((0, 2))),
        ptr=np.asarray(ptr, dtype=np.int64),
    )
    json_path.write_text(json.dumps({
        "stamp": osmcache.stamp([pbf], _scan_admin),
        "labels": labels,
        "want": {str(w): sorted(n) for w, n in want.items()},
    }, ensure_ascii=False), encoding="utf-8")


def _load_admin(pbf):
    """저장해 둔 추출본 하나의 경계 조각. 도장이 안 맞으면 None."""
    npz_path, json_path = _admin_files(pbf)
    if not npz_path.exists():
        return None
    meta = osmcache.read_meta(json_path, [pbf], _scan_admin)
    if meta is None:
        return None
    try:
        z = np.load(npz_path)
    except (OSError, ValueError) as e:
        print(f"  (저장해 둔 행정경계를 못 읽었다: {e})", flush=True)
        return None
    ids, pts, ptr = z["ids"], z["pts"], z["ptr"]
    arrs = [pts[ptr[i]:ptr[i + 1]] for i in range(len(ids))]
    want = {int(w): set(n) for w, n in meta["want"].items()}
    return meta["labels"], want, [int(w) for w in ids], arrs


def prefecture_polygons() -> dict[str, list[np.ndarray]]:
    """현 이름 -> 닫힌 고리들.

    osmium 의 면 조립(with_areas)은 관계의 멤버가 하나라도 빠지면 면을 만들지
    않는다. 도쿄도 경계에는 이즈 제도와 오가사와라가 들어 있는데 그 way 들이
    간토 추출본 밖이라, 도쿄도가 통째로 빠진다. 시즈오카·야마나시처럼 추출본
    가장자리에서 잘린 현도 마찬가지다.

    그래서 멤버 way 를 직접 받아 이어 붙인다. 섬 쪽 조각이 없어도 본토 고리는
    제 힘으로 닫히므로, 닫힌 고리만 남기면 된다.

    훑기는 추출본 하나에만 매이므로 캐시도 하나씩 둔다. 현 경계가 추출본을
    넘어가도 괜찮다. 경계 way 가 놓인 추출본에는 그 way 를 멤버로 둔 관계도
    함께 들어 있어서, 조각을 다 모은 뒤에 이으면 고리가 닫힌다.
    """
    t0 = time.time()
    rescan = os.environ.get("ADMIN_RESCAN") == "1"

    # 1) 추출본 하나씩 읽어 모은다. way id 는 전역이라 겹치는 영역의 way 가
    #    두 번 들어온다. 같은 조각이 둘이면 조각과 그 복제본이 서로 맞물려
    #    2개짜리 가짜 고리를 만들고, 정작 본토 고리는 못 닫힌다. 먼저 읽은
    #    것만 받는다.
    labels: dict[str, dict[str, str]] = {}
    want: dict[int, set[str]] = {}
    taken: set[int] = set()
    order: list[tuple[int, np.ndarray]] = []
    for pbf in PBFS:
        part = None if rescan else _load_admin(pbf)
        if part is not None:
            print(f"  저장해 둔 행정경계를 다시 쓴다: {pbf.name}", flush=True)
        else:
            part = _scan_admin(pbf)
            print(f"  행정경계 훑기: {pbf.name} (현 {len(part[0])}개, "
                  f"경계 way {len(part[2]):,}개)", flush=True)
            try:
                _save_admin(pbf, *part)
            except Exception as e:  # 캐시를 못 써도 빌드는 계속한다
                print(f"  (행정경계를 저장하지 못했다: {e})", flush=True)
        plabels, pwant, ids, arrs = part
        labels.update(plabels)
        for w, names in pwant.items():
            want.setdefault(w, set()).update(names)
        for w, arr in zip(ids, arrs):
            if w in taken:
                continue
            taken.add(w)
            order.append((w, arr))
    print(f"  현 {len(labels)}개, 경계 way {len(want):,}개 "
          f"({time.time() - t0:.0f}s)", flush=True)

    # 2) 조각을 현별로 나눠 담고
    pieces: dict[str, list[np.ndarray]] = {n: [] for n in labels}
    for w, arr in order:
        for name in want.get(w, ()):
            pieces[name].append(arr)

    # 3) 이어 고리로
    out: dict[str, list[np.ndarray]] = {}
    for name, parts in pieces.items():
        rings = _stitch(parts)
        if rings:
            out[name] = rings
    print(f"  닫힌 고리 {sum(len(v) for v in out.values())}개 "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out, {k: v for k, v in labels.items() if k in out}


def _stitch(parts: list[np.ndarray], tol: float = 1e-7) -> list[np.ndarray]:
    """끝점이 맞는 조각들을 이어 닫힌 고리를 만든다. 못 닫으면 버린다."""
    from collections import defaultdict

    ends = defaultdict(list)

    def key(p):
        return (round(float(p[0]) / tol), round(float(p[1]) / tol))

    for i, part in enumerate(parts):
        ends[key(part[0])].append(i)
        ends[key(part[-1])].append(i)

    used = [False] * len(parts)
    rings: list[np.ndarray] = []
    for start in range(len(parts)):
        if used[start]:
            continue
        used[start] = True
        chain = [parts[start]]
        head, tail = key(parts[start][0]), key(parts[start][-1])
        # 꼬리에 붙일 조각을 계속 찾는다
        while tail != head:
            nxt = None
            for j in ends.get(tail, ()):
                if used[j]:
                    continue
                nxt = j
                break
            if nxt is None:
                break
            used[nxt] = True
            piece = parts[nxt]
            if key(piece[0]) != tail:
                piece = piece[::-1]
            chain.append(piece[1:])
            tail = key(piece[-1])
        if tail == head:
            ring = np.concatenate(chain)
            if len(ring) >= 4:
                rings.append(ring)
    return rings


def assign(polygons: dict[str, list[np.ndarray]], coords: np.ndarray) -> list[str | None]:
    """역 좌표를 현 다각형에 넣어 본다.

    현 하나에 고리가 수백 개씩(섬) 나오므로, 먼저 경계상자로 거르고 남은
    것만 포함 검사를 한다. 그러지 않으면 역 2,610개 x 고리 수천 개가 된다.
    """
    from matplotlib.path import Path as MplPath

    entries = []
    for name, rings in polygons.items():
        for ring in rings:
            entries.append((name, ring[:, 0].min(), ring[:, 0].max(),
                            ring[:, 1].min(), ring[:, 1].max(), MplPath(ring)))
    # 큰 고리부터 본다. 본토가 먼저 걸리면 섬 수백 개를 건너뛴다.
    entries.sort(key=lambda e: -((e[2] - e[1]) * (e[4] - e[3])))

    out: list[str | None] = []
    for lon, lat in coords:
        hit = None
        if np.isfinite(lon):
            for name, x0, x1, y0, y1, path in entries:
                if not (x0 <= lon <= x1 and y0 <= lat <= y1):
                    continue
                if path.contains_point((lon, lat)):
                    hit = name
                    break
        out.append(hit)
    return out


# 해안선으로 바다를 자를 때 쓰는 값
GAP_BRIDGE_M = 6000.0     # 해안선이 끊긴 끝끼리 직선으로 잇는 한계
SAMPLE_CELLS = 400_000    # 육지/바다를 가를 때 쓸 마스크 표본 칸 수
SLIVER_KM2 = 0.05         # 이보다 작은 조각은 버린다
# 경계를 줄이는 정도. 판정에 쓰는 고리와 지도에 그리는 선이 같은 값이어야
# 한다. 다르면 그린 선 안인데 클릭이 거부되는 띠가 그 차이만큼 생긴다.
BOUND_TOLERANCE_M = 15.0


def coast_lines(bb, scale):
    """해안선 사슬을 상자 안으로 자르고 끊긴 자리를 메운다.

    면을 만들려면 선이 닫혀 있어야 한다. 열린 끝이 하나라도 남으면 그
    자리에서 육지와 바다가 한 면으로 새어 붙는다. 간사이 상자 안에는
    끊긴 자리가 세 곳 있다(마이즈루 2km, 욧카이치 3.8km, 추출본 서쪽
    가장자리 두 곳). 가까운 끝끼리는 직선으로 잇고, 멀면 상자
    가장자리까지 곧장 내보낸다.
    """
    from shapely.geometry import LineString

    path = BASE / "walk" / "coast.npz"
    if not path.exists():
        print(f"  해안선 사슬이 없습니다: {path}", flush=True)
        return None, None
    z = np.load(path)
    pts, ptr = z["pts"], z["ptr"]

    chains = []
    for k in range(len(ptr) - 1):
        c = pts[ptr[k]:ptr[k + 1]]
        if len(c) < 2:
            continue
        g = LineString(c).intersection(bb)
        if g.is_empty:
            continue
        parts = [g] if g.geom_type == "LineString" else list(getattr(g, "geoms", []))
        for p in parts:
            if p.geom_type == "LineString" and len(p.coords) >= 2:
                chains.append(np.asarray(p.coords, dtype=np.float64))
    print(f"  해안선 사슬 {len(ptr) - 1:,}개 중 상자 안 {len(chains):,}개", flush=True)

    x0, y0, x1, y1 = bb.bounds
    eps = 1e-9

    def on_edge(p):
        return (abs(p[0] - x0) < eps or abs(p[0] - x1) < eps
                or abs(p[1] - y0) < eps or abs(p[1] - y1) < eps)

    loose = []
    for c in chains:
        if np.allclose(c[0], c[-1]):
            continue
        for p in (c[0], c[-1]):
            if not on_edge(p):
                loose.append(p)

    fix, bridged, pushed = [], 0, 0
    if loose:
        E = np.asarray(loose, dtype=np.float64)
        mx, my = E[:, 0] * scale * 111_320.0, E[:, 1] * 111_132.0
        d = np.hypot(mx[:, None] - mx[None, :], my[:, None] - my[None, :])
        np.fill_diagonal(d, np.inf)
        used = np.zeros(len(E), dtype=bool)
        for flat in np.argsort(d, axis=None):
            i, j = np.unravel_index(flat, d.shape)
            if d[i, j] > GAP_BRIDGE_M:
                break
            if used[i] or used[j]:
                continue
            used[i] = used[j] = True
            fix.append(np.array([E[i], E[j]]))
            bridged += 1
        for i in np.flatnonzero(~used):
            p = E[i]
            cand = [(abs(p[0] - x0), (x0, p[1])), (abs(p[0] - x1), (x1, p[1])),
                    (abs(p[1] - y0), (p[0], y0)), (abs(p[1] - y1), (p[0], y1))]
            fix.append(np.array([p, min(cand, key=lambda t: t[0])[1]]))
            pushed += 1
        span = sorted((float(np.hypot((f[1, 0] - f[0, 0]) * scale * 111_320.0,
                                      (f[1, 1] - f[0, 1]) * 111_132.0))
                       for f in fix[:bridged]), reverse=True)
        print(f"  끊긴 끝 {len(E)}개: 직선으로 이음 {bridged}쌍, "
              f"상자 밖으로 내보냄 {pushed}개", flush=True)
        if span:
            print("    이어 붙인 길이: "
                  + ", ".join(f"{v:,.0f}m" for v in span[:6])
                  + (" ..." if len(span) > 6 else ""), flush=True)
    return chains, fix


def land_polygons(bb, scale, land=None, land_grid=None):
    """해안선으로 육지 다각형을 만든다. 선은 벡터 그대로 쓴다.

    해안선 사슬을 그대로 평면에 깔면 상자가 면 여러 개로 쪼개진다.
    그 면을 하나씩 육지와 바다로 가르면 육지 다각형이 나온다. 자르는
    선은 OSM 해안선 좌표 그대로라 격자 오차가 없다.

    어느 면이 육지인지는 육지 마스크에 묻는다. 해안선 way 에는 "육지를
    왼쪽에 둔다" 는 방향 규약이 있어 그것으로 가르려 했는데, 쓸 수
    없었다. coast.npz 는 way 를 끝점끼리 이어 사슬로 만들면서 자유로운
    끝을 찾아 거꾸로 걷기도 해서, 방향이 뒤집힌 사슬이 섞여 있다.
    간사이는 우연히 맞았고 간토는 육지와 바다가 통째로 뒤바뀌었다.

    그래서 판정만 250m 마스크에 맡긴다. 면이 육지인지 바다인지는 칸
    하나보다 훨씬 큰 물음이라 마스크로 충분하고, 정작 눈에 보이는
    해안선은 벡터 그대로 남는다.
    """
    from shapely import STRtree, points as sh_points
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    if land is None:
        print("  육지 마스크가 없어 바다를 가릴 수 없습니다", flush=True)
        return None
    chains, fix = coast_lines(bb, scale)
    if not chains:
        return None
    lines = [LineString(c) for c in chains] + [LineString(f) for f in fix]
    faces = list(polygonize(unary_union(lines + [bb.exterior])))
    if not faces:
        print("  해안선이 면을 만들지 못했습니다", flush=True)
        return None
    slack = abs(sum(f.area for f in faces) / bb.area - 1.0)
    print(f"  해안선이 가른 면 {len(faces):,}개 (상자를 덮은 비율 "
          f"{(1 - slack) * 100:.3f}%)", flush=True)

    lon0, lat0, cell, w, h, m_lon, m_lat = land_grid
    w, h = int(w), int(h)
    assert land.shape == (h, w), f"육지 마스크 모양이 어긋납니다: {land.shape}"
    # 마스크 칸을 솎아 표본으로 쓴다. 전부 쓰면 수백만 점이 된다.
    # 마스크는 상자보다 훨씬 넓을 수 있으므로(간토는 주부까지 덮는다)
    # 상자 안 칸 수로 간격을 정해야 권역마다 촘촘함이 같아진다.
    x0, y0, x1, y1 = bb.bounds
    cx0 = max(0, int(np.floor((x0 - lon0) * m_lon / cell)))
    cx1 = min(w, int(np.ceil((x1 - lon0) * m_lon / cell)))
    cy0 = max(0, int(np.floor((y0 - lat0) * m_lat / cell)))
    cy1 = min(h, int(np.ceil((y1 - lat0) * m_lat / cell)))
    if cx1 <= cx0 or cy1 <= cy0:
        print("  육지 마스크가 권역을 덮지 않습니다", flush=True)
        return None
    step = max(1, round(float(np.sqrt((cx1 - cx0) * (cy1 - cy0) / SAMPLE_CELLS))))
    iy, ix = np.mgrid[cy0:cy1:step, cx0:cx1:step]
    iy, ix = iy.ravel(), ix.ravel()
    gx = lon0 + (ix + 0.5) * cell / m_lon
    gy = lat0 + (iy + 0.5) * cell / m_lat
    flag = land[iy, ix].astype(np.int64)

    votes = np.zeros((len(faces), 2), dtype=np.int64)
    qi, fi = STRtree(faces).query(sh_points(np.stack([gx, gy], axis=1)),
                                  predicate="within")
    np.add.at(votes, (fi, flag[qi]), 1)

    # 칸보다 작은 섬은 표본이 하나도 안 걸린다. 대표점으로 가른다.
    blind = np.flatnonzero(votes.sum(axis=1) == 0)
    for k in blind:
        c = faces[k].representative_point()
        cx = int(np.floor((c.x - lon0) * m_lon / cell))
        cy = int(np.floor((c.y - lat0) * m_lat / cell))
        if 0 <= cx < w and 0 <= cy < h and land[cy, cx]:
            votes[k, 1] = 1

    keep = [faces[k] for k in range(len(faces)) if votes[k, 1] > votes[k, 0]]
    order = sorted(range(len(faces)), key=lambda k: -faces[k].area)[:6]
    print(f"  표본 {len(flag):,}개로 가름. 가장 큰 면들", flush=True)
    for k in order:
        c = faces[k].representative_point()
        km2 = faces[k].area * scale * 111.320 * 111.132
        tot = max(votes[k].sum(), 1)
        print(f"     {c.x:7.3f},{c.y:6.3f} {km2:>9,.0f}km2  "
              f"{'육지' if votes[k, 1] > votes[k, 0] else '바다'} "
              f"({max(votes[k]) / tot * 100:.0f}%, 표본 {votes[k].sum():,})",
              flush=True)
    print(f"  육지 면 {len(keep):,}개, 표본을 못 받은 면 {len(blind)}개",
          flush=True)
    return unary_union(keep)


def rings_of(geom, scale):
    """다각형에서 바깥 고리만 꺼낸다. 부스러기는 버린다."""
    if geom is None or geom.is_empty:
        return []
    parts = [geom] if geom.geom_type == "Polygon" else [
        g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    out = []
    for p in parts:
        if p.area * scale * 111.320 * 111.132 < SLIVER_KM2:
            continue
        out.append(np.asarray(p.exterior.coords, dtype=np.float64))
    return out


def boundary_rings(geom, scale):
    """그릴 경계. 바깥 고리와 안쪽 구멍을 모두 닫힌 선으로 낸다."""
    if geom is None or geom.is_empty:
        return []
    parts = [geom] if geom.geom_type == "Polygon" else [
        g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    out = []
    for p in parts:
        if p.area * scale * 111.320 * 111.132 < SLIVER_KM2:
            continue
        out.append(np.asarray(p.exterior.coords, dtype=np.float64))
        for r in p.interiors:
            a = np.asarray(r.coords, dtype=np.float64)
            if abs(_shoelace(a)) * scale * 111.320 * 111.132 >= SLIVER_KM2:
                out.append(a)
    return out


def _shoelace(a):
    return 0.5 * float(np.sum(a[:-1, 0] * a[1:, 1] - a[1:, 0] * a[:-1, 1]))


def simplify(ring: np.ndarray, tol_m: float) -> np.ndarray:
    """더글러스-포이커. 경계는 판정에만 쓰므로 거칠어도 된다."""
    if len(ring) < 4:
        return ring
    scale = np.cos(np.radians(float(ring[:, 1].mean())))
    keep = np.zeros(len(ring), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(ring) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = ring[a], ring[b]
        seg = ring[a + 1:b]
        dx = (q[0] - p[0]) * scale * 111_320.0
        dy = (q[1] - p[1]) * 111_132.0
        length = np.hypot(dx, dy)
        sx = (seg[:, 0] - p[0]) * scale * 111_320.0
        sy = (seg[:, 1] - p[1]) * 111_132.0
        dist = np.hypot(sx, sy) if length < 1e-9 else np.abs(sx * dy - sy * dx) / length
        i = int(np.argmax(dist))
        if dist[i] > tol_m:
            keep[a + 1 + i] = True
            stack.append((a, a + 1 + i))
            stack.append((a + 1 + i, b))
    return ring[keep]

def check_outline(lines, scale):
    """내보내기 전에 경계가 실제로 이어져 있는지 확인한다.

    예전 방식은 고리를 토막 내어 내보냈고, 토막 사이가 몇십 km 씩
    벌어져도 아무도 몰랐다. 닫힌 고리인지, 걸음이 튀지 않는지 센다.
    """
    bad_open, steps = 0, []
    for a in lines:
        a = np.asarray(a, dtype=np.float64)
        if len(a) < 4 or not np.allclose(a[0], a[-1]):
            bad_open += 1
        d = np.hypot(np.diff(a[:, 0]) * scale * 111_320.0,
                     np.diff(a[:, 1]) * 111_132.0)
        if len(d):
            steps.append(d)
    if not steps:
        print("  !! 그릴 경계가 없습니다", flush=True)
        return
    d = np.concatenate(steps)
    print(f"  확인: 닫힌 고리 {len(lines) - bad_open}/{len(lines)}개, "
          f"점 {len(d) + len(lines):,}개, 걸음 중앙 {np.median(d):.0f}m "
          f"최대 {d.max():,.0f}m", flush=True)
    if bad_open:
        print(f"  !! 닫히지 않은 고리가 {bad_open}개 있습니다", flush=True)


# 해안선 자르기(115초)는 역 목록과 무관하다. 같은 추출본이면 결과가 같으므로,
# build_rail 을 다시 돌려 역 번호만 밀렸을 때는 2단계(역에 현 붙이기)만 하면
# 된다. 3분 16초가 몇 초로 준다.
#
#     REGION=<권역> ADMIN_REUSE=1 python src/build_admin.py
#
# 켤 때만 쓴다. PBF 를 새로 받았으면 그냥 전부 다시 돌린다.
def main() -> None:
    missing = [p for p in PBFS if not p.exists()]
    if missing:
        sys.exit("OSM 추출본이 없습니다: " + ", ".join(str(p) for p in missing))
    stops_path = BASE / "stops.json"
    if not stops_path.exists():
        sys.exit(f"역 목록이 없습니다: {stops_path}")

    # ADMIN_REUSE 는 "3) 권역 경계 자르기" 를 건너뛸지만 정한다. 행정경계
    # 자체는 추출본 캐시에서 오므로 늘 다시 쓴다.
    # 결과를 담을 자리. 예전에는 행정경계 캐시를 권역 폴더에 쓰면서 덩달아
    # 생겼는데, 캐시를 공용으로 옮긴 뒤로는 여기서 만들어야 한다.
    (BASE / "raw").mkdir(parents=True, exist_ok=True)
    reuse = os.environ.get("ADMIN_REUSE") == "1"
    print("1) 행정경계 추출", flush=True)
    polygons, labels = prefecture_polygons()
    if not polygons:
        sys.exit("admin_level=4 경계를 찾지 못했습니다.")

    print("2) 역에 현 붙이기", flush=True)
    stops = json.loads(stops_path.read_text(encoding="utf-8"))
    coords = np.array(stops["coords"], dtype=np.float64)
    names = assign(polygons, coords)

    found = sum(1 for n in names if n)
    payload = {
        "names": labels,
        "stations": {sid: n for sid, n in zip(stops["ids"], names) if n},
    }
    (BASE / "prefectures.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8"
    )

    # 다각형도 남긴다. 권역 경계를 현 경계로 삼을 때 쓴다. 전부 담으면
    # 무거우니 region.json 이 고른 현만, 그리고 BOUND_TOLERANCE_M 만큼
    # 줄여 담는다.
    want = set(json.loads((BASE / "region.json").read_text(encoding="utf-8"))
               .get("prefectures") or [])
    # 3단계는 현 경계와 해안선만 본다. 역 목록과 무관하므로 다시 쓸 때는
    # 이미 그려 둔 것을 그대로 둔다.
    if reuse and (BASE / "raw" / "prefecture-rings.json").exists():
        print("3) 그려 둔 권역 경계를 그대로 쓴다", flush=True)
        want = set()
    if want:
        from shapely import make_valid
        from shapely.geometry import Polygon, box
        from shapely.ops import unary_union

        picked = {k: v for k, v in polygons.items() if k in want}
        missing_pref = sorted(want - set(picked))
        if missing_pref:
            print("  !! 경계를 못 찾은 현: " + ", ".join(missing_pref), flush=True)
        if not picked:
            # 하나도 못 찾았으면 아래에서 빈 배열을 이어 붙이다 죽는다.
            # 이름이 어긋난 것이니 조용히 넘기지 말고 분명히 알린다.
            sys.exit("region.json 의 prefectures 이름이 OSM 경계와 하나도 "
                     "맞지 않습니다. 현 경계를 만들지 못했습니다.")

        print("3) 바다 쪽을 해안선으로 자르기", flush=True)
        X = np.concatenate([r for v in picked.values() for r in v])
        scale = float(np.cos(np.radians(float(np.median(X[:, 1])))))
        bb = box(X[:, 0].min() - 0.05, X[:, 1].min() - 0.05,
                 X[:, 0].max() + 0.05, X[:, 1].max() + 0.05)
        land_npz = BASE / "walk" / "land.npz"
        z = np.load(land_npz) if land_npz.exists() else None
        sea = land_polygons(bb, scale,
                            None if z is None else z["land"],
                            None if z is None else z["grid"])

        # 현 경계는 영해까지 뻗어 있다. 육지와 겹쳐야 해안선이 그대로
        # 경계가 된다. 현과 현 사이 경계는 건드리지 않는다.
        geo = {}
        for name, rings in picked.items():
            g = unary_union([make_valid(Polygon(r)) for r in rings if len(r) >= 4])
            geo[name] = g if sea is None else g.intersection(sea)

        out = {k: [simplify(r, BOUND_TOLERANCE_M).round(6).tolist()
                   for r in rings_of(v, scale)]
               for k, v in geo.items()}
        (BASE / "raw" / "prefecture-rings.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8")

        # 지도에 그릴 경계. 현을 하나로 합치면 안쪽 경계선은 저절로
        # 사라지고 바깥 윤곽만 닫힌 고리로 남는다.
        whole = unary_union(list(geo.values()))
        lines = boundary_rings(whole, scale)
        (BASE / "raw" / "prefecture-outline.json").write_text(
            json.dumps([simplify(l, BOUND_TOLERANCE_M).round(6).tolist()
                        for l in lines if len(l) >= 4], ensure_ascii=False),
            encoding="utf-8")
        n = sum(len(v) for v in out.values())
        print(f"  현 경계 {len(out)}개 현, 고리 {n}개 / 그릴 선 {len(lines)}개 저장",
              flush=True)
        check_outline(lines, scale)

    from collections import Counter
    tally = Counter(n for n in names if n)
    print(f"  {found:,} / {len(names):,} 건에 현을 붙였다", flush=True)
    for name, count in tally.most_common():
        print(f"    {name:<8} {count:>5}", flush=True)


if __name__ == "__main__":
    main()
