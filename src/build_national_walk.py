"""전국 보행망을 1도 타일로 나눠 빌드하고 이어 붙인다(실험).

    python src/build_national_walk.py            # 그래프와 그리기 기하
    python src/build_national_walk.py --sheds    # 기존 권역 역들의 도보권까지

결과는 data/national/walk/ 에 권역 walk/ 와 같은 꼴로 쓴다.

격자는 전국 하나(NATIONAL)라 칸 번호가 어디서든 같다. 타일마다 둘레 MARGIN
도를 더 읽어 그래프를 만들고, 칸 중심이 타일 안에 있는 노드와 그 노드에서
나가는 간선만 남긴다. 읽은 범위 끝에서 길이 잘려 생기는 흠은 더 읽은 테두리에만
생기므로, 안쪽만 모으면 한 번에 빌드한 것과 같다(岡山·広島 로 확인했다).
도보권도 타일마다 그 안의 역만 계산한다. 40분 도보권(약 3 km)이 테두리
(약 10 km) 안에 들어온다.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "national"
# 오키나와를 빼야 40 m 칸 번호가 int32 안에 든다(16.2억 칸).
NATIONAL = {"lon0": 128.4, "lat0": 30.9, "lat_ref": 36.0,
            "span_x": 1_586_000.0, "span_y": 1_634_000.0}
TILE_DEG = 1.0
MARGIN = 0.1

# build_walk 는 격자를 올릴 때 권역 설정에서 읽는다. 전국 격자를 적어 두고 연다.
# 도보권을 나눠 도는 프로세스들도 이 파일을 다시 연다. 이미 같으면 쓰지 않는다
# (쓰는 도중에 다른 프로세스가 빈 파일을 읽는다).
(BASE / "walk").mkdir(parents=True, exist_ok=True)
_conf = json.dumps({"grid": NATIONAL})
if not (BASE / "region.json").exists() or (BASE / "region.json").read_text(encoding="utf-8") != _conf:
    (BASE / "region.json").write_text(_conf, encoding="utf-8")
os.environ["REGION"] = str(BASE)
sys.path.insert(0, str(ROOT / "src"))
import build_walk as bw  # noqa: E402

OUT = BASE / "walk"
TILES = BASE / "tiles"
# 이 노트북은 약정 메모리 여유가 6 GB 남짓이다. 도보권 프로세스를 반으로 줄인다.
bw.SHED_WORKERS = min(bw.SHED_WORKERS, 4)


def load_extracts() -> list[dict]:
    """추출본마다 보행로(정수 좌표)를 올린다. 캐시가 없으면 훑는다."""
    parts = []
    for path in sorted((ROOT / "data" / "osm").glob("*-latest.osm.pbf")):
        x, y, bounds, steps = bw._walk_part(path)
        if len(bounds) < 2:
            continue
        parts.append({"name": path.name, "x": x, "y": y,
                      "bounds": bounds.astype(np.int64), "steps": steps,
                      "box": (int(x.min()), int(y.min()), int(x.max()), int(y.max()))})
    return parts


def window_ways(parts: list[dict], w) -> tuple[tuple, np.ndarray]:
    """범위 [서, 남, 동, 북] 안의 보행로. build_walk.extract_ways 와 같은 규칙이다."""
    x0, y0, x1, y1 = (int(round(v * 1e7)) for v in w)
    lons, lats, counts, steps = [], [], [], []
    for p in parts:
        bx0, by0, bx1, by1 = p["box"]
        if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
            continue
        x, y, b = p["x"], p["y"], p["bounds"]
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if not inside.any():
            continue
        n_in = np.add.reduceat(inside.astype(np.int64), b[:-1])
        ok = n_in >= 2
        take = inside & np.repeat(ok, np.diff(b))
        lons.append(x[take].astype(np.float64) / 1e7)
        lats.append(y[take].astype(np.float64) / 1e7)
        counts.append(n_in[ok])
        steps.append(p["steps"][ok])
    if not lons:
        return None, None
    n = np.concatenate(counts)
    bounds = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    return (np.concatenate(lons), np.concatenate(lats), bounds), np.concatenate(steps)


def in_box(lon, lat, box) -> np.ndarray:
    """반열린 상자 [서, 동) x [남, 북). 타일 경계의 칸이 두 타일에 들지 않게."""
    return (lon >= box[0]) & (lon < box[2]) & (lat >= box[1]) & (lat < box[3])


def build_tile(parts, core, stations=None) -> dict | None:
    """core 상자의 노드·간선·그리기 선분(과 core 안 역의 도보권)."""
    w = (core[0] - MARGIN, core[1] - MARGIN, core[2] + MARGIN, core[3] + MARGIN)
    points, steps = window_ways(parts, w)
    if points is None:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        graph = bw.prune_small_components(bw.build_graph(points, steps))
    if graph["n_nodes"] == 0:
        return None
    cell = graph["node_cell"]
    clon, clat = bw.cell_center(cell)
    mine = in_box(clon, clat, core)
    src, dst, cost = graph["src"], graph["dst"], graph["cost"]
    ek = mine[src]
    out = {"cell": cell[mine], "lon": graph["node_lon"][mine], "lat": graph["node_lat"][mine],
           "e_src": cell[src[ek]], "e_dst": cell[dst[ek]], "e_cost": cost[ek]}

    # 그리기 선분: 첫 점이 core 안인 것. 선분 끝점까지 남기고 나머지 점은 버린다.
    lon, lat, bounds = points
    seg = np.arange(len(lon) - 1, dtype=np.int64)
    boundary = np.zeros(len(lon) - 1, dtype=bool)
    boundary[bounds[1:-1] - 1] = True
    seg = seg[~boundary]
    seg = seg[in_box(lon[seg], lat[seg], core)]
    need = np.zeros(len(lon), dtype=bool)
    need[seg] = True
    need[seg + 1] = True
    new_of = np.cumsum(need) - 1
    out["pts"] = np.stack([np.rint(lon[need] * 1e7), np.rint(lat[need] * 1e7)],
                          axis=1).astype(np.int32)
    out["seg"] = new_of[seg].astype(np.int32)
    out["seg_tile"] = bw.fine_tile_index(lon[seg], lat[seg]).astype(np.int32)

    if stations is not None:
        sel = np.flatnonzero(in_box(stations[:, 0], stations[:, 1], core))
        out["st_index"] = sel
        if len(sel):
            indptr, indices, data = bw.to_csr(graph)
            node_shed = bw.shed_cell_index(clon, clat).astype(np.int32)
            with contextlib.redirect_stdout(io.StringIO()):
                snapped = bw.snap_stations(cell, stations[sel])
                out["sheds"] = bw.build_sheds(indptr, indices, data, graph["n_nodes"],
                                              node_shed, snapped)
            # 역이 붙은 노드를 전국 번호로 옮기려고 칸 번호로 들고 나온다
            out["st_cell"] = np.where(snapped >= 0, cell[np.clip(snapped, 0, None)], -1)
            sh = out.pop("sheds")
            out.update(shed_cell=sh["shed_cell"], shed_sec=sh["shed_sec"],
                       shed_ptr=sh["shed_ptr"])
    return out


def preset_stations() -> np.ndarray:
    """기존 권역(조합·ODPT 간토·한국 빼고) 역 좌표. 같은 자리는 하나로."""
    coords = []
    for d in sorted((ROOT / "data" / "regions").iterdir()):
        if d.name.startswith("custom_") or d.name in ("kanto", "korea"):
            continue
        f = d / "stops.json"
        if f.exists():
            c = np.array(json.loads(f.read_text(encoding="utf-8"))["coords"], dtype=np.float64)
            coords.append(c[np.isfinite(c[:, 0])])
    c = np.concatenate(coords)
    _, first = np.unique(np.round(c, 5), axis=0, return_index=True)
    return c[np.sort(first)]


def build_tiles(stations) -> list[Path]:
    """타일마다 결과를 tiles/ 에 쓴다. 메모리에 쌓지 않는다."""
    print("1) 추출본 보행로 올리기", flush=True)
    t0 = time.time()
    parts = load_extracts()
    print(f"  {len(parts)}개, 점 {sum(len(p['x']) for p in parts):,}개 "
          f"({time.time() - t0:.0f}s)", flush=True)
    print("2) 타일", flush=True)
    TILES.mkdir(parents=True, exist_ok=True)
    for f in TILES.glob("*.npz"):
        f.unlink()
    tiles = [(float(x), float(y), float(x) + TILE_DEG, float(y) + TILE_DEG)
             for y in np.arange(np.floor(NATIONAL["lat0"]), bw.GRID_LAT1, TILE_DEG)
             for x in np.arange(np.floor(NATIONAL["lon0"]), bw.GRID_LON1, TILE_DEG)]
    files = []
    t1 = time.time()
    for core in tiles:
        r = build_tile(parts, core, stations)
        if r is None or not len(r["cell"]):
            continue
        f = TILES / f"{core[0]:.0f}_{core[1]:.0f}.npz"
        np.savez(f, **r)
        files.append(f)
        extra = f" 역 {len(r['st_index'])}" if "st_index" in r else ""
        print(f"  {core[0]:.0f}E {core[1]:.0f}N 노드 {len(r['cell']):,}{extra} "
              f"({time.time() - t1:.0f}s)", flush=True)
    print(f"  타일 {len(files)}개 ({time.time() - t1:.0f}s)", flush=True)
    return files


def merge_graph(files) -> np.ndarray:
    """노드는 칸 번호 순(walknet.nearest_node 가 이분 탐색한다), 간선은 CSR."""
    print("3) 그래프 이어 붙이기", flush=True)
    cells, lons, lats = [], [], []
    for f in files:
        z = np.load(f)
        cells.append(z["cell"])
        lons.append(z["lon"])
        lats.append(z["lat"])
    cell = np.concatenate(cells)
    order = np.argsort(cell, kind="stable")
    cell = cell[order]
    assert not (cell[1:] == cell[:-1]).any(), "두 타일이 같은 칸을 가졌다"
    node_lon = np.concatenate(lons)[order]
    node_lat = np.concatenate(lats)[order]
    del cells, lons, lats, order

    src, dst, cost = [], [], []
    dropped = 0
    for f in files:
        z = np.load(f)
        si = np.searchsorted(cell, z["e_src"])
        di = np.clip(np.searchsorted(cell, z["e_dst"]), 0, len(cell) - 1)
        ok = cell[di] == z["e_dst"]      # 이웃 타일에서 파편으로 걷힌 칸이면 버린다
        dropped += int((~ok).sum())
        src.append(si[ok].astype(np.int32))
        dst.append(di[ok].astype(np.int32))
        cost.append(z["e_cost"][ok])
    graph = {"n_nodes": len(cell), "src": np.concatenate(src),
             "dst": np.concatenate(dst), "cost": np.concatenate(cost)}
    del src, dst, cost
    indptr, indices, data = bw.to_csr(graph)
    del graph
    nlon, nlat = bw.cell_center(cell)
    node_shed = bw.shed_cell_index(nlon, nlat).astype(np.int32)
    del nlon, nlat
    np.savez_compressed(
        OUT / "graph.npz", node_cell=cell, node_lon=node_lon, node_lat=node_lat,
        node_shed=node_shed, indptr=indptr, indices=indices, data=data,
        grid=np.array([bw.GRID_LON0, bw.GRID_LAT0, bw.CELL_M, bw.GRID_W, bw.GRID_H,
                       bw.M_PER_DEG_LON, bw.M_PER_DEG_LAT]),
        shed_grid=np.array([bw.GRID_LON0, bw.GRID_LAT0, bw.SHED_CELL_M, bw.SHED_W,
                            bw.SHED_H, bw.M_PER_DEG_LON, bw.M_PER_DEG_LAT]))
    print(f"  노드 {len(cell):,}, 간선 {len(indices):,} (이웃 없어 버린 간선 {dropped:,}), "
          f"graph.npz {(OUT / 'graph.npz').stat().st_size / 1e6:.0f} MB", flush=True)
    return cell


def merge_fine(files) -> None:
    """그리기 선분. 점은 파일에 바로 채우고, 선분만 타일 순으로 정렬한다."""
    print("4) 그리기 기하 이어 붙이기", flush=True)
    sizes = [int(np.load(f)["pts"].shape[0]) for f in files]
    pts = np.lib.format.open_memmap(OUT / "fine_pt.npy", mode="w+", dtype=np.int32,
                                    shape=(sum(sizes), 2))
    segs, tiles = [], []
    off = 0
    for f, n in zip(files, sizes):
        z = np.load(f)
        pts[off:off + n] = z["pts"]
        keep = z["seg_tile"] >= 0
        segs.append(z["seg"][keep] + np.int32(off))
        tiles.append(z["seg_tile"][keep])
        off += n
    pts.flush()
    del pts
    seg = np.concatenate(segs)
    tile = np.concatenate(tiles)
    del segs, tiles
    order = np.argsort(tile, kind="stable")
    seg, tile = seg[order], tile[order]
    del order
    ptr = np.searchsorted(tile, np.arange(bw.FINE_W * bw.FINE_H + 1, dtype=np.int64)).astype(np.int32)
    np.save(OUT / "fine_seg.npy", seg)
    np.save(OUT / "fine_ptr.npy", ptr)
    np.save(OUT / "fine_grid.npy", np.array([bw.GRID_LON0, bw.GRID_LAT0, bw.FINE_TILE_M,
                                             bw.FINE_W, bw.FINE_H, bw.M_PER_DEG_LON,
                                             bw.M_PER_DEG_LAT]))
    mb = sum((OUT / f).stat().st_size for f in
             ("fine_pt.npy", "fine_seg.npy", "fine_ptr.npy")) / 1e6
    print(f"  점 {off:,}, 선분 {len(seg):,}, {mb:.0f} MB", flush=True)


def merge_sheds(files, cell, stations) -> None:
    print("5) 도보권 이어 붙이기", flush=True)
    n = len(stations)
    st_cell = np.full(n, -1, dtype=np.int64)
    per = [None] * n
    for f in files:
        z = np.load(f)
        if "shed_ptr" not in z.files:
            continue
        ptr, sc, ss = z["shed_ptr"], z["shed_cell"], z["shed_sec"]
        for k, i in enumerate(z["st_index"]):
            per[i] = (sc[ptr[k]:ptr[k + 1]], ss[ptr[k]:ptr[k + 1]])
        st_cell[z["st_index"]] = z["st_cell"]
    empty = (np.zeros(0, np.int32), np.zeros(0, np.float32))
    per = [p or empty for p in per]
    ptr = np.concatenate([[0], np.cumsum([len(p[0]) for p in per])]).astype(np.int32)
    at = np.clip(np.searchsorted(cell, st_cell), 0, len(cell) - 1)
    station_node = np.where((st_cell >= 0) & (cell[at] == st_cell), at, -1).astype(np.int32)
    np.savez_compressed(OUT / "sheds.npz", station_node=station_node,
                        shed_cell=np.concatenate([p[0] for p in per]),
                        shed_sec=np.concatenate([p[1] for p in per]), shed_ptr=ptr,
                        station_coords=stations)
    print(f"  {int((station_node >= 0).sum()):,}/{n:,}역, sheds.npz "
          f"{(OUT / 'sheds.npz').stat().st_size / 1e6:.0f} MB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheds", action="store_true", help="기존 권역 역들의 도보권까지")
    ap.add_argument("--merge-only", action="store_true", help="tiles/ 에 있는 것만 이어 붙인다")
    args = ap.parse_args()
    import psutil

    bw.check_grid_fits()
    t0 = time.time()
    stations = preset_stations() if args.sheds else None
    if stations is not None:
        print(f"역 {len(stations):,}곳", flush=True)
    files = sorted(TILES.glob("*.npz")) if args.merge_only else build_tiles(stations)
    cell = merge_graph(files)
    merge_fine(files)
    if stations is not None:
        merge_sheds(files, cell, stations)
    peak = psutil.Process().memory_info().peak_wset / 1e9
    print(f"끝 ({(time.time() - t0) / 60:.1f}분, 최대 메모리 {peak:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
