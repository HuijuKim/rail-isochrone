"""원본 시각표 JSON을 라우팅용 압축 배열로 변환한다.

출력 (data/graph.npz + data/stops.json):
  - 정차 이벤트를 (trip, seq) 순으로 평탄화한 배열
  - 같은 역 그룹 안에서의 환승 간선
  - 역 좌표 / 이름 테이블

수도권 특유의 직통운전은 nt(next train) 체인을 따라가 하나의 연속 운행으로
병합한다. 이렇게 해야 시부야에서 한조몬선으로 갈아타지 않고 그대로 실려가는
승객에게 환승 패널티가 잘못 붙지 않는다.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REGION = os.environ.get("REGION", "kanto")
BASE = ROOT / "data" / "regions" / REGION
RAW = BASE / "raw"
TIMETABLES = RAW / "train-timetables"
OUT = BASE

# 같은 역 구내에서의 환승(플랫폼 이동)에 드는 시간
TRANSFER_SAME_COMPLEX_SEC = 180
# 같은 역명이지만 별개 건물/개찰구인 경우 (예: 고탄다의 JR / 도큐 / 도에이)
TRANSFER_CROSS_COMPLEX_SEC = 300

TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def parse_time(text: str) -> int | None:
    """'25:13' 같은 심야 표기를 자정 기준 초로 바꾼다."""
    m = TIME_RE.match(text)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    return hh * 3600 + mm * 60


def load_reference() -> tuple[dict, dict]:
    stations = json.loads((RAW / "stations.json").read_text(encoding="utf-8"))
    railways = json.loads((RAW / "railways.json").read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in stations}
    rail_by_id = {r["id"]: r for r in railways}
    return by_id, rail_by_id


def load_timetables(calendar: str) -> dict[str, dict]:
    """해당 달력(Weekday / SaturdayHoliday)의 운행만 모은다."""
    trips: dict[str, dict] = {}
    for path in sorted(TIMETABLES.glob("*.json")):
        for trip in json.loads(path.read_text(encoding="utf-8")):
            if trip["id"].rsplit(".", 1)[-1] != calendar:
                continue
            trips[trip["id"]] = trip
    return trips


def merge_through_services(trips: dict[str, dict]) -> list[list[dict]]:
    """nt 체인을 따라 직통운전을 하나의 연속 운행으로 잇는다."""
    has_prev = set()
    for trip in trips.values():
        for nxt in trip.get("nt", []):
            if nxt in trips:
                has_prev.add(nxt)

    chains: list[list[dict]] = []
    consumed: set[str] = set()

    for tid, trip in trips.items():
        if tid in has_prev or tid in consumed:
            continue  # 체인의 시작점만 출발선으로 삼는다
        chain: list[dict] = []
        cur: str | None = tid
        while cur is not None and cur in trips and cur not in consumed:
            consumed.add(cur)
            chain.append(trips[cur])
            # 분기하는 직통은 첫 번째 계승 편성만 따라간다
            nxts = [n for n in trips[cur].get("nt", []) if n in trips and n not in consumed]
            cur = nxts[0] if nxts else None
        chains.append(chain)

    # 체인 시작점을 못 찾은 순환 구조(야마노테선 등)는 개별 운행으로 남긴다
    for tid, trip in trips.items():
        if tid not in consumed:
            consumed.add(tid)
            chains.append([trip])

    return chains


def chain_to_stops(chain: list[dict]) -> list[tuple[str, int, int]]:
    """연결된 운행을 (역, 도착초, 출발초) 목록으로 편다."""
    stops: list[tuple[str, int, int]] = []
    for trip in chain:
        for entry in trip["tt"]:
            station = entry["s"]
            arr = parse_time(entry["a"]) if "a" in entry else None
            dep = parse_time(entry["d"]) if "d" in entry else None
            if arr is None and dep is None:
                continue
            arr = arr if arr is not None else dep
            dep = dep if dep is not None else arr
            # 직통 이음매에서 같은 역이 두 번 나오면 하나로 합친다
            if stops and stops[-1][0] == station:
                prev_station, prev_arr, _ = stops[-1]
                stops[-1] = (prev_station, prev_arr, dep)
                continue
            stops.append((station, arr, dep))

    # 자정을 넘겨 시각이 되감기면 24시간을 더해 단조 증가로 만든다
    fixed: list[tuple[str, int, int]] = []
    offset = 0
    last = -1
    for station, arr, dep in stops:
        a, d = arr + offset, dep + offset
        if a < last - 6 * 3600:  # 큰 폭의 되감김 = 날짜 변경
            offset += 24 * 3600
            a, d = arr + offset, dep + offset
        if d < a:
            d = a
        fixed.append((station, a, d))
        last = d
    return fixed


def build_station_index(by_id: dict) -> tuple[list[str], dict[str, int], np.ndarray]:
    ids = sorted(by_id)
    index = {sid: i for i, sid in enumerate(ids)}
    # 센다이/이와키처럼 간토 권역 밖 종착역은 좌표가 비어 있다. 경로 계산에는
    # 그대로 쓰되 지도에 그릴 때 빠지도록 NaN 으로 둔다.
    coords = np.array(
        [by_id[s].get("coord") or (np.nan, np.nan) for s in ids], dtype=np.float64
    )
    return ids, index, coords


def build_transfers(index: dict[str, int]) -> list[tuple[int, int, int]]:
    """역 그룹 정보로 환승 간선을 만든다.

    station-groups.json 의 한 그룹은 여러 하위 묶음으로 나뉘어 있는데, 같은
    묶음 안이면 같은 역 구내라 환승이 짧고, 묶음이 다르면 개찰구를 나갔다
    들어와야 해서 더 걸린다.
    """
    groups = json.loads((RAW / "station-groups.json").read_text(encoding="utf-8"))
    edges: list[tuple[int, int, int]] = []

    for group in groups:
        clusters = [[index[s] for s in sub if s in index] for sub in group]
        flat = [(ci, node) for ci, sub in enumerate(clusters) for node in sub]
        for ci, a in flat:
            for cj, b in flat:
                if a == b:
                    continue
                cost = (
                    TRANSFER_SAME_COMPLEX_SEC
                    if ci == cj
                    else TRANSFER_CROSS_COMPLEX_SEC
                )
                edges.append((a, b, cost))

    # 같은 좌표를 공유하는 동일역 (그룹 파일에 안 잡힌 경우) 보강
    return edges


def build(calendar: str = "Weekday") -> dict:
    by_id, _ = load_reference()
    ids, index, coords = build_station_index(by_id)

    print(f"[{calendar}] 시각표 적재 중...", flush=True)
    trips = load_timetables(calendar)
    print(f"  개별 운행 {len(trips):,}건", flush=True)

    chains = merge_through_services(trips)
    print(f"  직통 병합 후 {len(chains):,}건", flush=True)

    # 정차 이벤트 평탄화
    ev_stop: list[int] = []
    ev_arr: list[int] = []
    ev_dep: list[int] = []
    trip_start: list[int] = []
    skipped = 0

    for chain in chains:
        stops = chain_to_stops(chain)
        if len(stops) < 2:
            skipped += 1
            continue
        missing = [s for s, _, _ in stops if s not in index]
        if missing:
            skipped += 1
            continue
        trip_start.append(len(ev_stop))
        for station, arr, dep in stops:
            ev_stop.append(index[station])
            ev_arr.append(arr)
            ev_dep.append(dep)
    trip_start.append(len(ev_stop))

    print(f"  정차 이벤트 {len(ev_stop):,}개 (제외 {skipped:,}건)", flush=True)

    transfers = build_transfers(index)
    print(f"  환승 간선 {len(transfers):,}개", flush=True)

    tr = np.array(transfers, dtype=np.int32) if transfers else np.zeros((0, 3), np.int32)
    order = np.lexsort((tr[:, 1], tr[:, 0])) if len(tr) else np.array([], dtype=int)
    tr = tr[order]
    tr_ptr = np.searchsorted(tr[:, 0], np.arange(len(ids) + 1)) if len(tr) else np.zeros(len(ids) + 1, np.int64)

    return {
        "calendar": calendar,
        "ids": ids,
        "coords": coords,
        "titles_ja": [by_id[s]["title"].get("ja", "") for s in ids],
        "titles_en": [by_id[s]["title"].get("en", "") for s in ids],
        "titles_ko": [by_id[s]["title"].get("ko", "") for s in ids],
        "railway": [by_id[s].get("railway", "") for s in ids],
        "ev_stop": np.array(ev_stop, dtype=np.int32),
        "ev_arr": np.array(ev_arr, dtype=np.int32),
        "ev_dep": np.array(ev_dep, dtype=np.int32),
        "trip_start": np.array(trip_start, dtype=np.int32),
        "tr_to": tr[:, 1].astype(np.int32) if len(tr) else np.zeros(0, np.int32),
        "tr_cost": tr[:, 2].astype(np.int32) if len(tr) else np.zeros(0, np.int32),
        "tr_ptr": tr_ptr.astype(np.int64),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for calendar in ("Weekday", "SaturdayHoliday"):
        g = build(calendar)
        np.savez_compressed(
            OUT / f"graph-{calendar}.npz",
            coords=g["coords"],
            ev_stop=g["ev_stop"],
            ev_arr=g["ev_arr"],
            ev_dep=g["ev_dep"],
            trip_start=g["trip_start"],
            tr_to=g["tr_to"],
            tr_cost=g["tr_cost"],
            tr_ptr=g["tr_ptr"],
        )
        (OUT / "stops.json").write_text(
            json.dumps(
                {
                    "ids": g["ids"],
                    "coords": g["coords"].tolist(),
                    "ja": g["titles_ja"],
                    "en": g["titles_en"],
                    "ko": g["titles_ko"],
                    "railway": g["railway"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        size = (OUT / f"graph-{calendar}.npz").stat().st_size
        print(f"  저장 완료 graph-{calendar}.npz  {size/1e6:.1f} MB\n", flush=True)


if __name__ == "__main__":
    main()
