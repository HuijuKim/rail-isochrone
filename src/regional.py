"""권역별로 적어 둔 손질 사전을 현 조합 권역에도 쓴다.

line-extensions·line-headways·line-operators 는 "권역 -> 노선 -> ..." 로
적혀 있다. 현을 골라 만든 권역(make_region.py)은 id 가 사전에 없으므로,
고른 현과 겹치는 기존 권역의 항목을 모아 쓴다. 岡山·香川 조합이면 주고쿠와
시코쿠 것을 합친다. 항목은 역 이름으로 걸리므로 그 역이 권역에 없으면 아무
일도 하지 않는다.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS_DIR = ROOT / "data" / "regions"


@lru_cache(maxsize=None)
def _meta(region_id: str) -> dict:
    try:
        return json.loads((REGIONS_DIR / region_id / "region.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def prefectures_of(region_id: str) -> frozenset:
    return frozenset(_meta(region_id).get("prefectures") or ())


def is_custom(region_id: str) -> bool:
    """make_region.py 로 현을 골라 만든 권역인가."""
    return bool(_meta(region_id).get("custom"))


def neighbours(region_id: str, ids) -> list[str]:
    """ids 가운데 region_id 와 현을 하나라도 같이 가진 권역."""
    mine = prefectures_of(region_id)
    return [r for r in ids if r != region_id and mine & prefectures_of(r)]


def merge_books(book: dict, region_id: str) -> dict:
    """{권역: {노선: 항목}} 에서 이 권역이 쓸 {노선: 항목}.

    사전에 권역이 있으면 그것만 쓴다. 현 조합 권역이면 현이 겹치는 권역
    것을 합친다. 기존 권역은 제 항목이 없어도 이웃 것을 빌리지 않는다.
    같은 노선이 두 권역에 있으면 항목 안의 열쇠끼리 합치고, 목록은 이어
    붙인다. 같은 열쇠가 서로 다르면 먼저 나온 권역 것을 쓴다.
    """
    own = book.get(region_id)
    if isinstance(own, dict):
        return own
    if not is_custom(region_id):
        return {}
    merged: dict = {}
    for rid in neighbours(region_id, [k for k, v in book.items() if isinstance(v, dict)]):
        for line, spec in book[rid].items():
            if not isinstance(spec, dict):
                merged.setdefault(line, spec)
                continue
            cur = merged.setdefault(line, {})
            if not isinstance(cur, dict):
                continue
            for k, v in spec.items():
                if isinstance(v, list) and isinstance(cur.get(k), list):
                    cur[k] = cur[k] + [x for x in v if x not in cur[k]]
                else:
                    cur.setdefault(k, v)
    return merged


def book_for(path: Path, region_id: str) -> dict:
    """파일에서 읽어 merge_books 한 것. 파일이 없거나 깨졌으면 빈 사전."""
    try:
        book = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return merge_books(book, region_id)
