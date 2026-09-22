"""추출본 단위 훑기 캐시.

PBF 를 훑는 일은 빌드에서 가장 비싸고(사이타마 조합 20분 중 12분), 훑는
내용은 권역이 아니라 추출본에만 매인다. 그래서 결과를 권역 폴더가 아니라
data/cache/<종류>/<추출본 이름>.* 에 두고, 같은 추출본을 읽는 다른 권역이
그대로 가져다 쓴다. 권역마다 다른 거르기(현 경계 등)는 캐시를 읽은 뒤에
한다.

도장에는 PBF 의 크기·수정 시각과 읽는 코드의 해시가 들어간다. 추출본을 새로
받거나 읽는 코드를 고치면 저절로 다시 훑는다.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "cache"


def name_of(pbfs) -> str:
    return "+".join(sorted(Path(p).name.replace("-latest.osm.pbf", "")
                           for p in pbfs)) or "none"


def files(kind: str, pbfs, *exts: str) -> tuple:
    """캐시 파일 경로들. 폴더는 미리 만들어 둔다."""
    base = DIR / kind
    base.mkdir(parents=True, exist_ok=True)
    return tuple(base / (name_of(pbfs) + e) for e in (exts or (".npz", ".json")))


def reader_hash(*objs) -> str:
    """읽는 코드(핸들러·함수)의 소스 해시. 판 번호를 손으로 올리지 않아도 된다."""
    try:
        src = "".join(inspect.getsource(o) for o in objs)
    except (OSError, TypeError):
        return "?"
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]


def stamp(pbfs, *readers, v: int = 1) -> dict:
    def of(p):
        p = Path(p)
        try:
            st = p.stat()
            return [p.name, st.st_size, int(st.st_mtime)]
        except OSError:
            return [str(p), 0, 0]
    return {"v": v, "pbf": [of(p) for p in pbfs], "readers": reader_hash(*readers)}


def read_meta(path: Path, pbfs, *readers, v: int = 1):
    """도장이 맞으면 meta 를, 아니면 None 을."""
    try:
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if meta.get("stamp") == stamp(pbfs, *readers, v=v) else None
