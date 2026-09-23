"""현 조합 권역의 노선 이름을 기존 권역에서 빌려 적는다.

현 조합은 기존 권역과 같은 OSM 관계로 노선을 다시 짠다. 그런데 이름은 OSM 에
일본어만 있는 노선이 많아, 기존 권역은 따로 채워 둔 이름을 쓴다. 간토OSM 은
시각표 권역에서 빌려 온 raw/line-names.json 이 있다. 현 조합은 그 경로를 안
타서, 사이타마·지바·도쿄·가나가와 조합은 노선 136개 중 65개가 한국어 이름
없이 일본어로 나왔다.

일본어 이름과 운영사가 같은 기존 권역 노선의 이름을 가져와 이 권역의
raw/line-names.json 에 적는다. region.py 가 빈 언어만 이것으로 메운다.

    REGION=<권역> python src/build_names.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ROOT / "data" / "regions"
REGION = os.environ.get("REGION", "")
LANGS = ("en", "ko", "zh-Hans", "zh-Hant")

# 이름 끝의 방향·구간 괄호는 떼고 견준다. 예전에 빌드한 권역에는
# "阿武隈急行線（下り）" 처럼 방향이 남아 있고, 새로 빌드한 조합에는 없다.
_DIR = re.compile(r"\s*[(（]\s*(?:内回り|外回り|右回り|左回り|上り|下り|内回|外回)\s*[)）]\s*$")
_ARROW = re.compile(r"\s*[(（][^()（）]*(?:=>|->|→|⇒)[^()（）]*[)）]?\s*$")


_COLON = re.compile(r"\s*[:：][^:：]*(?:=>|->|→|⇒)[^:：]*$")


def _key(ja: str) -> str:
    s = _COLON.sub("", (ja or "").strip())
    return _ARROW.sub("", _DIR.sub("", s)).strip()


def named_lines() -> dict:
    """기존 권역들의 (일본어 이름, 운영사) -> 언어별 이름. 먼저 찾은 것이 이긴다."""
    out: dict = {}
    for base in sorted(REGIONS.iterdir()):
        meta_path = base / "region.json"
        rail_path = base / "raw" / "railways.json"
        if not (meta_path.exists() and rail_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("custom") or base.name == REGION:
            continue
        lent_path = base / "raw" / "line-names.json"
        lent = (json.loads(lent_path.read_text(encoding="utf-8"))
                if lent_path.exists() else {})
        for r in json.loads(rail_path.read_text(encoding="utf-8")):
            title = dict(r.get("title") or {})
            for lang, v in (lent.get(r["id"]) or {}).items():
                if lang in LANGS and not (title.get(lang) or "").strip():
                    title[lang] = v
            ja = _key(title.get("ja"))
            if not ja:
                continue
            slot = out.setdefault((ja, (r.get("operator") or "").strip()), {})
            for lang in LANGS:
                v = (title.get(lang) or "").strip()
                if v and v != ja and lang not in slot:
                    slot[lang] = v
    return out


def main() -> None:
    base = REGIONS / REGION
    meta = json.loads((base / "region.json").read_text(encoding="utf-8"))
    if not meta.get("custom"):
        print("현 조합 권역이 아니라 건너뛴다", flush=True)
        return
    known = named_lines()
    # 운영사 표기는 관계마다 흔들린다(関鉄/関東鉄道, 비어 있는 것). 일본어
    # 이름이 기존 권역 전체에서 하나뿐이면 운영사가 달라도 같은 노선으로 본다.
    by_ja: dict = {}
    for (ja, _op), names in known.items():
        by_ja.setdefault(ja, []).append(names)
    lent: dict = {}
    for r in json.loads((base / "raw" / "railways.json").read_text(encoding="utf-8")):
        title = r.get("title") or {}
        ja = _key(title.get("ja"))
        got = known.get((ja, (r.get("operator") or "").strip()))
        if not got and len(by_ja.get(ja, [])) == 1:
            got = by_ja[ja][0]
        if not got:
            continue
        # 빈 언어만 채운다. 이 권역이 제 이름을 들고 있으면 그것이 이긴다.
        fill = {lang: v for lang, v in got.items() if not (title.get(lang) or "").strip()}
        if fill:
            lent[r["id"]] = fill
    (base / "raw" / "line-names.json").write_text(
        json.dumps(lent, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {lang: sum(1 for v in lent.values() if lang in v) for lang in LANGS}
    print(f"[{REGION}] 기존 권역에서 빌린 노선 이름 {len(lent)}개 "
          + ", ".join(f"{k} {v}" for k, v in counts.items()), flush=True)


if __name__ == "__main__":
    main()
