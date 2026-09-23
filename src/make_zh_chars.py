"""역·노선 이름을 중국어로 옮길 글자표를 만든다. data/zh-chars.json 을 쓴다.

일본 역 이름은 중국어로도 한자를 그대로 쓰되 글자꼴만 바꿔 적는다(広島 ->
廣島 / 广岛). 서버가 이 표로 글자를 하나씩 바꾼다. 표를 다시 만들 때만 이
스크립트를 돌리고, OpenCC 가 있어야 한다(pip install opencc). 서버와 빌드는
OpenCC 없이 표만 읽는다.

    python src/make_zh_chars.py

OpenCC 의 jp2t(일본 신자체 -> 번체)와 t2s(번체 -> 간체)를 글자마다 거친 뒤,
ODPT 공식 중국어 역 이름과 견줘 어긋나는 것을 손으로 고친 값(FIX)을 덮는다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import opencc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "zh-chars.json"
HAN = re.compile(r"[㐀-鿿豈-﫿]")

# OpenCC 가 안 바꾸거나 다른 이체자로 바꾸는 것. ODPT 공식 이름을 따른다.
#   글자 -> (번체, 간체)
FIX = {
    "戸": ("戶", "户"),        # 亀戸 -> 龜戶 / 龟户
    "郷": ("鄉", "乡"),        # jp2t 가 鄕 로 바꾼다
    "並": ("並", "并"),        # jp2t 가 竝 로 바꾼다
    "台": ("台", "台"),        # jp2t 가 臺 로 바꾼다. 지명은 台 로 적는다
    "駄": ("馱", "驮"),
    "渋": ("澀", "涩"),        # jp2t 가 澁 로 바꾸고 t2s 가 못 받는다
    "黒": ("黑", "黑"),
    "徳": ("德", "德"),
    "瀬": ("瀨", "濑"),
    "鷹": ("鷹", "鹰"),
    "俣": ("俁", "俣"),        # jp2t 가 안 바꾼다(二俣川 -> 二俁川)
    "内": ("內", "内"),
    "真": ("真", "真"),        # jp2t 가 眞 로 바꾼다
    "姫": ("姬", "姬"),
    "弁": ("弁", "弁"),        # jp2t 가 辨 으로 바꾼다(弁天橋)
    "駅": ("站", "站"),        # 大塚駅前 -> 大塚站前
    "氷": ("冰", "冰"),        # 氷川台 -> 冰川台
    "楡": ("榆", "榆"),
    "栢": ("栢", "柏"),
    "栃": ("栃", "枥"),        # 栃木 -> 枥木
}
# 글자 하나로는 못 가르는 낱말. 記念 은 중국어로 紀念 이다.
WORDS = {
    "記念": ("紀念", "纪念"),
    "地下鉄": ("地鐵", "地铁"),
    "空港": ("機場", "机场"),
}


def names():
    """빌드해 둔 권역과 이름 사전에 나오는 모든 일본어 이름."""
    got = set()
    for d in (ROOT / "data" / "regions").iterdir():
        for f in ("raw/stations.json", "raw/railways.json"):
            p = d / f
            if p.exists():
                for r in json.loads(p.read_text(encoding="utf-8")):
                    got.add(((r.get("title") or {}).get("ja") or ""))
    book = json.loads((ROOT / "data" / "station-names.json").read_text(encoding="utf-8"))
    got.update(k for k in book if k != "note")
    return got


def main() -> None:
    jp2t, t2s = opencc.OpenCC("jp2t"), opencc.OpenCC("t2s")
    chars = sorted({c for n in names() for c in n if HAN.match(c)})
    table = {}
    for c in chars:
        if c in FIX:
            t, s = FIX[c]
        else:
            t = jp2t.convert(c)
            s = t2s.convert(t)
            # 확장 영역 한자(鉾 -> 𫓴)는 글꼴이 없는 곳이 많아 원래 글자로 둔다
            t = t if all(ord(x) < 0x20000 and not 0x3400 <= ord(x) < 0x4DC0 for x in t) else c
            s = s if all(ord(x) < 0x20000 and not 0x3400 <= ord(x) < 0x4DC0 for x in s) else t
        if (t, s) != (c, c):
            table[c] = [t, s]
    OUT.write_text(json.dumps({
        "note": "일본 한자 -> [번체, 간체]. 바뀌는 글자만 적는다. src/make_zh_chars.py 가"
                " OpenCC 로 만들고 ODPT 공식 이름과 어긋나는 것을 손으로 고쳤다. 가나가 든"
                " 이름은 옮기지 않는다(あざみ野 -> 蓟野 처럼 뜻으로 옮겨야 한다).",
        "chars": table,
        "words": {k: list(v) for k, v in WORDS.items()},
    }, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"글자 {len(chars):,}개 중 바뀌는 것 {len(table):,}개 -> {OUT}")


if __name__ == "__main__":
    main()
