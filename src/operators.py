"""역 ID 접두사에 대응하는 철도 운영사 이름.

mini-tokyo-3d 의 operators.json 은 항공사만 담고 있어서 철도사 이름은
따로 둔다. 검색 목록에서 같은 역명을 구분하는 데 쓴다.
"""
from __future__ import annotations

# 접두사 -> (일본어, 한국어)
OPERATORS: dict[str, tuple[str, str]] = {
    "Aizu": ("会津鉄道", "아이즈 철도"),
    "ChibaMonorail": ("千葉モノレール", "지바 모노레일"),
    "Chichibu": ("秩父鉄道", "지치부 철도"),
    "Choshi": ("銚子電鉄", "조시 전철"),
    "Enoden": ("江ノ電", "에노덴"),
    "Fujikyu": ("富士急行", "후지큐행"),
    "Hitachinaka": ("ひたちなか海浜鉄道", "히타치나카 해변철도"),
    "Hokuso": ("北総鉄道", "호쿠소 철도"),
    "Isumi": ("いすみ鉄道", "이스미 철도"),
    "IzuHakone": ("伊豆箱根鉄道", "이즈하코네 철도"),
    "Izukyu": ("伊豆急行", "이즈큐행"),
    "JR-Central": ("JR東海", "JR도카이"),
    "JR-East": ("JR東日本", "JR동일본"),
    "JR-Shikoku": ("JR四国", "JR시코쿠"),
    "JR-West": ("JR西日本", "JR서일본"),
    "Jomo": ("上毛電鉄", "조모 전철"),
    "Joshin": ("上信電鉄", "조신 전철"),
    "KantoRailway": ("関東鉄道", "간토 철도"),
    "KashimaRinkai": ("鹿島臨海鉄道", "가시마 임해철도"),
    "Keikyu": ("京急", "게이큐"),
    "Keio": ("京王", "게이오"),
    "Keisei": ("京成", "게이세이"),
    "Kominato": ("小湊鉄道", "고미나토 철도"),
    "MIR": ("つくばエクスプレス", "쓰쿠바 익스프레스"),
    "Minatomirai": ("みなとみらい線", "미나토미라이선"),
    "Moka": ("真岡鐵道", "모카 철도"),
    "Odakyu": ("小田急", "오다큐"),
    "OdakyuHakone": ("箱根登山鉄道", "하코네 등산철도"),
    "Ryutetsu": ("流鉄", "류테쓰"),
    "SaitamaRailway": ("埼玉高速鉄道", "사이타마 고속철도"),
    "SaitamaTransit": ("埼玉新都市交通", "사이타마 신도시교통"),
    "Seibu": ("西武", "세이부"),
    "Shibayama": ("芝山鉄道", "시바야마 철도"),
    "ShonanMonorail": ("湘南モノレール", "쇼난 모노레일"),
    "Sotetsu": ("相鉄", "소테쓰"),
    "TWR": ("りんかい線", "린카이선"),
    "TamaMonorail": ("多摩モノレール", "다마 모노레일"),
    "Tobu": ("東武", "도부"),
    "Toei": ("都営", "도에이"),
    "TokyoMetro": ("東京メトロ", "도쿄메트로"),
    "TokyoMonorail": ("東京モノレール", "도쿄 모노레일"),
    "Tokyu": ("東急", "도큐"),
    "ToyoRapid": ("東葉高速鉄道", "도요 고속철도"),
    "UtsunomiyaLightRail": ("宇都宮ライトレール", "우쓰노미야 라이트레일"),
    "WataraseKeikoku": ("わたらせ渓谷鐵道", "와타라세 계곡철도"),
    "Yagan": ("野岩鉄道", "야간 철도"),
    "Yamaman": ("山万", "야마만"),
    "YokohamaMunicipal": ("横浜市営地下鉄", "요코하마 시영지하철"),
    "YokohamaSeaside": ("金沢シーサイドライン", "가나자와 시사이드라인"),
    "Yurikamome": ("ゆりかもめ", "유리카모메"),
}

# 목록에서 먼저 보이면 좋은 큰 사업자 순서
PRIORITY = [
    "JR-East", "JR-Central", "TokyoMetro", "Toei",
    "Tokyu", "Odakyu", "Keio", "Seibu", "Tobu", "Keikyu", "Keisei",
]
_RANK = {name: i for i, name in enumerate(PRIORITY)}


def operator_of(station_id: str) -> str:
    """'JR-East.Yamanote.Shinjuku' -> 'JR-East'"""
    return station_id.split(".", 1)[0]


def title(prefix: str, lang: str = "ja") -> str:
    entry = OPERATORS.get(prefix)
    if entry is None:
        return prefix
    return entry[0] if lang == "ja" else entry[1]


def sort_key(prefix: str) -> tuple[int, str]:
    return (_RANK.get(prefix, len(PRIORITY)), prefix)
