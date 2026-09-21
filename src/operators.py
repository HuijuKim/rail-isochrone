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


# 노선 id 앞머리와 회사 표를 잇는다. 같은 회사가 권역마다 다른 이름으로
# 불리면 디버그 패널에서 무리가 갈린다. 시각표 권역은 都営 를 "도에이",
# OSM 권역은 "도영지하철" 로 불렀다. 아래에 짝이 있으면 회사 표를 따른다.
# 여기 없는 것은 OPERATORS 에 적힌 그대로 쓴다(시각표 권역에만 있는
# 회사들이다).
PREFIX_COMPANY: dict[str, str] = {
    "JR-East": "東日本旅客鉄道", "JR-West": "西日本旅客鉄道",
    "JR-Central": "東海旅客鉄道",
    "TokyoMetro": "東京地下鉄", "Toei": "東京都交通局",
    "YokohamaMunicipal": "横浜市交通局",
    "Tobu": "東武鉄道", "Seibu": "西武鉄道", "Keisei": "京成電鉄",
    "Keio": "京王電鉄", "Odakyu": "小田急電鉄", "Tokyu": "東急電鉄",
    "Keikyu": "京浜急行電鉄", "Sotetsu": "相模鉄道",
    "Chichibu": "秩父鉄道", "Choshi": "銚子電気鉄道",
    "Enoden": "江ノ島電鉄", "Hokuso": "北総鉄道",
    "Isumi": "いすみ鉄道", "IzuHakone": "伊豆箱根鉄道",
    "Jomo": "上毛電気鉄道", "Joshin": "上信電鉄",
    "KantoRailway": "関東鉄道", "KashimaRinkai": "鹿島臨海鉄道",
    "Kominato": "小湊鐵道", "Moka": "真岡鐵道", "Ryutetsu": "流鉄",
    "Hitachinaka": "ひたちなか海浜鉄道", "Yagan": "野岩鉄道",
    "WataraseKeikoku": "わたらせ渓谷鐵道", "Yamaman": "山万",
    "MIR": "首都圏新都市鉄道", "TWR": "東京臨海高速鉄道",
    "ToyoRapid": "東葉高速鉄道", "SaitamaRailway": "埼玉高速鉄道",
    "SaitamaTransit": "埼玉新都市交通", "Minatomirai": "横浜高速鉄道",
    "YokohamaSeaside": "横浜シーサイドライン",
    "OdakyuHakone": "箱根登山鉄道", "TamaMonorail": "多摩都市モノレール",
    "ChibaMonorail": "千葉都市モノレール",
    "ShonanMonorail": "湘南モノレール", "TokyoMonorail": "東京モノレール",
    "UtsunomiyaLightRail": "宇都宮ライトレール", "Yurikamome": "ゆりかもめ",
    "Shibayama": "芝山鉄道", "Izukyu": "伊豆急行", "Fujikyu": "富士急行",
    "Aizu": "会津鉄道",
}
_LANGS = ("ja", "en", "ko", "zh-Hans", "zh-Hant")


def title(prefix: str, lang: str = "ja") -> str:
    key = PREFIX_COMPANY.get(prefix)
    got = RAIL_OPERATORS.get(key) if key else None
    if got is not None:
        table = dict(zip(_LANGS, got))
        return table.get(lang) or table["ja"]
    entry = OPERATORS.get(prefix)
    if entry is None:
        return prefix
    return entry[0] if lang == "ja" else entry[1]


def sort_key(prefix: str) -> tuple[int, str]:
    return (_RANK.get(prefix, len(PRIORITY)), prefix)


# ---------------------------------------------------------------------------
# OSM operator 태그 -> 언어별 회사 이름
#
# 노선 디버그 패널이 운영사로 묶어 보여주는데, OSM 태그는 일본어 한 가지뿐이고
# 같은 회사를 여러 표기로 적어 둔다(相模鉄道 / 相鉄 / 相模鉄道株式会社,
# 東京地下鉄 / 東京メトロ / Tokyo Metro). 표기를 하나로 모으고 언어별 이름을
# 붙인다. 표에 없는 회사는 적힌 그대로 보여준다.
# ---------------------------------------------------------------------------

RAIL_OPERATORS: dict[str, tuple[str, str, str, str, str]] = {
    # 키(일본어 정식), (ja, en, ko, zh-Hans, zh-Hant)
    "東日本旅客鉄道": ("JR東日本", "JR East", "JR동일본", "JR东日本", "JR東日本"),
    "西日本旅客鉄道": ("JR西日本", "JR West", "JR서일본", "JR西日本", "JR西日本"),
    "東海旅客鉄道": ("JR東海", "JR Central", "JR도카이", "JR东海", "JR東海"),
    "東京地下鉄": ("東京メトロ", "Tokyo Metro", "도쿄메트로", "东京地铁", "東京地鐵"),
    "東京都交通局": ("都営地下鉄", "Toei Subway", "도영지하철", "都营地铁", "都營地鐵"),
    "横浜市交通局": ("横浜市営地下鉄", "Yokohama Subway", "요코하마 시영지하철",
                 "横滨市营地铁", "橫濱市營地鐵"),
    "大阪市高速電気軌道": ("Osaka Metro", "Osaka Metro", "오사카메트로",
                   "大阪地铁", "大阪地鐵"),
    "神戸市交通局": ("神戸市営地下鉄", "Kobe Subway", "고베 시영지하철",
                "神户市营地铁", "神戶市營地鐵"),
    "近畿日本鉄道": ("近鉄", "Kintetsu", "긴테쓰", "近铁", "近鐵"),
    "東武鉄道": ("東武鉄道", "Tobu", "도부", "东武铁道", "東武鐵道"),
    "西武鉄道": ("西武鉄道", "Seibu", "세이부", "西武铁道", "西武鐵道"),
    "京成電鉄": ("京成電鉄", "Keisei", "게이세이", "京成电铁", "京成電鐵"),
    "京王電鉄": ("京王電鉄", "Keio", "게이오", "京王电铁", "京王電鐵"),
    "小田急電鉄": ("小田急電鉄", "Odakyu", "오다큐", "小田急电铁", "小田急電鐵"),
    "東急電鉄": ("東急電鉄", "Tokyu", "도큐", "东急电铁", "東急電鐵"),
    "京浜急行電鉄": ("京急", "Keikyu", "게이큐", "京急", "京急"),
    "相模鉄道": ("相鉄", "Sotetsu", "소테쓰", "相铁", "相鐵"),
    "阪急電鉄": ("阪急電鉄", "Hankyu", "한큐", "阪急电铁", "阪急電鐵"),
    "阪神電気鉄道": ("阪神電鉄", "Hanshin", "한신", "阪神电铁", "阪神電鐵"),
    "京阪電気鉄道": ("京阪電鉄", "Keihan", "게이한", "京阪电铁", "京阪電鐵"),
    "南海電気鉄道": ("南海電鉄", "Nankai", "난카이", "南海电铁", "南海電鐵"),
    "山陽電気鉄道": ("山陽電鉄", "Sanyo", "산요전철", "山阳电铁", "山陽電鐵"),
    "神戸電鉄": ("神戸電鉄", "Shintetsu", "고베전철", "神户电铁", "神戶電鐵"),
    "近江鉄道": ("近江鉄道", "Ohmi", "오미철도", "近江铁道", "近江鐵道"),
    "秩父鉄道": ("秩父鉄道", "Chichibu", "지치부철도", "秩父铁道", "秩父鐵道"),
    "上信電鉄": ("上信電鉄", "Joshin", "조신전철", "上信电铁", "上信電鐵"),
    "箱根登山鉄道": ("箱根登山鉄道", "Hakone Tozan", "하코네 등산철도",
               "箱根登山铁道", "箱根登山鐵道"),
    "伊豆箱根鉄道": ("伊豆箱根鉄道", "Izuhakone", "이즈하코네철도",
               "伊豆箱根铁道", "伊豆箱根鐵道"),
    "江ノ島電鉄": ("江ノ電", "Enoden", "에노덴", "江之岛电铁", "江之島電鐵"),
    "湘南モノレール": ("湘南モノレール", "Shonan Monorail", "쇼난 모노레일",
                "湘南单轨", "湘南單軌"),
    "東京モノレール": ("東京モノレール", "Tokyo Monorail", "도쿄 모노레일",
                "东京单轨", "東京單軌"),
    "多摩都市モノレール": ("多摩モノレール", "Tama Monorail", "다마 모노레일",
                  "多摩单轨", "多摩單軌"),
    "千葉都市モノレール": ("千葉モノレール", "Chiba Monorail", "지바 모노레일",
                  "千叶单轨", "千葉單軌"),
    "大阪モノレール": ("大阪モノレール", "Osaka Monorail", "오사카 모노레일",
                "大阪单轨", "大阪單軌"),
    "首都圏新都市鉄道": ("つくばエクスプレス", "Tsukuba Express", "쓰쿠바 익스프레스",
                 "筑波快线", "筑波快線"),
    "東京臨海高速鉄道": ("りんかい線", "Rinkai Line", "린카이선",
                 "临海线", "臨海線"),
    "東葉高速鉄道": ("東葉高速鉄道", "Toyo Rapid", "도요 고속철도",
               "东叶高速铁道", "東葉高速鐵道"),
    "埼玉高速鉄道": ("埼玉高速鉄道", "Saitama Rapid", "사이타마 고속철도",
               "埼玉高速铁道", "埼玉高速鐵道"),
    "埼玉新都市交通": ("ニューシャトル", "New Shuttle", "뉴셔틀",
                "新穿梭线", "新穿梭線"),
    "横浜高速鉄道": ("みなとみらい線", "Minatomirai Line", "미나토미라이선",
               "港未来线", "港未來線"),
    "横浜シーサイドライン": ("シーサイドライン", "Seaside Line", "시사이드 라인",
                   "海岸线", "海岸線"),
    "神戸新交通": ("神戸新交通", "Kobe New Transit", "고베 신교통",
              "神户新交通", "神戶新交通"),
    "ゆりかもめ": ("ゆりかもめ", "Yurikamome", "유리카모메",
              "百合海鸥线", "百合海鷗線"),
    "銚子電気鉄道": ("銚子電鉄", "Choshi", "조시전철", "铫子电铁", "銚子電鐵"),
    "いすみ鉄道": ("いすみ鉄道", "Isumi", "이스미철도", "夷隅铁道", "夷隅鐵道"),
    "わたらせ渓谷鐵道": ("わたらせ渓谷鐵道", "Watarase Keikoku",
                 "와타라세 계곡철도", "渡良濑溪谷铁道", "渡良瀨溪谷鐵道"),
    "ひたちなか海浜鉄道": ("ひたちなか海浜鉄道", "Hitachinaka Seaside",
                  "히타치나카 해변철도", "常陆那珂海滨铁道", "常陸那珂海濱鐵道"),
    "野岩鉄道": ("野岩鉄道", "Yagan", "야간철도", "野岩铁道", "野岩鐵道"),
    "三岐鉄道": ("三岐鉄道", "Sangi", "산기철도", "三岐铁道", "三岐鐵道"),
    "京都丹後鉄道": ("京都丹後鉄道", "Kyoto Tango", "교토 단고철도",
               "京都丹后铁道", "京都丹後鐵道"),
    "関東鉄道": ("関東鉄道", "Kanto Railway", "간토철도", "关东铁道", "關東鐵道"),
    "北総鉄道": ("北総鉄道", "Hokuso", "호쿠소철도", "北总铁道", "北總鐵道"),
    "山万": ("山万", "Yamaman", "야마만", "山万", "山萬"),
    "真岡鐵道": ("真岡鐵道", "Moka", "모카철도", "真冈铁道", "真岡鐵道"),
    "上毛電気鉄道": ("上毛電鉄", "Jomo", "조모전철", "上毛电铁", "上毛電鐵"),
    "小湊鐵道": ("小湊鐵道", "Kominato", "고미나토철도", "小凑铁道", "小湊鐵道"),
    "流鉄": ("流鉄", "Ryutetsu", "류테쓰", "流铁", "流鐵"),
    "鹿島臨海鉄道": ("鹿島臨海鉄道", "Kashima Rinkai", "가시마 임해철도",
               "鹿岛临海铁道", "鹿島臨海鐵道"),
    "宇都宮ライトレール": ("宇都宮ライトレール", "Utsunomiya Light Rail",
                  "우쓰노미야 라이트레일", "宇都宫轻轨", "宇都宮輕軌"),
    "能勢電鉄": ("能勢電鉄", "Nose", "노세전철", "能势电铁", "能勢電鐵"),
    "智頭急行": ("智頭急行", "Chizu Express", "지즈급행", "智头急行", "智頭急行"),
    "北大阪急行電鉄": ("北大阪急行", "Kita-Osaka Kyuko", "기타오사카급행",
                "北大阪急行", "北大阪急行"),
    "泉北高速鉄道": ("泉北高速鉄道", "Semboku", "센보쿠 고속철도",
               "泉北高速铁道", "泉北高速鐵道"),
    "芝山鉄道": ("芝山鉄道", "Shibayama", "시바야마철도", "芝山铁道", "芝山鐵道"),
    "伊豆急行": ("伊豆急行", "Izukyu", "이즈급행", "伊豆急行", "伊豆急行"),
    "富士急行": ("富士急行", "Fujikyu", "후지급행", "富士急行", "富士急行"),
    "会津鉄道": ("会津鉄道", "Aizu", "아이즈철도", "会津铁道", "會津鐵道"),
    "阪堺電気軌道": ("阪堺電気軌道", "Hankai", "한카이 전기궤도",
               "阪堺电气轨道", "阪堺電氣軌道"),
    "神戸高速鉄道": ("神戸高速鉄道", "Kobe Rapid Transit", "고베 고속철도",
              "神户高速铁道", "神戶高速鐵道"),
    "京都市交通局": ("京都市営地下鉄", "Kyoto Subway", "교토 시영지하철",
              "京都市营地铁", "京都市營地鐵"),
    "伊勢鉄道": ("伊勢鉄道", "Ise", "이세철도", "伊势铁道", "伊勢鐵道"),
    "伊賀鉄道": ("伊賀鉄道", "Igatetsu", "이가철도", "伊贺铁道", "伊賀鐵道"),
    "信楽高原鐵道": ("信楽高原鐵道", "Shigaraki Kohgen", "시가라키고원철도",
              "信乐高原铁道", "信樂高原鐵道"),
    "北条鉄道": ("北条鉄道", "Hojo", "호조철도", "北条铁道", "北條鐵道"),
    "和歌山電鐵": ("和歌山電鐵", "Wakayama Electric", "와카야마전철",
             "和歌山电铁", "和歌山電鐵"),
    "嵯峨野観光鉄道": ("嵯峨野観光鉄道", "Sagano Scenic", "사가노 관광철도",
               "嵯峨野观光铁道", "嵯峨野觀光鐵道"),
    "水間鉄道": ("水間鉄道", "Mizuma", "미즈마철도", "水间铁道", "水間鐵道"),
    "紀州鉄道": ("紀州鉄道", "Kishu", "기슈철도", "纪州铁道", "紀州鐵道"),
    "養老鉄道": ("養老鉄道", "Yoro", "요로철도", "养老铁道", "養老鐵道"),
    "四日市あすなろう鉄道": ("四日市あすなろう鉄道", "Yokkaichi Asunarou",
                   "욧카이치 아스나로철도", "四日市明日狭轨铁道",
                   "四日市明日狹軌鐵道"),
    "京福電気鉄道": ("京福電鉄", "Randen", "게이후쿠전철", "京福电铁", "京福電鐵"),
    "叡山電鉄": ("叡山電鉄", "Eizan", "에이잔전철", "叡山电铁", "叡山電鐵"),
    "都電荒川線": ("都電荒川線", "Toden Arakawa", "도덴 아라카와선",
              "都电荒川线", "都電荒川線"),
    # 도카이
    "名古屋鉄道": ("名鉄", "Meitetsu", "메이테쓰", "名铁", "名鐵"),
    "名古屋市交通局": ("名古屋市営地下鉄", "Nagoya Subway", "나고야 시영지하철",
                "名古屋市营地铁", "名古屋市營地鐵"),
    "名古屋臨海高速鉄道": ("あおなみ線", "Aonami Line", "아오나미선",
                  "青波线", "青波線"),
    "愛知環状鉄道": ("愛知環状鉄道", "Aichi Loop Railway", "아이치 환상철도",
               "爱知环状铁道", "愛知環狀鐵道"),
    "愛知高速交通": ("リニモ", "Linimo", "리니모", "东部丘陵线", "東部丘陵線"),
    "東海交通事業": ("東海交通事業", "Tokai Transport Service", "도카이 교통사업",
               "东海交通事业", "東海交通事業"),
    "豊橋鉄道": ("豊橋鉄道", "Toyotetsu", "도요하시철도", "丰桥铁道", "豐橋鐵道"),
    "遠州鉄道": ("遠州鉄道", "Enshu Railway", "엔슈철도", "远州铁道", "遠州鐵道"),
    "静岡鉄道": ("静岡鉄道", "Shizuoka Railway", "시즈오카철도",
             "静冈铁道", "靜岡鐵道"),
    "天竜浜名湖鉄道": ("天竜浜名湖鉄道", "Tenryu Hamanako Railroad",
                "덴류하마나코철도", "天龙滨名湖铁道", "天龍濱名湖鐵道"),
    "大井川鐵道": ("大井川鐵道", "Oigawa Railway", "오이가와철도",
              "大井川铁道", "大井川鐵道"),
    "岳南電車": ("岳南電車", "Gakunan Railway", "가쿠난전차", "岳南电车", "岳南電車"),
    "長良川鉄道": ("長良川鉄道", "Nagaragawa Railway", "나가라가와철도",
              "长良川铁道", "長良川鐵道"),
    "明知鉄道": ("明知鉄道", "Akechi Railway", "아케치철도", "明知铁道", "明知鐵道"),
    "樽見鉄道": ("樽見鉄道", "Tarumi Railway", "다루미철도", "樽见铁道", "樽見鐵道"),
}

# 같은 회사를 달리 적은 것들. 왼쪽을 오른쪽 정식 이름으로 본다.
OPERATOR_ALIAS: dict[str, str] = {
    "Tokyo Metro": "東京地下鉄", "東京メトロ": "東京地下鉄",
    "相鉄": "相模鉄道", "関鉄": "関東鉄道", "北総": "北総鉄道",
    "アーバンネットワーク": "西日本旅客鉄道",
    "池袋線系統": "西武鉄道", "多摩湖線系統": "西武鉄道",
    "川越線": "東日本旅客鉄道",
    "能勢電": "能勢電鉄", "四日市": "四日市あすなろう鉄道", "阪堺電車": "阪堺電気軌道",
    "Kintetsu Corporation": "近畿日本鉄道",
    "Kōbe Rapid Transit Railway": "神戸高速鉄道",
    "Seibu Railway Company, Ltd.": "西武鉄道",
    "国分寺線系統": "西武鉄道",
    "京都市営地下鉄": "京都市交通局",
    "東京都交通局;東京地下鉄": "東京都交通局",
    "Tokyo Tama Intercity Monorail Co., Ltd.": "多摩都市モノレール",
    "Saitama New Urban Transit Co., Ltd.": "埼玉新都市交通",
    "小湊鉄道": "小湊鐵道", "真岡鉄道": "真岡鐵道",
    "総武流山電鉄": "流鉄", "流鉄流山線": "流鉄",
    "東武": "東武鉄道", "西武": "西武鉄道", "京成": "京成電鉄",
    "京王": "京王電鉄", "小田急": "小田急電鉄", "東急": "東急電鉄",
    "京急": "京浜急行電鉄", "阪急": "阪急電鉄", "阪神": "阪神電気鉄道",
    "京阪": "京阪電気鉄道", "南海": "南海電気鉄道", "近鉄": "近畿日本鉄道",
    "山陽電鉄": "山陽電気鉄道", "神鉄": "神戸電鉄",
    "名鉄": "名古屋鉄道", "Meitetsu": "名古屋鉄道", "名鉄線路線": "名古屋鉄道",
    "名古屋市": "名古屋市交通局", "名古屋市営地下鉄": "名古屋市交通局",
}


def _canon_operator(name: str) -> str:
    """표기 흔들림을 하나로. 株式会社·(JR East) 같은 꼬리를 떼고 별칭을 편다.

    OSM 은 operator 자리에 회사가 아니라 노선을 적어 두기도 한다.
    日比谷線 은 operator 가 "東京メトロ日比谷線;東武スカイツリーライン" 이라
    아는 회사가 하나도 없는 것으로 읽혀 "그 밖" 으로 떨어졌다. 아는 회사
    이름이 안에 들어 있으면 그걸 회사로 본다. 가장 긴 것을 고른다.
    """
    s = (name or "").split(";")[0].strip()     # 여러 회사면 첫 회사
    s = s.split("(")[0].split("（")[0].strip()
    for tail in ("株式会社", "㈱"):
        s = s.replace(tail, "")
    s = s.strip()
    if not s:
        return ""
    got = OPERATOR_ALIAS.get(s)
    if got:
        return got
    if s in RAIL_OPERATORS:
        return s
    best = ""
    for key in list(RAIL_OPERATORS) + list(OPERATOR_ALIAS):
        if key and key in s and len(key) > len(best):
            best = key
    if best:
        return OPERATOR_ALIAS.get(best, best)
    return s


# 이름 안에서 회사를 찾을 때 볼 별칭. 나머지 별칭은 회사가 아니라 노선
# 이름이라("川越線", "池袋線系統") 노선 이름 안에서 찾으면 엉뚱하게 걸린다.
_NAME_ALIAS = ("東京メトロ", "相鉄", "関鉄", "北総",
               "小湊鉄道", "真岡鉄道", "総武流山電鉄",
               # 이름에는 줄여 적는다. 東武東上線 에 "東武鉄道" 는 없다.
               "東武", "西武", "京成", "京王", "小田急", "東急", "京急",
               "阪急", "阪神", "京阪", "南海", "近鉄", "山陽電鉄", "神鉄",
               "名鉄")


def operator_in_name(name: str) -> str:
    """노선 이름 안에 든 회사. 없으면 빈 문자열.

    operator 태그가 없는데 이름에는 회사가 들어 있는 노선이 많다
    (真岡鐵道真岡線, 上毛電気鉄道上毛線, 西武鉄道狭山線). _canon_operator 는
    못 찾으면 적힌 문자열을 그대로 돌려주므로 여기서는 쓸 수 없다.
    """
    s = (name or "").split("(")[0].split("（")[0].strip()
    best = ""
    for key in list(RAIL_OPERATORS) + list(_NAME_ALIAS):
        if key in s and len(key) > len(best):
            best = key
    return OPERATOR_ALIAS.get(best, best) if best else ""


_HEADS = tuple(sorted(
    set(list(RAIL_OPERATORS) + list(OPERATOR_ALIAS) + list(_NAME_ALIAS)
        + [v[0] for v in RAIL_OPERATORS.values()]),
    key=len, reverse=True))


def strip_operator_head(name: str) -> str:
    """이름 앞에 붙은 회사 이름을 뗀다. 없으면 그대로 돌려준다.

    "相鉄直通線" 에서 "直通線" 을, "小田急通勤急行" 에서 "通勤急行" 을
    꺼내 그것이 노선 이름인지 종별인지 가리는 데 쓴다.
    """
    s = (name or "").strip()
    for _ in range(2):
        for k in _HEADS:
            if k and s.startswith(k) and len(s) > len(k):
                s = s[len(k):].lstrip(" 　のノ・")
                break
        else:
            break
    return s


def rail_operator(name: str) -> dict[str, str]:
    """OSM operator 태그를 언어별 이름으로. 모르는 회사는 적힌 그대로."""
    key = _canon_operator(name)
    if not key:
        return {}
    got = RAIL_OPERATORS.get(key)
    if got is None:
        return {lang: key for lang in ("ja", "en", "ko", "zh-Hans", "zh-Hant")}
    return dict(zip(("ja", "en", "ko", "zh-Hans", "zh-Hant"), got))
