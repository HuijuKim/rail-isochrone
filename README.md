# 철도 등시선 (Rail Isochrone)

지도에서 한 곳을 찍으면, 그 시각에 집을 나서 **전철과 도보만으로** 몇 분 안에 어디까지 갈 수
있는지 그려 준다. 브라우저에서 돌아가고, 계산은 내 컴퓨터에서 한다.

![화면](docs/screenshot.jpg)

신주쿠에서 오전 9시에 나서면 60분 안에 닿는 곳(진한 쪽이 가깝다). 간토에서는 역 1,934개 중
823개에 닿는다.

## 써 보기

1. **출발지를 정한다.** 검색창에 역 이름을 치거나 지도를 왼쪽 클릭한다. 한국어·일본어·영어
   아무 말로나 치면 되고, 한글은 초성(ㅅㅈㅋ)이나 치는 중인 글자(신ㅈ)로도 찾는다.
2. **시각과 상한을 정한다.** 출발 시각, 평일/휴일, 소요 시간 상한을 고른다. "지금" 을 누르면
   현재 시각으로 맞춘다.
3. **띠를 읽는다.** 상한을 3등분해 세 겹으로 그린다. 진한 안쪽이 가깝고 옅은 바깥쪽이 멀다.
   전철에서 내려 걷는 시간(하차 후 도보)도 함께 센다.
4. **도착지를 주면 경로가 나온다.** 지도를 오른쪽 클릭하거나 도착지 칸에 역을 넣으면, 몇 시
   차를 타서 어디서 갈아타고 내려서 몇 분 걷는지 알려 준다. 걷는 편이 빠르면 그렇다고 말한다.

그 밖에

- **전철 없이 도보만**: 걷기만으로 얼마나 가는지.
- **지원 역·노선 표시**(지도 왼쪽 아래): 다룰 수 있는 역과 노선을 겹쳐 본다. 옆의 삼각형으로
  현별로 골라 본다.
- **권역 탭**: 간토·간사이처럼 미리 만들어 둔 지역을 고른다.
- **현 조합 탭**: 원하는 현을 골라 새 권역을 만든다. 이어진 현만 켜지고, "이 조합 만들기" 를
  누르면 컴퓨터가 뒤에서 만든다(10-40분). 만드는 동안 다른 권역은 그대로 쓸 수 있고, 다 되면
  같은 탭에 남는다. 권역 경계를 넘는 노선(瀬戸大橋線)도 이 방식이면 끊기지 않는다.
- 언어는 한국어·일본어·영어·중국어(간체/번체)를 고를 수 있다.

## 띄우기

빌드해 둔 데이터를 받는 쪽이 빠르다. 파이썬 3.11 이상이 필요하다.

```
python -m pip install numpy scipy matplotlib flask
python src/fetch_release.py --list        # 어떤 권역이 있는지
python src/fetch_release.py kyushu        # 골라 받기 (권역당 64-245 MB)
python src/server.py                      # http://127.0.0.1:5173
```

지도는 OpenStreetMap 으로 키 없이 뜬다. 구글 지도를 쓰려면 Maps JavaScript API 키를
환경변수 `GOOGLE_MAPS_API_KEY` 에 두거나 `config.json` 에 적는다(저장소에 올리지 않는다).

```json
{ "google_maps_api_key": "AIza..." }
```

## 권역

| 권역 | 범위 | 시각표 |
|---|---|---|
| `kanto` | 수도권 | 실제(ODPT, mini-tokyo-3d) |
| `kanto_osm` | 간토 1도 6현 | 추정 |
| `kansai` | 간사이 2부 4현 | 추정 |
| `tokai` | 아이치·기후·시즈오카·미에 | 추정 |
| `hokuriku` | 도야마·이시카와·후쿠이 | 추정 |
| `koshinetsu` | 야마나시·나가노·니가타 | 추정 |
| `chugoku` | 주고쿠 5현 | 추정 |
| `shikoku` | 시코쿠 4현 | 추정 |
| `kyushu` | 규슈 7현 | 추정 |
| `tohoku_s` | 미야기·야마가타·후쿠시마 | 추정 |
| `tohoku_n` | 아오모리·이와테·아키타 | 추정 |

홋카이도와 오키나와는 철도망이 성겨 뺐다. 권역은 현 단위로 묶여 있고, 경계를 넘는 조합은
화면의 현 조합 탭이나 `python src/make_region.py 岡山 香川 --build` 로 만든다.

## 직접 빌드하기

[Geofabrik](https://download.geofabrik.de/asia/japan.html) 추출본을 `data/osm/` 에 두고
(권역이 읽는 목록은 `data/regions/<권역>/region.json` 의 `osm_files`), `osmium` 을 더 설치한
뒤 차례대로 돌린다.

```
REGION=<권역> python src/build_walk.py       # 보행망·해안선·육지 마스크
REGION=<권역> python src/build_admin.py      # 행정경계 -> 권역 경계
REGION=<권역> python src/build_rail.py       # OSM 관계 -> 노선과 정차 순서
REGION=<권역> python src/build_express.py    # 위키백과에서 통과 계통 정차역 (없으면 건너뛴다)
REGION=<권역> python src/build_track.py      # 역 사이 선로 선형과 등급
REGION=<권역> python src/build_naive.py      # 배차·주행 시간을 추정해 시각표를 짓는다
REGION=<권역> ADMIN_REUSE=1 python src/build_admin.py   # 역에 현 붙이기
REGION=<권역> WALK_REUSE=1 python src/build_walk.py     # 역별 도보권
python src/build_colors.py                   # 노선 색
```

추출본을 훑은 결과는 `data/cache/` 에 추출본 이름으로 남아, 같은 추출본을 읽는 다른 권역이
그대로 쓴다. 간토 실제 시각표 권역만 순서가 다르다(`fetch_data.py` → `build_graph.py` →
`build_walk.py` → `build_admin.py`). 검사는 `python -m pytest tests/ -q` 다.

## 어떻게 계산하나

- **도달 시간**: RAPTOR 를 벡터화해 돌린다. 한 번 계산에 역 전부의 도착 시각이 나온다.
  상한 60분 기준 등시선 0.6초, 한 지점까지의 경로 0.3초다.
- **시각표**: 간토는 실제 시각표다. 나머지 권역은 지어낸다. 구간별 운행 횟수 데이터로 배차를,
  OSM 선로 태그(전철화·궤도 종류)로 속도를 정한다. 완행은 실제 대비 ±12%, 쾌속·특급은
  5-24% 느리다.
- **도보**: OSM 보행로를 40m 격자에 붙여 그래프로 만들고, 역마다 40분 도보권을 미리 계산해
  둔다. 분속 80m, 계단은 2.2배로 친다.
- **권역 경계**: 현 경계를 OSM 해안선으로 자른 것이다. 역에서 보행망으로 이어진 섬(다리·방파제)
  만 넣고, 배로만 가는 섬은 뺀다.

## 알려진 한계

- 신칸센이 없다. 권역 사이 이동은 현 조합으로 한 권역을 만들어야 계산된다.
- 버스가 없다. 역에서 먼 곳은 실제보다 좁게 나온다.
- 혼잡·지연·신호 대기를 세지 않는다. 역 구내 환승은 3분, 개찰 밖은 5분으로 갈음한다.
- 시각표가 없는 권역은 쾌속·특급이 부족하다. 정차역을 위키백과 표에서 읽는데 노선마다 표
  모양이 달라 많이 못 읽었다.
- 운휴 구간은 뺀다(肥薩線 八代-吉松 등).

## 폴더

```
src/    빌드 스크립트(OSM -> 노선·시각표·보행망)와 서버, 계산 코드
web/    화면 한 장 (index.html)
data/   손으로 적은 사전(노선 이름·색·손질 규칙)과 권역 설정.
        빌드 결과와 OSM 추출본, 캐시도 여기에 쌓이지만 저장소에는 올리지 않는다
tests/  회귀 검사. 권역마다 경계·노선·역 수를 기준선과 견준다
docs/   README 에 쓰는 그림
```

[HANDOVER.md](HANDOVER.md) 에 빌드 순서와 손질 사전, 코드가 왜 이 모양인지 적어 두었다.
[WORKLOG.md](WORKLOG.md) 는 무엇을 언제 고쳤는지의 기록이다.

## 데이터와 라이선스

코드는 [PolyForm Noncommercial 1.0.0](LICENSE) 이다. 영리 목적이 아니면 쓰고 고치고 다시
배포할 수 있다. OSI 의 오픈소스 정의를 만족하는 라이선스는 아니다.

저장소에는 코드와 손으로 적은 사전만 있다. 빌드한 권역 데이터는 GitHub Releases 에 따로
올리고, 코드와 다른 조건을 따른다.

| 항목 | 출처 | 조건 |
|---|---|---|
| 노선·역·선로·보행망·해안선·행정경계 | [OpenStreetMap](https://www.openstreetmap.org/copyright) 기여자 | ODbL 1.0 |
| 구간별 운행 횟수 | [全国鉄道運行本数データ](https://gtfs-gis.jp/railway_honsu/) (西澤明) | CC BY 4.0 / ODbL |
| 통과 계통 정차역 일부 | [ja.wikipedia.org](https://ja.wikipedia.org/) 의 駅一覧 | CC BY-SA 4.0 |
| 간토 실제 시각표 | [mini-tokyo-3d](https://github.com/nagix/mini-tokyo-3d) / [ODPT](https://www.odpt.org/) | 릴리스에 넣지 않는다 |

릴리스 데이터는 OSM 파생 데이터베이스라 **ODbL 1.0** 으로 배포하고 "© OpenStreetMap
contributors" 를 밝힌다. 가공해 다시 배포하면 같은 조건을 따라야 한다. 간토 실제 시각표
권역을 릴리스에 넣지 않는 것은 ODPT 데이터셋의 재배포 조건을 확인하지 못해서다. 그 권역은
`python src/fetch_data.py` 로 각자 받는다.
