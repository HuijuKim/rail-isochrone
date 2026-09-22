"""全国鉄道運行本数データ(구간별 평일 운행 횟수)를 나이브 시각표의 배차로 쓴다.

출처: 西沢明(地域・交通データ研究所), https://gtfs-gis.jp/railway_honsu/
라이선스 CC BY 4.0 / ODbL. 특급처럼 요금을 더 받는 열차는 세지 않았다.

파일은 data/honsu/unkohonsu2026_kukan.txt (탭 구분, WKT). 한 줄이 한
회사의 분기역 사이 선로 구간이고, 방향별 하루 운행 횟수가 적혀 있다.
그 선로를 지나는 열차를 노선 구분 없이 합친 것이다. 路線별 파일
(rosen_kukan)은 呉線 열차를 山陽本線 선로 위에서도 呉線 으로 따로 세어,
広島-海田市 가 선로 총량(142회)의 절반으로 읽혔다.

우리 노선의 이웃한 두 역이 같은 회사 구간 선형 위에 둘 다 놓이면 그
구간의 횟수를 쓴다.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "honsu" / "unkohonsu2026_kukan.txt"

NEAR_M = 80.0        # 역이 구간 선형에서 이만큼 안이면 그 위에 있다고 본다
STEP_M = 40.0        # 선형 꼭짓점 사이를 이 간격으로 메운다
SERVICE_HOURS = 18.0  # 하루 횟수를 시간당으로 옮길 때 나누는 값(05-23시)

_WKT = re.compile(r"-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?")


def _canon(name: str) -> str:
    from operators import _canon_operator

    return _canon_operator(name or "")


def _honsu_operator(name: str) -> str:
    """이 데이터의 사업자 이름을 우리 회사 표의 열쇠로.

    도시 교통국은 도시 이름만 적혀 있다(東京都, 横浜市, 福岡市).
    """
    from operators import RAIL_OPERATORS

    got = _canon(name)
    if got in RAIL_OPERATORS:
        return got
    if (name + "交通局") in RAIL_OPERATORS:
        return name + "交通局"
    return got


class Honsu:
    def __init__(self, scale: float, bbox: tuple[float, float, float, float], path: Path = PATH):
        from scipy.spatial import cKDTree

        self.scale = scale
        csv.field_size_limit(1 << 30)
        lo_x, lo_y, hi_x, hi_y = bbox
        ops, per_day, names, xy, owner = [], [], [], [], []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                pts = np.array([[float(v) for v in p.split()]
                                for p in _WKT.findall(row["geometry"])])
                if not len(pts):
                    continue
                if (pts[:, 0].max() < lo_x or pts[:, 0].min() > hi_x
                        or pts[:, 1].max() < lo_y or pts[:, 1].min() > hi_y):
                    continue
                k = len(ops)
                ops.append(_honsu_operator(row["事業者名"]))
                fwd = float(row["順方向運行本数2024"] or 0)
                bwd = float(row["逆方向運行本数2024"] or 0)
                per_day.append((fwd + bwd) / 2.0)
                names.append(f'{row["事業者名"]} {row["路線名"]} '
                             f'{row["起点駅"]}-{row["終点駅"]}')
                m = self._xy(pts)
                for p, q in zip(m, m[1:]):
                    n = max(int(np.hypot(*(q - p)) // STEP_M), 1)
                    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
                    xy.append(p + (q - p) * t)
                    owner.extend([k] * n)
                xy.append(m[-1:])
                owner.append(k)
        self.ops, self.per_day, self.names = ops, per_day, names
        self.owner = np.asarray(owner, dtype=np.int64)
        self.tree = cKDTree(np.concatenate(xy)) if xy else None

    def _xy(self, lonlat):
        a = np.asarray(lonlat, dtype=np.float64).reshape(-1, 2)
        return np.c_[a[:, 0] * self.scale * 111_320.0, a[:, 1] * 111_132.0]

    def near(self, lon: float, lat: float) -> dict[int, float]:
        """이 점에서 NEAR_M 안의 구간과 거리."""
        if self.tree is None:
            return {}
        p = self._xy([lon, lat])[0]
        out: dict[int, float] = {}
        for i in self.tree.query_ball_point(p, NEAR_M):
            k = int(self.owner[i])
            d = float(np.hypot(*(self.tree.data[i] - p)))
            if d < out.get(k, np.inf):
                out[k] = d
        return out

    def section(self, a, b, operator: str | None = None) -> int | None:
        """두 역을 모두 지나는 구간. 회사를 알면 그 회사 것만 본다."""
        na, nb = self.near(*a), self.near(*b)
        both = [k for k in na if k in nb]
        if operator:
            both = [k for k in both if self.ops[k] == operator]
        if not both:
            return None
        return min(both, key=lambda k: na[k] + nb[k])

    def per_hour(self, k: int) -> float:
        return self.per_day[k] / SERVICE_HOURS
