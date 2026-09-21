"""걸어서 갈아타는 이웃 역을 찾아 환승 간선을 만든다.

시각표 권역(build_graph)과 가상 시각표 권역(build_naive)이 함께 쓴다.
둘이 따로 갖고 있으면 한쪽만 고치게 되고, 실제로 그런 일이 있었다.

왜 필요한가. 高槻市(한큐)와 高槻(JR)는 520m 떨어져 있고 사람들은 그
사이를 걸어서 갈아탄다. 구글 지도도 그 환승을 내놓는다. 그런데 우리는
이어 둔 범위가 400m 라 그 쌍이 빠져 있었고, 라우터는 한참 돌아가는
길을 골랐다. 시각표가 아니라 환승망이 없어서 생긴 일이다.

이미 이어진 쌍(ODPT 역 그룹, 이름이 같아 한 묶음이 된 역)은 부르는
쪽에서 걸러 넘긴다.

비용은 거리에 따라 매긴다. 예전처럼 400m 안이면 전부 300초로 두면,
범위를 넓히는 순간 1km 를 5분에 걷는 셈이 되어 도달권역이 부풀어
오른다. 실제로 걷는 시간을 재서 매기면 멀수록 비싸지고, 타고 가는
편이 빠른 자리에서는 라우터가 알아서 안 쓴다.
"""
from __future__ import annotations

import numpy as np

# 이보다 멀면 환승으로 치지 않는다. 구글 지도가 역 사이 도보 환승을
# 내놓는 범위와 대략 같다.
WALK_XFER_MAX_M = 1000.0
# 개찰을 나갔다 들어오는 데 드는 몫. 거리와 무관하게 붙는다.
WALK_XFER_BASE_SEC = 90.0
# 직선거리를 실제 걷는 거리로 바꾸는 배수. 길이 곧게 나 있지 않다.
WALK_XFER_DETOUR = 1.25
# 아무리 가까워도 개찰 밖 환승보다 싸지는 않다.
WALK_XFER_MIN_SEC = 300.0


def walk_cost(dist_m: float, speed_m_per_sec: float) -> int:
    """이웃 역까지 걸어서 갈아타는 데 드는 시간(초)."""
    walk = dist_m * WALK_XFER_DETOUR / speed_m_per_sec
    return int(round(max(WALK_XFER_MIN_SEC, WALK_XFER_BASE_SEC + walk)))


def near_pairs(lons, lats, speed_m_per_sec, limit_m: float = WALK_XFER_MAX_M):
    """(i, j, 비용) 목록. i < j 이고 한 쌍마다 한 번만 나온다.

    좌표는 같은 순서의 배열이면 무엇이든 된다. 누구를 이을지(역 묶음인지
    역 하나인지)는 부르는 쪽이 정한다.
    """
    from scipy.spatial import cKDTree

    P = np.asarray(np.stack([np.asarray(lons, dtype=np.float64),
                             np.asarray(lats, dtype=np.float64)], axis=1))
    if len(P) < 2:
        return []
    scale = float(np.cos(np.radians(float(np.nanmedian(P[:, 1])))))
    x = P[:, 0] * scale * 111_320.0
    y = P[:, 1] * 111_132.0
    ok = np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return []
    tree = cKDTree(np.stack([x[idx], y[idx]], axis=1))
    out = []
    for a, b in tree.query_pairs(limit_m):
        i, j = int(idx[a]), int(idx[b])
        d = float(np.hypot(x[i] - x[j], y[i] - y[j]))
        out.append((i, j, walk_cost(d, speed_m_per_sec)))
    return out
