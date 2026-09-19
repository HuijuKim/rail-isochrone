import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


REGION = ROOT / "data" / "regions" / "kanto"


@pytest.fixture(scope="session")
def stops():
    return json.loads((REGION / "stops.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def graph():
    from router import load_graph

    return load_graph(REGION / "graph-Weekday.npz")


@pytest.fixture(scope="session")
def walk():
    import walknet

    net = walknet.load(REGION / "walk")
    if net is None:
        pytest.skip("보행망이 없습니다 (build_walk.py 미실행)")
    return net


@pytest.fixture(scope="session")
def station_index(stops):
    """역 ID -> 배열 인덱스"""
    return {sid: i for i, sid in enumerate(stops["ids"])}


@pytest.fixture(scope="session")
def shinjuku(stops, station_index):
    lon, lat = stops["coords"][station_index["JR-East.Yamanote.Shinjuku"]]
    return float(lon), float(lat)


def minutes_to(best, depart, index):
    from router import INF

    if best[index] >= INF:
        return None
    return (int(best[index]) - depart) / 60.0
