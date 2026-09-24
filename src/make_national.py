"""전국을 권역 하나(national)로 빌드한다(실험).

    python src/build_national_walk.py      # 전국 보행망이 먼저 있어야 한다
    python src/make_national.py

철도 쪽은 현 조합 빌드(make_region)와 같은 단계를 돈다. 보행망만 다르다.
전국을 한 번에 build_walk 로 만들면 메모리가 모자라서, build_national_walk 가
타일로 만들어 이어 붙인 것을 하드링크로 가져오고 육지 마스크와 역별 도보권만
여기서 만든다. 北海道 는 추출본이 없고, 沖縄 는 전국 격자 밖이라 뺀다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import make_region as mr

ROOT = Path(__file__).resolve().parent.parent
RID = "national"
# data/regions/ 에 두면 원래 서버가 올리고, 조합 빌드의 이름·색 단계가 기존 권역으로
# 읽는다. 따로 둔다. 서버는 REGIONS_DIR 로 이곳을 가리킨다.
BASE = ROOT / "data" / "national" / "regions" / RID
NATIONAL_WALK = ROOT / "data" / "national" / "walk"
WALK_FILES = ["graph.npz", "fine_pt.npy", "fine_seg.npy", "fine_ptr.npy", "fine_grid.npy"]
SKIP = ("北海道", "沖縄県")

STEPS = [
    # 보행 그래프는 가져온 것을 쓰고 육지 마스크만 만든다
    ("build_walk.py", {"env": {"WALK_REUSE": "1", "WALK_WORKERS": "4"}}),
    ("build_admin.py", {}),
    ("build_rail.py", {}),
    ("build_names.py", {"optional": True}),
    ("build_express.py", {"optional": True}),
    ("build_track.py", {}),
    ("build_naive.py", {}),
    ("build_admin.py", {"env": {"ADMIN_REUSE": "1"}}),
    ("build_walk.py", {"env": {"WALK_REUSE": "1", "WALK_WORKERS": "4"}}),
    # build_colors 는 빼다. 모든 권역이 함께 쓰는 data/line-colors.json 을 고쳐 쓴다
    # (전국 권역의 "中央線" 이 大阪メトロ 색으로 들어가 다른 권역에 번졌다).
]


def setup() -> None:
    from build_national_walk import NATIONAL

    prefs = [p for p in mr.PREFS if p not in SKIP]
    meta = mr.region_json(RID, prefs)
    meta.pop("custom", None)
    meta["national"] = True      # build_names 가 기존 권역의 노선 이름을 빌려 온다
    meta["names"] = {"ja": "全国", "en": "Japan", "ko": "전국",
                     "zh-Hans": "全国", "zh-Hant": "全國"}
    meta["grid"] = dict(NATIONAL, ocean_seeds="auto")
    have = sorted(p.name for p in mr.OSM.glob("*-latest.osm.pbf"))
    meta["osm_extracts"] = [n.replace("-latest.osm.pbf", "") for n in have]
    meta["osm_files"] = have
    meta["note"] = ("전국을 권역 하나로 빌드한 것(make_national.py, 실험). 北海道·沖縄 는 "
                    "뺐다. 보행망은 build_national_walk 가 타일로 만든 것이다.")
    (BASE / "walk").mkdir(parents=True, exist_ok=True)
    (BASE / "raw").mkdir(parents=True, exist_ok=True)
    (BASE / "region.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8")
    (BASE / "stops.json").write_text(json.dumps(
        {"ids": ["STUB"], "coords": [meta["center"]]}), encoding="utf-8")

    for name in WALK_FILES:
        dst = BASE / "walk" / name
        if dst.exists():
            dst.unlink()
        os.link(NATIONAL_WALK / name, dst)
    for name in ("sheds.npz", "land.npz", "coast.npz"):
        (BASE / "walk" / name).unlink(missing_ok=True)

    # 급행 계통의 위키 표는 기존 권역들이 받아 둔 것을 모은다(조합 빌드는 없이 돈다)
    wiki = {}
    for f in sorted(mr.REGIONS.glob("*/raw/wiki-stops.json")):
        if f.parent.parent.name.startswith("custom_") or f.parent.parent.name == RID:
            continue
        wiki.update(json.loads(f.read_text(encoding="utf-8")))
    (BASE / "raw" / "wiki-stops.json").write_text(json.dumps(wiki, ensure_ascii=False),
                                                   encoding="utf-8")
    print(f"현 {len(prefs)}개, 추출본 {len(have)}개, 위키 표 {len(wiki)}개", flush=True)


def build() -> None:
    log = BASE / "build.log"
    # 단계 스크립트는 data/regions/<REGION> 을 쓰는데, 절대 경로를 주면 그곳을 쓴다
    env = dict(os.environ, REGION=str(BASE), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        for k, (script, opt) in enumerate(STEPS, 1):
            t = time.time()
            print(f"[{k}/{len(STEPS)}] {script}", end="", flush=True)
            f.write(f"\n===== [{k}/{len(STEPS)}] {script} =====\n")
            f.flush()
            r = subprocess.run([sys.executable, str(ROOT / "src" / script)],
                               env=dict(env, **opt.get("env", {})), cwd=ROOT,
                               stdout=f, stderr=subprocess.STDOUT)
            print(f"  {time.time() - t:.0f}s" + ("" if r.returncode == 0 else f"  (종료 코드 {r.returncode})"),
                  flush=True)
            if r.returncode != 0 and not opt.get("optional"):
                raise SystemExit(f"{script} 가 실패했다. {log} 를 보세요.")
            if script == "build_admin.py" and not opt.get("env"):
                rings = json.loads((BASE / "raw" / "prefecture-rings.json").read_text(encoding="utf-8"))
                prefs = json.loads((BASE / "region.json").read_text(encoding="utf-8"))["prefectures"]
                missing = [p for p in prefs if not rings.get(p)]
                if missing:
                    raise SystemExit("현 경계를 닫지 못했다: " + ", ".join(missing))
    print(f"끝 ({(time.time() - t0) / 60:.1f}분)", flush=True)


if __name__ == "__main__":
    setup()
    build()
