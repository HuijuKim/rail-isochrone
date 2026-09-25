"""운영 서버가 불러 쓰는 입구. server.py 의 app.run 은 개발용이다.

    cd src
    waitress-serve --host=127.0.0.1 --port=8000 --threads=8 wsgi:app

설정은 환경 변수로 받는다. 따로 주지 않으면 전국 권역 하나만 올리고(온라인
모드가 되어 현 조합 빌드가 막힌다), 보행망은 메모리 매핑으로 읽는다.

    REGIONS_DIR            권역 폴더(기본 data/regions)
    GOOGLE_MAPS_API_KEY    브라우저용 지도 키(웹사이트 제한)
    GOOGLE_SERVER_API_KEY  장소 수 세기용 서버 키(없으면 그 칸이 안 보인다)
    TRUST_PROXY=1          nginx 뒤에 둘 때. X-Forwarded-For 를 믿어야 IP 별
                           한도가 걸린다(안 그러면 모든 요청이 127.0.0.1 이다)
"""
import os

os.environ.setdefault("SERVE_REGIONS", "national")
os.environ.setdefault("WALK_MMAP", "1")

from server import app  # noqa: E402

if os.environ.get("TRUST_PROXY") == "1":
    from werkzeug.middleware.proxy_fix import ProxyFix

    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
