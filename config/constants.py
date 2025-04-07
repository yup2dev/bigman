import os

# 기본 HTTP 헤더
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

# 제외 키워드 (기사 URL 또는 제목 기준 필터링용)
EXCLUDED_KEYWORDS = {"video", "sports", "opinion", "audio", "sport"}

# 주요 키워드 (특정 인물 중심)
TARGET_KEYWORDS = {"Trump", "Donald Trump", "트럼프"}

# 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 수집 설정
DEFAULT_MEMO_LIMIT = 50
RETRY_COUNT = 3
RETRY_DELAY = 2  # 초 단위

# 저장 포맷
JSON_ENCODING = "utf-8"
DEFAULT_FILE_SUFFIX = ".json"

# 로깅 설정
DEFAULT_LOG_LEVEL = "INFO"

# ✅ 새로 추가할 경로 상수
SITES_CONFIG_PATH = os.path.join(BASE_DIR, "config", "sites.yaml")
