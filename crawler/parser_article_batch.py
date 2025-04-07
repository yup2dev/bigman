import os
import logging
from .utils import save_json
from .article_parser import parse_articles

def parse_articles_batch(urls, save_dir=None, source_name=None, log_dir="bigman/logs"):
    """
    여러 개의 URL을 받아 파싱하고, 결과를 리스트로 반환하며,
    개별 json 파일 저장 + 실패 로그 기록

    :param urls: 기사 URL 리스트
    :param save_dir: 저장할 디렉토리 경로 (None이면 저장 안함)
    :param source_name: 소스 이름이 있으면 파일 이름에 포함
    :param log_dir: 실패 로그 저장 경로
    :return: 파싱된 기사 리스트
    """
    results = []
    failed_urls = []

    # 로그 디렉토리 준비
    os.makedirs(log_dir, exist_ok=True)
    error_log_path = os.path.join(log_dir, f"{source_name or 'unknown'}_errors.log")

    for idx, url in enumerate(urls):
        print(f"[{idx+1}/{len(urls)}] Parsing: {url}")
        try:
            article = parse_article(url)

            if article:
                results.append(article)

                # 저장 옵션이 있다면 저장
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    file_prefix = source_name or article.get("source", "unknown")
                    filename = f"{file_prefix}_{idx+1:03d}.json"
                    path = os.path.join(save_dir, filename)
                    save_json(article, path)
            else:
                failed_urls.append(url)

        except Exception as e:
            failed_urls.append(url)
            print(f"❌ Error parsing {url}: {e}")

    # 실패한 URL 로그 파일로 저장
    if failed_urls:
        with open(error_log_path, "w") as f:
            f.write("\n".join(failed_urls))
        print(f"⚠️  {len(failed_urls)} failed URLs logged to {error_log_path}")

    return results
