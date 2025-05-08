import os
import json
import time
import logging
import re
from datetime import datetime
from dotenv import load_dotenv
import openai
from openai import OpenAI
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 로그 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('nlp_processor.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 환경 변수 로드
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 .env에 정의되지 않았습니다.")

client = OpenAI(api_key=openai_api_key)


class NLPProcessor:
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.2, save_dir: str = None):
        self.model = model
        self.temperature = temperature
        self.save_dir = save_dir or os.path.join(base_dir, "data", "analyzed")
        os.makedirs(self.save_dir, exist_ok=True)

    def _build_prompt(self, article):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file_path = os.path.join(current_dir, "..", "config", "prompt_template.txt")

        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            logger.warning(f"프롬프트 템플릿 파일을 찾을 수 없습니다: {prompt_file_path}")
            return ""
        except IOError as e:
            logger.error(f"프롬프트 파일 읽기 오류: {e}")
            return ""

        return template.format(
            title=article.get("title", ""),
            text=article.get("text", "")
        )

    def _call_gpt(self, prompt, max_retries=1, wait_time=10):
        attempt = 0
        while attempt < max_retries:
            try:
                logger.info(f"GPT API 호출 시도 {attempt + 1}/{max_retries}")
                logger.debug(f"프롬프트 내용 일부: {prompt[:300]}")

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )

                content = response.choices[0].message.content
                logger.debug(f"GPT 응답 내용 일부: {content[:500]}")
                if not content:
                    logger.warning("GPT 응답이 비어 있음.")
                    return None
                return content
            except openai.RateLimitError:
                logger.error(f"Rate limit 초과. {wait_time}초 대기 후 재시도.")
                time.sleep(wait_time * (attempt + 1))
                attempt += 1
            except openai.APIError as e:
                logger.error(f"API 오류 발생: {e}")
                return None
            except Exception as e:
                logger.error(f"예상치 못한 오류 발생: {e}")
                logger.exception("상세 에러 정보:")
                return None
        return None

    def _save_to_file(self, data: dict, article: dict, analysis: list) -> None:
        first_event = analysis[0] if analysis else {}
        person = re.sub(r'\s+', '_', first_event.get('person', 'unknown'))[:40]
        impact_types = first_event.get("impact_type", "unknown")
        impact_tokens = [re.sub(r'\W+', '', t.strip().lower()) for t in impact_types.split(',')]
        impact_part = "_".join(impact_tokens)[:40]

        pub_date = article.get("published") or datetime.today().strftime("%Y-%m-%d")
        try:
            date_str = datetime.strptime(pub_date[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except:
            date_str = datetime.today().strftime("%Y-%m-%d")

        filename = f"{person}_{date_str}_{impact_part}.json"
        folder_path = os.path.join(self.save_dir, date_str)
        os.makedirs(folder_path, exist_ok=True)
        full_path = os.path.join(folder_path, filename)

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"파일 저장 완료: {full_path}")
        except Exception as e:
            logger.error(f"파일 저장 실패: {e}")

    def _normalize_person_name(self, name: str) -> str:
        name = name.strip().lower()
        if "trump" in name:
            return "Donald Trump"
        return name.title()

    def _validate_response(self, result):
        try:
            cleaned_result = result.strip()
            if not (cleaned_result.startswith('[') and cleaned_result.endswith(']')):
                cleaned_result = f"[{cleaned_result}]" if cleaned_result else "[]"

            data = json.loads(cleaned_result)
            if not isinstance(data, list):
                logger.warning("Not List type")
                return None

            required_fields = ['person', 'decision', 'effect', 'context', 'impact_type']
            allowed_types = [
                'positive', 'negative', 'neutral', 'economic', 'diplomatic',
                'political', 'social', 'legal', 'military', 'health'
            ]

            for item in data:
                missing = [field for field in required_fields if field not in item]
                if missing:
                    logger.warning(f"필수 필드 누락: {missing} | item: {json.dumps(item, indent=2)}")
                    return None

                impact_types = list(set(t.strip().lower() for t in item.get('impact_type', '').split(',')))
                item['impact_type'] = ", ".join(impact_types)
                for t in impact_types:
                    if t not in allowed_types:
                        logger.warning(f"허용되지 않은 impact_type: {t}")
                        return None

            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            logger.error(f"응답 원본: {result[:500]}")
            return None

    def process_article(self, article):
        logger.info("기사 분석 시작")
        prompt = self._build_prompt(article)
        if not prompt:
            logger.warning("프롬프트 생성 실패")
            return None

        time.sleep(1)

        result = self._call_gpt(prompt)
        if result is None:
            logger.warning("GPT 응답 실패")
            return None

        validated_result = self._validate_response(result)
        if validated_result:
            logger.info("GPT 응답 검증 완료")

            for item in validated_result:
                item['person'] = self._normalize_person_name(item.get('person', 'unknown'))

                impact_types = list(set(t.strip().lower() for t in item.get('impact_type', '').split(',')))
                item['impact_type'] = ", ".join(impact_types)

                optional_fields = ['cause', 'cause_time', 'decision_time', 'effect', 'effect_time', 'method', 'tone']
                for field in optional_fields:
                    item.setdefault(field, None)

            combined_data = article.copy()
            combined_data['events'] = validated_result

            self._save_to_file(combined_data, article, validated_result)
            return combined_data
        else:
            logger.warning("응답 검증 실패")
            return None
