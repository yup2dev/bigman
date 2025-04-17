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
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('nlp_processor.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path)

# OpenAI API 키 로드
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 .env에 정의되지 않았습니다.")

client = OpenAI(api_key=openai_api_key)

class NLPProcessor:
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.2, save_dir: str = "data/processed"):
        self.model = model
        self.temperature = temperature
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _build_prompt(self, article):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file_path = os.path.join(current_dir, "..", "config", "prompt_template.txt")
        
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            logger.warning(f"프롬프트 템플릿 파일을 찾을 수 없습니다: {prompt_file_path}. 기본 템플릿 사용.")
            template = self.default_prompt_template
        except IOError as e:
            logger.error(f"프롬프트 파일 읽기 오류: {e}. 기본 템플릿 사용.")
            template = self.default_prompt_template

        return template.format(
            title=article.get("title", ""),
            text=article.get("text", "")
        )

    def _call_gpt(self, prompt, max_retries=1, wait_time=10):
        attempt = 0
        while attempt < max_retries:
            try:
                logger.info(f"GPT API 호출 시도 {attempt + 1}/{max_retries}")
                logger.debug(f"프롬프트 내용: {prompt[:200]}...")
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                
                logger.debug(f"GPT 응답 전체: {response}")
                content = response.choices[0].message.content
                logger.info(f"GPT 응답 content 길이: {len(content) if content else 0}")

                if not content:
                    logger.warning("GPT 응답 content가 비어 있음.")
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
                logger.error(f"예상치 못한 오류: {e}")
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

        # 날짜별 폴더 생성
        folder_path = os.path.join(self.save_dir, date_str)
        os.makedirs(folder_path, exist_ok=True)
        full_path = os.path.join(folder_path, filename)

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f" Sava Successfully: {full_path}")
        except Exception as e:
            logger.error(f" Failed to Save: {e}")

    def _validate_response(self, result):
        try:
            cleaned_result = result.strip()
            if not (cleaned_result.startswith('[') and cleaned_result.endswith(']')):
                cleaned_result = f"[{cleaned_result}]" if cleaned_result else "[]"

            data = json.loads(cleaned_result)
            logger.info(f"Successfully Parse JSON. DataType: {type(data)}")

            if not isinstance(data, list):
                logger.warning("Response is not a list")
                return None

            required_fields = ['person', 'decision', 'outcome', 'context', 'impact_type']
            allowed_types = ['positive', 'negative', 'neutral', 'economic', 'diplomatic', 'political', 'social', 'economic', 'legal', 'social']

            for item in data:
                if not all(field in item for field in required_fields):
                    logger.warning(f"Missing fields in item: {item}")
                    return None

                impact_types = [t.strip().lower() for t in item['impact_type'].split(',')]
                for t in impact_types:
                    if t not in allowed_types:
                        logger.warning(f"Invalid impact_type value: {t}")
                        return None

            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to Parse JSON: {e}")
            logger.error(f"ORG resopnse: {result[:500]}...")
            return None

    def process_article(self, article):
        logger.info("Starting article processing...")
        prompt = self._build_prompt(article)
        logger.info("Prompt built successfully")

        time.sleep(2)

        result = self._call_gpt(prompt)
        logger.info("GPT API call completed")

        if result is None:
            logger.warning("GPT no response.")
            return None

        validated_result = self._validate_response(result)
        if validated_result:
            logger.info("Response validated successfully")

            combined_data = article.copy()
            combined_data['events'] = validated_result

            self._save_to_file(combined_data, article, validated_result)
            return combined_data
        else:
            logger.warning("Failed to Valid")
            return None
