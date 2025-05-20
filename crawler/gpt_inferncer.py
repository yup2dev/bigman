# pipeline/gpt_belief_keyword_mapper.py

from typing import Dict
import openai
from utils.util import ensure_openai_api_key

ensure_openai_api_key()  # API 키 환경변수에서 로드


class GPTUtteranceInterpreter:
    def __init__(self, model="gpt-4"):
        self.model = model

    def extract_intent_and_keywords(self, utterance: str) -> Dict:
        prompt = self._build_prompt(utterance)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an expert political analyst who summarizes speaker intent and extracts core keywords."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        return self._parse_response(response)

    def _build_prompt(self, utterance: str) -> str:
        return f"""
        Given the following political utterance, extract:
        1. The speaker's main behavioral *intent*.
        2. A list of 3 to 7 relevant *keywords* from the utterance.
        
        Respond in this JSON format:
        {{
          "intent": "...",
          "keywords": ["...", "...", "..."]
        }}
        
        Utterance:
        \"{utterance}\"
        """

    def _parse_response(self, response) -> Dict:
        content = response["choices"][0]["message"]["content"].strip()
        try:
            # 안전하게 JSON-like 텍스트 파싱
            parsed = eval(content, {"__builtins__": {}})
            assert isinstance(parsed, dict) and "intent" in parsed and "keywords" in parsed
            return parsed
        except Exception as e:
            print("⚠️ Parsing error:", e)
            print("GPT raw output:\n", content)
            return {
                "intent": None,
                "keywords": []
            }
