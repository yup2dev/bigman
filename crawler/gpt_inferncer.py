import openai
from typing import Dict, List
from utils.util import ensure_openai_api_key

ensure_openai_api_key()


class GPTUtteranceInterpreter:
    def __init__(self, model="gpt-4"):
        self.model = model

    def _call_gpt(self, system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"].strip()

    def _safe_parse(self, content: str, expected_keys: List[str]) -> Dict:
        try:
            parsed = eval(content, {"__builtins__": {}})
            assert isinstance(parsed, dict)
            for key in expected_keys:
                if key not in parsed:
                    raise ValueError(f"Missing key: {key}")
            return parsed
        except Exception as e:
            print("⚠️ Parsing error:", e)
            print("GPT raw output:\n", content)
            return {key: [] if 'policies' in key or 'effects' in key else None for key in expected_keys}

    def analyze_sentiment(self, utterance: str) -> Dict:
        prompt = f"""
        Analyze the sentiment of the following political utterance.
        Respond in JSON format:
        {{
          "label": "POSITIVE or NEGATIVE or NEUTRAL",
          "emotion_arousal": float between -1.0 and 1.0
        }}
        Utterance:
        \"{utterance}\"
        """
        content = self._call_gpt(
            "You are a political analyst who evaluates tone and emotional intensity.",
            prompt,
            max_tokens=200
        )
        return self._safe_parse(content, ["label", "emotion_arousal"])

    def extract_conditions_and_conclusion(self, text: str) -> (List[str], str):
        prompt = f"""
        Given the following statement, extract any conditional phrases or constraints (e.g., 'if...', 'unless...')
        and the main conclusion or claim being made.

        Respond in JSON format:
        {{
          "conditions": ["..."],
          "conclusion": "..."
        }}

        Statement:
        \"{text}\"
        """
        content = self._call_gpt(
            "You are an expert in logic and reasoning.",
            prompt,
            max_tokens=300
        )
        parsed = self._safe_parse(content, ["conditions", "conclusion"])
        return parsed["conditions"], parsed["conclusion"]

    def extract_policies_and_effects(self, utterance: str) -> Dict:
        prompt = f"""
        From the following statement, extract:
        1. Up to 2 policy ideas the speaker supports
        2. Any likely or known consequences (effects) of such policies
        3. Up to 2 expected effects the speaker supports

        Respond in JSON format:
        {{
          "predicted_policies": ["..."],
          "expected_effects": ["..."]
        }}

        Statement:
        \"{utterance}\"
        """
        content = self._call_gpt(
            "You are a policy analyst trained to infer both policy direction and consequences from public statements.",
            prompt
        )
        return self._safe_parse(content, ["predicted_policies", "expected_effects"])

    def infer_persona(self, conditions, conclusion, sentiment_label, arousal, empath) -> Dict:
        # 기본 휴리스틱 기반 퍼소나 예시
        return {
            "fiscal_caution": 0.85 if "cost" in conclusion.lower() or "billion" in conclusion.lower() else 0.4,
            "infrastructure_focus": 0.75 if any("highway" in c.lower() or "transport" in c.lower() for c in conditions) else 0.3,
            "mixed_use_preference": 0.65 if "hotel" in conclusion.lower() else 0.25
        }

    def validate_policy(self, policy: str) -> float:
        # 실제 이력을 기준으로 검증하는 곳에서 활용 가능 (현재는 모의값)
        return 0.82 if policy else 0.0

    def classify_tone(self, text: str) -> str:
        prompt = f"""Classify the emotional tone of the following statement:

        "{text}"

        Respond with one of: assertive, dismissive, conciliatory, neutral, passionate.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )
            return response["choices"][0]["message"]["content"].strip().lower()
        except Exception as e:
            print("GPT tone classification failed:", e)
            return "neutral"

