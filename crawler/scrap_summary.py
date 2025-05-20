import re, os
from typing import List, Dict

import numpy as np
import pandas as pd
from empath import Empath
from nltk.tokenize import word_tokenize
from wordfreq import word_frequency


# 감정 사전 기반
class NRCEmotionArousal:
    def __init__(self, vad_path: str = "NRC-VAD-Lexicon-v2.1.txt"):
        abs_path = os.path.join(os.path.dirname(__file__), vad_path)
        self.vad = self._load_vad(abs_path)

    def _load_vad(self, path: str):
        df = pd.read_csv(path, sep='\t')
        return {row['term']: row['arousal'] for _, row in df.iterrows()}

    def score(self, text: str) -> float:
        words = word_tokenize(text.lower())
        scores = [self.vad[w] for w in words if w in self.vad]
        return round(np.mean(scores), 4) if scores else 0.0


# wordfreq 기반 희귀도 계산
class WordfreqRarity:
    def __init__(self, lang='en'):
        self.lang = lang

    def score(self, text: str) -> float:
        words = word_tokenize(text.lower())
        scores = [1 - word_frequency(w, self.lang) for w in words if word_frequency(w, self.lang) > 0]
        return round(np.mean(scores), 4) if scores else 0.0


# 강조 점수
class EmphasisDetector:
    def __init__(self):
        self.exclam_pattern = re.compile(r"[!]{1,}")
        self.repeat_pattern = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
        self.strong_words = {'really', 'very', 'absolutely', 'totally', 'so', 'too'}

    def score(self, text: str) -> float:
        if not text.strip():
            return 0.0

        words = word_tokenize(text)
        total = len(words)
        if total == 0:
            return 0.0

        emph_count = 0

        # 대문자 단어
        emph_count += sum(1 for w in words if w.isupper() and len(w) > 1)

        # 반복 단어
        emph_count += len(self.repeat_pattern.findall(text))

        # 감탄사
        emph_count += len(self.exclam_pattern.findall(text))

        # 강조 부사
        emph_count += sum(1 for w in words if w.lower() in self.strong_words)

        return round(emph_count / total, 4)


class EmpathAnalyzer:
    def __init__(self, include: List[str] = None, exclude: List[str] = None, threshold: float = 0.0):
        self.lexicon = Empath()
        all_categories = set(self.lexicon.cats)
        self.include = set(include) if include else all_categories
        self.exclude = set(exclude) if exclude else set()
        self.threshold = threshold

        self.categories = sorted(list(self.include - self.exclude))

    def analyze(self, text: str) -> Dict[str, float]:
        scores = self.lexicon.analyze(text, normalize=True, categories=self.categories)
        return {
            category: round(score, 4)
            for category, score in scores.items()
            if score >= self.threshold
        }


# 통합
class UtteranceScorer:
    def __init__(self, vad_path="NRC-VAD-Lexicon-v2.1.txt"):
        self.emotion_model = NRCEmotionArousal(vad_path)
        self.rarity_model = WordfreqRarity()
        self.emphasis_model = EmphasisDetector()
        self.analyzer = EmpathAnalyzer(threshold=0.05)

    def score(self, text: str):
        return {
            "emotion_arousal": self.emotion_model.score(text),
            "keyword_rarity": self.rarity_model.score(text),
            "structural_emphasis": self.emphasis_model.score(text)
        }
