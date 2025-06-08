import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd
from empath import Empath
from nltk.tokenize import word_tokenize
from wordfreq import word_frequency


# 🔹 감정적 각성도 점수 (NRC VAD Lexicon 기반)
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


# 🔹 희귀 단어 점수 (wordfreq 기반)
class WordfreqRarity:
    def __init__(self, lang='en'):
        self.lang = lang

    def score(self, text: str) -> float:
        words = word_tokenize(text.lower())
        scores = [1 - word_frequency(w, self.lang) for w in words if word_frequency(w, self.lang) > 0]
        return round(np.mean(scores), 4) if scores else 0.0


# 🔹 구조적 강조 점수
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
        emph_count += sum(1 for w in words if w.isupper() and len(w) > 1)
        emph_count += len(self.repeat_pattern.findall(text))
        emph_count += len(self.exclam_pattern.findall(text))
        emph_count += sum(1 for w in words if w.lower() in self.strong_words)

        return round(emph_count / total, 4)


# 🔹 주제/감정 범주 분석 (Empath 사용)
class EmpathAnalyzer:
    def __init__(self, threshold: float = 0.0):
        self.lexicon = Empath()
        self.categories = self.lexicon.cats
        self.threshold = threshold

    def analyze(self, text: str) -> Dict[str, float]:
        scores = self.lexicon.analyze(text, normalize=True, categories=self.categories)
        return {
            category: round(score, 4)
            for category, score in scores.items()
            if score >= self.threshold
        }


# 🔹 전체 발화 스코어링 통합 클래스
class UtteranceScorer:
    def __init__(self, vad_path="NRC-VAD-Lexicon-v2.1.txt", empath_threshold: float = 0.05):
        self.emotion_model = NRCEmotionArousal(vad_path)
        self.rarity_model = WordfreqRarity()
        self.emphasis_model = EmphasisDetector()
        self.analyzer = EmpathAnalyzer(threshold=empath_threshold)

    def score(self, text: str) -> Dict[str, float]:
        return {
            "emotion_arousal": self.emotion_model.score(text),
            "keyword_rarity": self.rarity_model.score(text),
            "structural_emphasis": self.emphasis_model.score(text),
        }

    def full_score(self, text: str) -> Dict:
        return {
            "emotion_arousal": self.emotion_model.score(text),
            "keyword_rarity": self.rarity_model.score(text),
            "structural_emphasis": self.emphasis_model.score(text),
            "empath": self.analyzer.analyze(text)
        }
