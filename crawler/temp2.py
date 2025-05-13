import re
import numpy as np
from typing import Dict
from nltk.tokenize import word_tokenize
from wordfreq import word_frequency
import pandas as pd


class EmotionLexicon:
    def __init__(self, vad_path: str = "NRC-VAD-Lexicon-v2.1.txt"):
        self.vad_dict = self._load_vad_lexicon(vad_path)

    def _load_vad_lexicon(self, path: str) -> Dict[str, Dict[str, float]]:
        df = pd.read_csv(path, sep='\t')
        return {
            row['term']: {
                'valence': row['valence'],
                'arousal': row['arousal'],
                'dominance': row['dominance']
            }
            for _, row in df.iterrows()
        }

    def get_arousal(self, text: str) -> float:
        words = word_tokenize(text.lower())
        arousal_scores = [self.vad_dict[w]['arousal'] for w in words if w in self.vad_dict]
        return np.mean(arousal_scores) if arousal_scores else 0.0


class KeywordRarityCalculator:
    def __init__(self, lang: str = 'en'):
        self.lang = lang

    def calculate_rarity(self, text: str) -> float:
        words = word_tokenize(text.lower())
        if not words:
            return 0.0
        freq_scores = [1 - word_frequency(w, self.lang) for w in words if word_frequency(w, self.lang) > 0]
        return np.mean(freq_scores) if freq_scores else 0.0


class StructuralEmphasisDetector:
    def __init__(self):
        self.exclam_pattern = re.compile(r"!+")
        self.repetition_pattern = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

    def compute_emphasis(self, text: str) -> float:
        if not text.strip():
            return 0.0

        total_words = len(word_tokenize(text))
        if total_words == 0:
            return 0.0

        emph_count = 0

        # 1. 대문자 단어 (강조)
        emph_count += sum(1 for word in text.split() if word.isupper() and len(word) > 1)

        # 2. 감탄사 사용
        emph_count += len(self.exclam_pattern.findall(text))

        # 3. 반복 단어 (like "very very good")
        emph_count += len(self.repetition_pattern.findall(text))

        return emph_count / total_words


class UtteranceScorer:
    def __init__(self, vad_path: str = "NRC-VAD-Lexicon-v2.1.txt"):
        self.emotion_lexicon = EmotionLexicon(vad_path)
        self.keyword_rarity = KeywordRarityCalculator()
        self.structural_emphasis = StructuralEmphasisDetector()

    def score(self, text: str) -> Dict[str, float]:
        return {
            "emotion_arousal": self.emotion_lexicon.get_arousal(text),
            "keyword_rarity": self.keyword_rarity.calculate_rarity(text),
            "structural_emphasis": self.structural_emphasis.compute_emphasis(text)
        }
