import re
import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from wordfreq import word_frequency

nlp = spacy.load("en_core_web_sm")


def load_vad_lexicon(filepath: str = "NRC-VAD-Lexicon-v2.1.txt") -> dict:
    df = pd.read_csv(filepath, sep='\t')
    vad_dict = {
        row['term']: {
            'valence': row['valence'],
            'arousal': row['arousal'],
            'dominance': row['dominance']
        }
        for _, row in df.iterrows()
    }
    return vad_dict


VAD_DICT = load_vad_lexicon()


class UtteranceScorer:
    @staticmethod
    def compute_emotion_arousal(text: str) -> float:
        words = word_tokenize(text.lower())
        scores = [VAD_DICT[w]['arousal'] for w in words if w in VAD_DICT]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    @staticmethod
    def compute_keyword_rarity(text: str, lang='en') -> float:
        words = word_tokenize(text.lower())
        freqs = [word_frequency(w, lang) for w in words if w.isalpha()]
        rarity_scores = [-np.log10(f) if f > 0 else 0 for f in freqs]
        return round(np.mean(rarity_scores), 3) if rarity_scores else 0.0

    @staticmethod
    def compute_structural_emphasis(text: str) -> float:
        patterns = [
            r'\b[A-Z]{2,}\b',  # ALL CAPS
            r'!+',  # exclamations
            r'\.{2,}',  # ...
            r'\b(\w+)\s+\1\b'  # repeated words
        ]
        base_score = sum(len(re.findall(p, text)) for p in patterns)

        doc = nlp(text)
        discourse_markers = sum(1 for t in doc if t.text.lower() in ['but', 'however', 'really', 'very'])

        total_score = base_score + discourse_markers
        word_count = len([t for t in doc if t.is_alpha])
        return round(min(total_score / word_count, 1.0), 3) if word_count else 0.0

    @staticmethod
    def compute_memorability(arousal: float, rarity: float, emphasis: float) -> float:
        return round(0.5 * arousal + 0.3 * rarity + 0.2 * emphasis, 3)


# 사용 예시
if __name__ == "__main__":
    text = "Well no not really. I had a great faith in New York, primarily our purchases have been in New York, and at the... about five years ago in New York was not considered very hot and cities in general weren't considered too hot. And we purchased the old Commodore Hotel and we have reconverted that now into about a $110 million Grand Hyatt Hotel which is opening up next week in New York City, and we've made some other purchases that have been fine."

    arousal = UtteranceScorer.compute_emotion_arousal(text)
    rarity = UtteranceScorer.compute_keyword_rarity(text)
    emphasis = UtteranceScorer.compute_structural_emphasis(text)
    memorability = UtteranceScorer.compute_memorability(arousal, rarity, emphasis)

    print(f"emotion_arousal: {arousal}")
    print(f"keyword_rarity: {rarity}")
    print(f"structural_emphasis: {emphasis}")
    print(f"memorability_score: {memorability}")
