from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
from analyzer.nlp_processor import NLPProcessor


class ImportanceRanker:
    def __init__(self, top_k=3):
        self.top_k = top_k

    def rank_sentences(self, sentences: List[str]) -> List[str]:
        if len(sentences) <= self.top_k:
            return sentences

        tfidf_matrix = self._build_tfidf_matrix(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        scores = self._score_sentences(similarity_matrix)

        top_indices = np.argsort(scores)[::-1][:self.top_k]
        top_sentences = [sentences[i] for i in sorted(top_indices)]
        return top_sentences

    def _build_tfidf_matrix(self, sentences: List[str]):
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(sentences)

    def _score_sentences(self, sim_matrix) -> np.ndarray:
        return sim_matrix.sum(axis=1)

    def rank_articles_by_time(self, articles: List[Dict], time_key: str = "published") -> List[Dict]:
        return sorted(articles, key=lambda x: x.get(time_key, ""))


def filter_important_articles(articles: List[Dict], ranker=None, min_len: int = 200, person: str = "Trump") -> List[
    Dict]:
    if ranker is None:
        ranker = ImportanceRanker()

    nlp = NLPProcessor()
    filtered = []

    for article in articles:
        text = article.get("text", "")
        if len(text) < min_len:
            continue

        cleaned_text = nlp.clean_text(text)
        cause_effect_pairs = nlp.extract_cause_effect(cleaned_text)

        # 타겟 인물 이름이 들어간 문장만 필터링
        person_sentences = [
            f"{cause} {effect}".strip()
            for cause, effect in cause_effect_pairs
            if person.lower() in cause.lower() or person.lower() in effect.lower()
        ]

        if not person_sentences:
            continue

        top_sentences = ranker.rank_sentences(person_sentences)

        article["highlight"] = top_sentences
        article["cause_effect_pairs"] = cause_effect_pairs
        filtered.append(article)

    return filtered
