import nli as nli

from crawler.url_collector import collect_urls
from crawler.utils import load_articles_from_urls
from analyzer.nlp_processor import NLPProcessor
import spacy
from transformers import pipeline
from typing import List, Tuple

# Load NLP pipeline
nlp = spacy.load("en_core_web_sm")
nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 명시적 인과 키워드 정의
CAUSE_KEYWORDS = [
    "because", "due to", "as a result", "therefore", "thus", "hence",
    "resulting in", "leads to", "caused by", "causes", "so that"
]

def build_dataset(site_key="cnn", keywords=["trump"], limit=10):
    print("🔍 Step 1. 기사 URL 수집")
    urls = collect_urls(site_key=site_key, keywords=keywords, limit=limit)

    print("📰 Step 2. 기사 본문 수집")
    articles = load_articles_from_urls(urls)

    print("🧠 Step 3. NLP 처리 시작")
    nlp = NLPProcessor()
    for article in articles:
        def extract_cause_effect2(text: str, max_pairs: int = 5) -> List[Tuple[str, str, str]]:
            """
            뉴스 기사 등 일반 텍스트에서 원인-결과 관계를 추출하는 범용 함수.
            Returns: List of tuples (label: 'explicit' or 'implicit', cause, effect)
            """

            def split_sentences(text: str) -> List[str]:
                doc = nlp(text)
                return [sent.text.strip() for sent in doc.sents]

            def is_explicit_causal(sent: str) -> bool:
                return any(kw in sent.lower() for kw in CAUSE_KEYWORDS)

            def is_implicit_causal(sent1: str, sent2: str, threshold: float = 0.85) -> bool:
                hypothesis = f"The second sentence is caused by the first."
                result = nli(sequence=sent1 + " " + sent2, hypothesis=hypothesis,
                             candidate_labels=["entailment", "neutral", "contradiction"])
                return result['labels'][0] == "entailment" and result['scores'][0] >= threshold

            sentences = split_sentences(text)
            cause_effect_pairs = []

            # 명시적 인과 추출
            for sent in sentences:
                if is_explicit_causal(sent):
                    for kw in CAUSE_KEYWORDS:
                        if kw in sent.lower():
                            parts = sent.split(kw)
                            if len(parts) == 2:
                                cause = parts[0].strip(",. ")
                                effect = parts[1].strip(",. ")
                                cause_effect_pairs.append(("explicit", cause, effect))
                            break

            # 암시적 인과 추출
            for i in range(len(sentences) - 1):
                s1, s2 = sentences[i], sentences[i + 1]
                if is_implicit_causal(s1, s2):
                    cause_effect_pairs.append(("implicit", s1.strip(), s2.strip()))

            # 중복 제거 및 제한
            seen = set()
            unique_pairs = []
            for label, cause, effect in cause_effect_pairs:
                key = (cause, effect)
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append((label, cause, effect))

            return unique_pairs[:max_pairs]

        for article in articles:
            cleaned_text = nlp.clean_text(article["text"])
            print(f"clean_text = {cleaned_text}")
            print("-------------")
            print(extract_cause_effect2(cleaned_text))

if __name__ == "__main__":
    build_dataset()