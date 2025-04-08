import re
from typing import List, Dict, Any, Optional
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer
import torch

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class NLPProcessor:
    def __init__(self, summarizer_model: str = "sshleifer/distilbart-cnn-12-6"):
        device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline("summarization", model=summarizer_model, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
        self.cause_effect_keywords = [
            "because", "due to", "as a result", "therefore", "hence", "consequently"
        ]
        self.max_input_tokens = 1024

    def clean_text(self, text: str, remove_patterns: Optional[List[str]] = None) -> str:
        """불필요한 패턴 제거 및 텍스트 정리"""
        text = re.sub(r'\s+', ' ', text)
        default_patterns = [r'\[[^\]]*\]', r'\([^\)]*\)', r'(뉴스1|연합뉴스|Reuters|AP)']
        patterns = remove_patterns if remove_patterns else default_patterns
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()

    def chunk_text(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        """토큰 수 기준으로 긴 텍스트를 분할"""
        max_tokens = max_tokens or (self.max_input_tokens - 50)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks, current_chunk, current_length = [], [], 0

        for token in tokens:
            if current_length + 1 > max_tokens:
                chunks.append(self.tokenizer.decode(current_chunk, skip_special_tokens=True))
                current_chunk = [token]
                current_length = 1
            else:
                current_chunk.append(token)
                current_length += 1

        if current_chunk:
            chunks.append(self.tokenizer.decode(current_chunk, skip_special_tokens=True))
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """요약 함수"""
        if not text or len(text.split()) < 20:
            return text

        tokenized_input = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokenized_input) <= self.max_input_tokens:
            adjusted_max_length = min(max_length, max(len(tokenized_input) // 2, min_length))
            summary = self.summarizer(text, max_length=adjusted_max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']

        chunks = self.chunk_text(text)
        summaries = [
            self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            for chunk in chunks
        ]
        return ' '.join(dict.fromkeys(summaries))  # 중복 제거

    def extract_cause_effect(self, text: str) -> List[Dict[str, str]]:
        """원인-결과 관계 추출"""
        try:
            sentences = list(set(sent_tokenize(text)))
        except LookupError:
            nltk.download('punkt')
            sentences = list(set(sent_tokenize(text)))

        pairs = []
        for sentence in sentences:
            for keyword in self.cause_effect_keywords:
                pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
                match = pattern.search(sentence)
                if match:
                    idx = match.start()
                    cause, effect = sentence[:idx].strip(), sentence[idx:].strip()
                    if cause and effect:
                        pairs.append({
                            "cause": cause,
                            "effect": effect,
                            "keyword": match.group(),
                            "full_sentence": sentence
                        })
                    break
        return pairs

    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """기사 딕셔너리를 받아 요약 + 원인-결과 추출 결과 포함된 결과 반환"""
        content = article.get("content", "") or article.get("text", "")
        cleaned = self.clean_text(content)
        summary = self.summarize(cleaned)
        cause_effects = self.extract_cause_effect(cleaned)
        return {
            **article,
            "cleaned": cleaned,
            "summary": summary,
            "cause_effects": cause_effects
        }

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 URL 제거 후 기사 리스트 전체 처리"""
        seen_urls = set()
        results = []

        for article in articles:
            url = article.get("url")
            if not url or url in seen_urls:
                continue

            processed = self.process_article({
                "url": url,
                "title": article.get("title", ""),
                "content": article.get("text", ""),
                "published": article.get("published"),
                "source": article.get("source"),
            })
            results.append(processed)
            seen_urls.add(url)

        return results
