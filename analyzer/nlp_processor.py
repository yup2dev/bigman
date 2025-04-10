import re
from typing import List, Dict, Any, Optional
import nltk
import spacy
from newspaper.nlp import summarize
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, BertForSequenceClassification, BertTokenizer
import torch

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class NLPProcessor:
    def __init__(self, summarizer_model: str = "sshleifer/distilbart-cnn-12-6", bert_model: str = "bert-base-uncased"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Summarizer 초기화
        self.summarizer = pipeline("summarization", model=summarizer_model,
                                   device=0 if torch.cuda.is_available() else -1)
        self.tokenizer = AutoTokenizer.from_pretrained(summarizer_model)

        # spaCy 모델 초기화
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise OSError(
                f"Failed to load 'en_core_web_sm'. Install with: python -m spacy download en_core_web_sm. Error: {e}"
            )

        # BERT 모델 초기화 (fine-tuned for cause-effect classification)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=2).to(self.device)

        self.max_input_tokens = 1024

    def clean_text(self, text: str, remove_patterns: Optional[List[str]] = None) -> str:
        text = re.sub(r'\s+', ' ', text)
        default_patterns = [r'\[[^\]]*\]', r'\([^\)]*\)', r'(뉴스1|연합뉴스|Reuters|AP)']
        patterns = remove_patterns if remove_patterns else default_patterns
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()

    def chunk_text(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
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
        return ' '.join(dict.fromkeys(summaries))

    def extract_cause_effect(self, text: str) -> List[Dict[str, str]]:
        results = []

        # 1. spaCy 기반 추출
        sentences = sent_tokenize(text)
        trigger_words = [
            "because", "since", "as", "due to", "resulting in", "owing to", "thanks to", "leads to", "despite"
        ]
        for sent in sentences:
            doc = self.nlp(sent)
            for token in doc:
                if token.text.lower() in trigger_words:
                    # 절 단위로 분리하여 원인과 결과 추출
                    clauses = [clause.text for clause in doc.sents]
                    if len(clauses) > 1:
                        cause = clauses[1].strip()
                        effect = clauses[0].strip()
                        results.append({"cause": cause, "effect": effect, "source": "spacy"})
                    else:
                        # 단일 절에서 추출
                        cause = " ".join([t.text for t in doc[token.i + 1:]]).strip()
                        effect = " ".join([t.text for t in doc[:token.i]]).strip()
                        if cause and effect:
                            results.append({"cause": cause, "effect": effect, "source": "spacy"})

        # 2. 패턴 기반 추출
        patterns = [
            (r"(.+?) because (.+?)\.", "effect", "cause"),
            (r"(.+?) due to (.+?)\.", "effect", "cause"),
            (r"(.+?) led to (.+?)\.", "cause", "effect"),
            (r"(.+?) as a result of (.+?)\.", "effect", "cause"),
            (r"(.+?) resulted in (.+?)\.", "cause", "effect"),
            (r"(.+?) as a consequence of (.+?)\.", "effect", "cause"),
            (r"(.+?) caused by (.+?)\.", "effect", "cause"),
            (r"(.+?) owing to (.+?)\.", "effect", "cause"),
            (r"(.+?) thanks to (.+?)\.", "effect", "cause"),
            (r"(.+?) despite (.+?)\.", "effect", "cause")
        ]
        for sent in sentences:
            for pattern, effect_label, cause_label in patterns:
                match = re.search(pattern, sent)
                if match:
                    cause = match.group(2) if cause_label == "cause" else match.group(1)
                    effect = match.group(1) if effect_label == "effect" else match.group(2)
                    results.append({"cause": cause, "effect": effect, "source": "pattern"})

        # 3. BERT 기반 검증
        for sent in sentences:
            inputs = self.bert_tokenizer(sent, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                logits = self.bert_model(**inputs).logits
                prob = torch.softmax(logits, dim=1)[0][1].item()
            if prob > 0.5:  # 원인-결과 관계로 판단될 경우
                results.append({"cause": "BERT_detected_cause", "effect": sent, "confidence": prob, "source": "bert"})

        # 4. 중복 제거 및 후처리
        seen = set()
        final_results = []
        priority = {"pattern": 1, "spacy": 2, "bert": 3}
        sorted_results = sorted(results, key=lambda x: priority.get(x["source"], 4))

        for r in sorted_results:
            key = (r["cause"], r["effect"])
            if key not in seen:
                final_results.append(r)
                seen.add(key)

        return final_results

    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        content = article.get("content", "") or article.get("text", "")
        cleaned = self.clean_text(content)
        summary = self.summarize(cleaned) if summarize else ""
        cause_effects = self.extract_cause_effect(cleaned)
        return {
            **article,
            "cleaned": cleaned,
            "summary": summary,
            "cause_effects": cause_effects
        }

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


if __name__ == "__main__":
    nlp_processor = NLPProcessor()
    sample_text = "The event was canceled because it rained heavily."
    result = nlp_processor.extract_cause_effect(sample_text)
    print("Final result:", result)