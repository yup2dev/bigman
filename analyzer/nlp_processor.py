import re
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer
import torch

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


class NLPProcessor:
    def __init__(self, summarizer_model: str = "sshleifer/distilbart-cnn-12-6"):
        device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline("summarization", model=summarizer_model, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
        self.cause_effect_keywords = [
            "because", "due to", "as a result", "therefore", "hence", "consequently"
        ]
        self.max_input_tokens = 1024

    def chunk_text(self, text: str, max_tokens: int = None) -> List[str]:
        """텍스트를 토큰 수 기준으로 분할 - 특수 토큰 고려"""
        if max_tokens is None:
            max_tokens = self.max_input_tokens - 50  # 특수 토큰 여유분 확보
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        current_chunk = []
        current_length = 0

        for token in tokens:
            if current_length + 1 > max_tokens:
                chunk_text = self.tokenizer.decode(current_chunk, skip_special_tokens=True)
                # 디버깅: 분할된 청크의 토큰 수 확인
                chunk_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
                if len(chunk_tokens) > self.max_input_tokens:
                    raise ValueError(f"Chunk exceeds max tokens after decoding: {len(chunk_tokens)}")
                chunks.append(chunk_text)
                current_chunk = [token]
                current_length = 1
            else:
                current_chunk.append(token)
                current_length += 1

        if current_chunk:
            chunk_text = self.tokenizer.decode(current_chunk, skip_special_tokens=True)
            chunk_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
            if len(chunk_tokens) > self.max_input_tokens:
                raise ValueError(f"Final chunk exceeds max tokens: {len(chunk_tokens)}")
            chunks.append(chunk_text)
        return [chunk for chunk in chunks if chunk.strip()]

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """텍스트 요약 - 디버깅 강화"""
        if not text or len(text.split()) < 20:
            return text

        try:
            # 토큰화된 입력을 체크하고, 토큰 수가 너무 많을 경우 강제로 자르기
            tokenized_input = self.tokenizer.encode(text, truncation=False, add_special_tokens=True)
            print(f"Initial token length: {len(tokenized_input)}")  # 디버깅용

            if len(tokenized_input) > self.max_input_tokens:
                # 입력이 max_input_tokens보다 길면 강제로 자르기
                print(f"Input too long, truncating to {self.max_input_tokens} tokens.")
                text = self.tokenizer.decode(tokenized_input[:self.max_input_tokens], skip_special_tokens=True)
                tokenized_input = self.tokenizer.encode(text, add_special_tokens=True)
                print(f"Truncated input token length: {len(tokenized_input)}")

            # 텍스트가 max_input_tokens 이하일 경우 분할 없이 처리
            if len(tokenized_input) <= self.max_input_tokens:
                input_length = len(tokenized_input)
                adjusted_max_length = min(max_length, max(input_length // 2, min_length))
                summary = self.summarizer(text, max_length=adjusted_max_length, min_length=min_length, do_sample=False)
                return summary[0]['summary_text']

            else:
                chunks = self.chunk_text(text)
                summaries = []
                for i, chunk in enumerate(chunks):
                    chunk_tokens = self.tokenizer.encode(chunk, truncation=False, add_special_tokens=True)
                    print(f"Chunk {i + 1} token length: {len(chunk_tokens)}")  # 디버깅용

                    # 여전히 너무 긴 청크가 있다면, 청크를 추가로 나누기
                    while len(chunk_tokens) > self.max_input_tokens:
                        # 초과하는 경우 잘라서 다시 청크로 분할
                        print(f"Chunk {i + 1} is too long, splitting further.")
                        chunk = chunk[:len(chunk) // 2]  # 절반으로 나누기
                        chunk_tokens = self.tokenizer.encode(chunk, truncation=False, add_special_tokens=True)

                    adjusted_max_length = min(max_length, max(len(chunk_tokens) // 2, min_length))
                    s = self.summarizer(chunk, max_length=adjusted_max_length, min_length=min_length, do_sample=False)
                    summaries.append(s[0]['summary_text'])

                unique_summaries = list(dict.fromkeys(summaries))
                return ' '.join(unique_summaries)

        except Exception as e:
            raise ValueError(f"Summarization failed: {e}")

    def extract_cause_effect(self, text: str) -> List[Dict[str, str]]:
        """원인과 결과 관계 추출"""
        try:
            sentences = list(set(sent_tokenize(text)))
        except LookupError:
            nltk.download('punkt')
            sentences = list(set(sent_tokenize(text)))

        pairs = []
        for sentence in sentences:
            for keyword in self.cause_effect_keywords:
                pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
                matches = [(m.start(), m.group()) for m in pattern.finditer(sentence)]
                if matches:
                    idx, matched_keyword = matches[-1]
                    cause = sentence[:idx].strip()
                    effect = sentence[idx:].strip()
                    if cause and effect:
                        pairs.append({
                            "cause": cause,
                            "effect": effect,
                            "keyword": matched_keyword,
                            "full_sentence": sentence
                        })
                    break
        return pairs

    def clean_text(self, text: str, remove_patterns: List[str] = None) -> str:
        text = re.sub(r'\s+', ' ', text)
        default_patterns = [r'\[[^\]]*\]', r'\([^\)]*\)', r'(뉴스1|연합뉴스|Reuters|AP)']
        patterns = remove_patterns if remove_patterns is not None else default_patterns
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()


nlp_instance = NLPProcessor()


def extract_cause_effect_sentences(text: str) -> List[Dict[str, str]]:
    return nlp_instance.extract_cause_effect(text)


def summarize_text(text: str) -> str:
    return nlp_instance.summarize(text)


def build_dataset(text: str) -> List[Dict[str, str]]:
    cause_effect_pairs = extract_cause_effect_sentences(text)
    dataset = [{"cause": pair["cause"],
                "effect": pair["effect"],
                "full_sentence": pair["full_sentence"]}
               for pair in cause_effect_pairs]
    return dataset


if __name__ == "__main__":
    sample_text = ("The project failed because the team lacked experience. " * 50 +
                   "As a result, the company lost a lot of money. " * 50 +
                   "I didn’t succeed due to poor planning, not because of laziness.")
    try:
        dataset = build_dataset(sample_text)
        print(f"성공적으로 {len(dataset)}개의 원인-결과 페어를 추출했습니다:")
        for i, item in enumerate(dataset[:3]):
            print(f"{i+1}. Cause: {item['cause']}")
            print(f"   Effect: {item['effect']}")
            print(f"   Full Sentence: {item['full_sentence']}\n")

        summary = summarize_text(sample_text)
        print(f"요약: {summary}")
    except Exception as e:
        print(f"오류 발생: {e}")