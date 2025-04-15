import re
import spacy
import torch
import os
# import logging
from typing import List, Dict, Any, Optional
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer


class NLPProcessor:
    def __init__(self,
                 summarizer_model: str = "sshleifer/distilbart-cnn-12-6",
                 bert_model: str = "bert-base-uncased",
                 fine_tuned_bert_path: str = "../tune/cause_effect_model"):
        """NLPProcessor를 초기화합니다. 필요한 모든 모델을 로드합니다."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_summarizer(summarizer_model)
        self._initialize_spacy()
        self._initialize_bert_models(bert_model, fine_tuned_bert_path)
        self._initialize_sentence_transformer()
        self._initialize_trigger_map()
        self._initialize_cause_effect_model()

    def _initialize_summarizer(self, model_name: str):
        """텍스트 요약 파이프라인을 초기화합니다."""
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )

    def _initialize_spacy(self):
        """spaCy NLP 파이프라인을 초기화합니다."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError("spaCy 영어 모델이 필요합니다. 다음 명령어를 실행하세요: python -m spacy download en_core_web_sm")

    def _initialize_bert_models(self, base_model: str, fine_tuned_path: str):
        """T5 모델을 초기화합니다."""
        self.bert_tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # 상대 경로를 절대 경로로 변환
        if fine_tuned_path.startswith('./'):
            fine_tuned_path = os.path.abspath(fine_tuned_path)
            
        self.bert_model = T5ForConditionalGeneration.from_pretrained(
            fine_tuned_path,
            local_files_only=True
        ).to(self.device)
        self.bert_model.eval()

    def _initialize_sentence_transformer(self):
        """문장 임베딩 모델을 초기화합니다."""
        self.embedding_model = SentenceTransformer(
            "paraphrase-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def _initialize_trigger_map(self):
        """인과 관계 추출을 위한 트리거 단어를 초기화합니다."""
        self.trigger_map = {
            "because": ("effect", "cause"),
            "since": ("effect", "cause"),
            "as": ("effect", "cause"),
            "due to": ("effect", "cause"),
            "owing to": ("effect", "cause"),
            "thanks to": ("effect", "cause"),
            "resulting in": ("cause", "effect"),
            "leads to": ("cause", "effect"),
            "led to": ("cause", "effect"),
            "caused by": ("effect", "cause"),
            "triggered by": ("effect", "cause"),
            "result of": ("effect", "cause"),
            "consequence of": ("effect", "cause"),
            "despite": ("effect", "cause")
        }

    def _initialize_cause_effect_model(self):
        """미세조정된 T5 모델을 초기화합니다."""
        try:
            # 현재 파일의 디렉토리를 기준으로 모델 경로 설정
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "..", "tune", "cause_effect_model")
            
            if os.path.exists(model_path):
                print("미세조정된 T5 모델 로드 중...")
                self.cause_effect_model = T5ForConditionalGeneration.from_pretrained(model_path)
                self.cause_effect_tokenizer = T5Tokenizer.from_pretrained(model_path)
                self.cause_effect_model.to(self.device)
                self.cause_effect_model.eval()
                print("미세조정된 T5 모델 로드 완료!")
            else:
                print(f"미세조정된 T5 모델을 찾을 수 없습니다: {model_path}")
                print("기본 추출 방법을 사용합니다.")
                self.cause_effect_model = None
                self.cause_effect_tokenizer = None
        except Exception as e:
            print(f"미세조정된 T5 모델 초기화 실패: {str(e)}")
            self.cause_effect_model = None
            self.cause_effect_tokenizer = None

    def clean_text(self, text: str, remove_patterns: Optional[List[str]] = None) -> str:
        """입력 텍스트를 정리합니다.
        
        Args:
            text: 정리할 텍스트
            remove_patterns: 제거할 정규식 패턴 리스트 (기본값: 대괄호와 괄호 안의 내용)
            
        Returns:
            정리된 텍스트
        """
        text = re.sub(r'\s+', ' ', text)
        default_patterns = [r'\[[^\]]*\]', r'\([^\)]*\)']
        patterns = remove_patterns if remove_patterns else default_patterns
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """입력 텍스트의 요약을 생성합니다."""
        if not text or len(text.split()) < 20:
            return text

        tokenized_input = self.summarizer.tokenizer.encode(text, add_special_tokens=True)
        max_input_tokens = self.summarizer.tokenizer.model_max_length
        if len(tokenized_input) <= max_input_tokens:
            adjusted_max_length = min(max_length, max(len(tokenized_input) // 2, min_length))
            summary = self.summarizer(text, max_length=adjusted_max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']

        chunks = self.chunk_text(text)
        summaries = [
            self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            for chunk in chunks
        ]
        return ' '.join(dict.fromkeys(summaries))

    def process_articles(self, articles: List[Dict[str, Any]], target_person: str) -> List[Dict[str, Any]]:
        """여러 기사를 처리하여 인과 관계를 추출합니다."""
        results = []
        seen_urls = set()
        total_articles = len(articles)
        
        print(f"\n총 {total_articles}개의 기사를 처리합니다...")
        
        for idx, article in enumerate(articles, 1):
            if not isinstance(article, dict):
                print(f"경고: {idx}번째 기사가 유효하지 않은 형식입니다.")
                continue

            url = article.get("url", "")
            if url in seen_urls:
                print(f"경고: {idx}번째 기사가 이미 처리된 URL입니다.")
                continue

            print(f"\n[{idx}/{total_articles}] 기사 처리 중...")
            print(f"제목: {article.get('title', '제목 없음')}")
            
            processed = self._process_single_article(article, target_person)
            if processed:
                results.append(processed)
                seen_urls.add(url)
                print(f"✅ {idx}번째 기사 처리 완료")
            else:
                print(f"❌ {idx}번째 기사 처리 실패")
            
            # 진행률 표시
            progress = (idx / total_articles) * 100
            print(f"진행률: {progress:.1f}%")

        print(f"\n처리 완료! 총 {len(results)}개의 기사가 성공적으로 처리되었습니다.")
        return results

    def _process_single_article(self, article: Dict[str, Any], target_person: str) -> Dict[str, Any]:
        """단일 기사를 처리하고 관련 정보를 추출합니다."""
        content = article.get("content", "") or article.get("text", "")
        if not content:
            return None

        cleaned = self.clean_text(content)
        summary = self.summarize(cleaned)
        cause_effects = self.extract_cause_effect(cleaned)

        return {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "cleaned_text": cleaned,
            "summary": summary,
            "cause_effects": cause_effects,
            "person": target_person
        }

    def extract_cause_effect(self, text: str) -> List[Dict]:
        """텍스트에서 인과 관계를 추출합니다."""
        if not text:
            return []

        # 미세조정된 모델을 우선적으로 사용
        if self.cause_effect_model:
            try:
                results = self._extract_cause_effect_finetuned(text)
                if results:  # 결과가 있으면 반환
                    return results
            except Exception as e:
                print(f"미세조정된 모델 추출 실패: {str(e)}")

        # 미세조정된 모델 실패 시 패턴 매칭과 Spacy 조합 사용
        pattern_results = self._extract_cause_effect_pattern(text)
        spacy_results = self._extract_cause_effect_spacy(text)
        
        # 결과 통합 및 중복 제거
        all_results = pattern_results + spacy_results
        return self._deduplicate_results(all_results)

    def _extract_cause_effect_pattern(self, text: str) -> List[Dict]:
        """정규식 패턴을 사용하여 인과 관계 쌍을 추출합니다."""
        results = []
        for sent in sent_tokenize(text):
            for trig, (effect_label, cause_label) in self.trigger_map.items():
                pattern = fr"(.+?)\s{trig}\s(.+?)(?:\.|;|$)"
                match = re.search(pattern, sent.lower())
                if match:
                    cause = match.group(2) if cause_label == "cause" else match.group(1)
                    effect = match.group(1) if effect_label == "effect" else match.group(2)
                    if self._is_valid_pair(cause, effect):
                        results.append({
                            "cause": cause.strip(),
                            "effect": effect.strip(),
                            "source": "pattern",
                            "confidence": 0.9  # 정확한 매칭에 대한 높은 신뢰도
                        })
        return results

    def _extract_cause_effect_spacy(self, text: str) -> List[Dict]:
        """spaCy 의존성 구문 분석을 사용하여 인과 관계 쌍을 추출합니다."""
        results = []
        for sent in sent_tokenize(text):
            doc = self.nlp(sent)
            for token in doc:
                if token.text.lower() in self.trigger_map:
                    effect_dir, cause_dir = self.trigger_map[token.text.lower()]
                    subtree = list(token.subtree)
                    cause_start = min(subtree, key=lambda t: t.i).i
                    cause_end = max(subtree, key=lambda t: t.i).i + 1
                    cause_span = doc[cause_start:cause_end]

                    # 간단한 절 분할 (더 나은 NLP 로직으로 개선 필요)
                    clauses = [chunk.text for chunk in doc.noun_chunks]
                    if len(clauses) >= 2:
                        cause = clauses[1] if cause_dir == "cause" else clauses[0]
                        effect = clauses[0] if effect_dir == "effect" else clauses[1]
                        if self._is_valid_pair(cause, effect):
                            results.append({
                                "cause": cause,
                                "effect": effect,
                                "source": "spacy",
                                "confidence": 0.7
                            })
        return results

    def _extract_cause_effect_finetuned(self, text: str) -> List[Dict]:
        """T5 모델을 사용하여 상세한 정보와 함께 인과 관계 쌍을 추출합니다."""
        results = []
        for idx, sent in enumerate(sent_tokenize(text)):
            # T5를 위한 입력 텍스트 준비
            input_text = f"extract cause-effect: {sent}"
            
            # 구조화된 출력 생성
            inputs = self.bert_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # token_type_ids가 있는 경우 제거
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
                
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.bert_model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                
            # 생성된 텍스트 디코딩
            generated_text = self.bert_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 생성된 텍스트를 파싱하여 정보 추출
            try:
                # 예상 형식: "cause: [cause] | person: [person] | cause_time: [time] | effect: [effect] | effect_time: [time]"
                parts = generated_text.split(" | ")
                info = {}
                for part in parts:
                    if ":" in part:
                        key, value = part.split(":", 1)
                        info[key.strip()] = value.strip()
                
                if "cause" in info and "effect" in info:
                    results.append({
                        "id": str(idx + 1),  # 1부터 시작하는 인덱스를 ID로 사용
                        "person": info.get("person", ""),
                        "context": sent,  # 원본 문장을 컨텍스트로 사용
                        "cause": info["cause"],
                        "cause_time": info.get("cause_time", ""),
                        "effect": info["effect"],
                        "effect_time": info.get("effect_time", ""),
                        "confidence": 0.8,  # 모델 신뢰도에 따라 조정
                        "source": "t5"
                    })
            except Exception as e:
                print(f"생성된 텍스트 파싱 오류: {e}")
                continue

        return results

    def _is_valid_pair(self, cause: str, effect: str) -> bool:
        """추출된 쌍이 품질 기준을 충족하는지 검증합니다."""
        cause, effect = cause.strip(), effect.strip()
        return (len(cause.split()) >= 2 and
                len(effect.split()) >= 2 and
                cause.lower() != effect.lower() and
                not cause.startswith(('and', 'but', 'or')) and
                not effect.startswith(('and', 'but', 'or')))

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """의미적 유사성을 사용하여 중복 인과 관계 쌍을 제거합니다."""
        seen = set()
        unique_results = []

        for r in sorted(results, key=lambda x: -x.get('confidence', 0)):
            key = (r["cause"].lower(), r["effect"].lower())
            if key in seen:
                continue

            # 의미적 유사성 확인
            duplicate = False
            current_embedding = self.embedding_model.encode(
                f"{r['cause']} {r['effect']}",
                convert_to_tensor=True
            )

            for existing in unique_results:
                existing_embedding = self.embedding_model.encode(
                    f"{existing['cause']} {existing['effect']}",
                    convert_to_tensor=True
                )

                # PyTorch를 사용하여 코사인 유사도 계산
                sim = torch.nn.functional.cosine_similarity(
                    current_embedding.unsqueeze(0),
                    existing_embedding.unsqueeze(0)
                ).item()

                if sim > 0.85:  # 유사성 임계값
                    duplicate = True
                    break

            if not duplicate:
                unique_results.append(r)
                seen.add(key)

        return unique_results

    def chunk_text(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        """텍스트를 토큰 단위로 분할합니다."""
        max_tokens = max_tokens or (self.summarizer.tokenizer.model_max_length - 50)
        tokens = self.summarizer.tokenizer.encode(text, add_special_tokens=False)
        chunks, current_chunk, current_length = [], [], 0

        for token in tokens:
            if current_length + 1 > max_tokens:
                chunks.append(self.summarizer.tokenizer.decode(current_chunk, skip_special_tokens=True))
                current_chunk = [token]
                current_length = 1
            else:
                current_chunk.append(token)
                current_length += 1

        if current_chunk:
            chunks.append(self.summarizer.tokenizer.decode(current_chunk, skip_special_tokens=True))
        return [chunk.strip() for chunk in chunks if chunk.strip()]


# Example usage
if __name__ == "__main__":
    # Initialize processor (will automatically use local models)
    processor = NLPProcessor()

    # Sample article data
    articles = [{
        "title": "Company Layoffs Announcement",
        "url": "http://example.com/news1",
        "content": (
            "CEO John Smith announced major layoffs yesterday due to financial losses. "
            "This decision led to a 10% drop in company stock price. "
            "Because of the restructuring, many employees will lose their jobs."
        )
    }]

    # Process articles for specific person
    results = processor.process_articles(articles, "John Smith")

    # Print results
    import json

    print(json.dumps(results, indent=2))