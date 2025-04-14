import re
import spacy
import torch
import os
from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity


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

    def clean_text(self, text: str) -> str:
        """입력 텍스트에서 특수 문자와 여분의 공백을 제거합니다."""
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return re.sub(r"\s+", " ", text).strip()

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """입력 텍스트의 요약을 생성합니다."""
        # 텍스트가 너무 짧은 경우 원본 반환
        if not text or len(text.split()) < 20:
            return text

        try:
            # 입력 텍스트가 너무 긴 경우 처리
            if len(text.split()) > 1024:  # 모델의 최대 입력 길이 제한
                text = " ".join(text.split()[:1024])
                print("경고: 입력 텍스트가 1024 단어로 잘렸습니다")

            # 텍스트 토큰화 및 길이 조정
            tokenized_input = self.tokenizer.encode(text, add_special_tokens=True)
            if len(tokenized_input) <= self.max_input_tokens:
                # 토큰 길이에 맞게 요약 길이 조정
                adjusted_max_length = min(max_length, max(len(tokenized_input) // 2, min_length))
                summary = self.summarizer(text, max_length=adjusted_max_length, min_length=min_length, do_sample=False)
                return summary[0]['summary_text']

            # 긴 텍스트를 청크로 나누어 처리
            chunks = self.chunk_text(text)
            summaries = [
                self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
                for chunk in chunks
            ]
            
            # 중복 제거 후 요약문 결합
            return ' '.join(dict.fromkeys(summaries))
            
        except Exception as e:
            print(f"요약 실패: {str(e)}")
            print(f"입력 텍스트 길이: {len(text.split())} 단어")
            print(f"첫 100자: {text[:100]}...")
            return text[:max_length] + "..."

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
        cause_effects = self.extract_cause_effect(cleaned, target_person)

        return {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "cleaned_text": cleaned,
            "summary": summary,
            "cause_effects": cause_effects,
            "person": target_person
        }

    def extract_cause_effect(self, text: str, target_person: str) -> List[Dict[str, str]]:
        """여러 방법을 사용하여 인과 관계를 추출합니다."""
        if not text:
            return []

        sentences = sent_tokenize(text)
        raw_results = []

        # 방법 1: 패턴 기반 추출
        raw_results.extend(self._extract_by_pattern(sentences))

        # 방법 2: spaCy 기반 추출
        raw_results.extend(self._extract_by_spacy(sentences))

        # 방법 3: BERT 기반 분류
        raw_results.extend(self._extract_by_bert(sentences))

        # 후처리
        final_results = self._post_process_results(raw_results, target_person)
        return final_results

    def _extract_by_pattern(self, sentences: List[str]) -> List[Dict[str, str]]:
        """정규식 패턴을 사용하여 인과 관계 쌍을 추출합니다."""
        results = []
        for sent in sentences:
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

    def _extract_by_spacy(self, sentences: List[str]) -> List[Dict[str, str]]:
        """spaCy 의존성 구문 분석을 사용하여 인과 관계 쌍을 추출합니다."""
        results = []
        for sent in sentences:
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

    def _extract_by_bert(self, sentences: List[str]) -> List[Dict[str, str]]:
        """T5 모델을 사용하여 상세한 정보와 함께 인과 관계 쌍을 추출합니다."""
        results = []
        for idx, sent in enumerate(sentences):
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

    def _post_process_results(self, raw_results: List[Dict], target_person: str) -> List[Dict]:
        """결과를 중복 제거하고 필터링합니다."""
        # 중복 제거
        unique_results = self._deduplicate_results(raw_results)

        # 타겟 인물 필터링
        filtered = []
        for r in unique_results:
            if (target_person.lower() in r["cause"].lower() or
                    target_person.lower() in r["effect"].lower()):
                filtered.append(r)
                continue

            # 인물 엔티티 확인
            doc = self.nlp(r["cause"] + " " + r["effect"])
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if any(target_person.lower() in p.lower() for p in persons):
                filtered.append(r)

        return filtered

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
                sim = cosine_similarity(
                    current_embedding.reshape(1, -1),
                    existing_embedding.reshape(1, -1)
                )[0][0]

                if sim > 0.85:  # 유사성 임계값
                    duplicate = True
                    break

            if not duplicate:
                unique_results.append(r)
                seen.add(key)

        return unique_results


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