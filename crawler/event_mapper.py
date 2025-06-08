import os.path
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from utils.constants import BASE_DIR
from utils.util import load_json


class GDELTPolicyEventMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        sentence-transformers 모델 초기화
        """
        self.model = SentenceTransformer(model_name)

    def _combine_event_text(self, event: Dict) -> str:
        """
        GDELT 이벤트 정보에서 의미 비교용 텍스트 구성
        """
        return " | ".join([
            event.get("EventDescription", ""),
            event.get("Actor1Name", ""),
            event.get("Actor2Name", ""),
            event.get("ActionGeo_FullName", ""),
            event.get("EventCode", ""),
            event.get("EventBaseCode", ""),
        ])

    def _build_query_text(self, utterance_info: Dict) -> str:
        """
        발언 intent + keywords 조합 → 의미적 검색 쿼리
        """
        intent = utterance_info.get("intent", "")
        keywords = utterance_info.get("keywords", [])
        return intent + " " + " ".join(keywords)

    def match(
        self,
        utterance_info: Dict,
        gdelt_events: List[Dict],
        top_k: int = 3,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        utterance_info에 가장 의미적으로 유사한 GDELT 사건 N개 매핑
        """
        # 1. 쿼리 임베딩
        query_text = self._build_query_text(utterance_info)
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)

        # 2. GDELT 이벤트 텍스트 구성 및 임베딩
        event_texts = [self._combine_event_text(e) for e in gdelt_events]
        event_embeddings = self.model.encode(event_texts, convert_to_tensor=True)

        # 3. 디바이스 일치 처리
        device = query_embedding.device
        event_embeddings = event_embeddings.to(device)

        # 4. cosine 유사도 계산
        similarities = util.cos_sim(query_embedding, event_embeddings)[0].cpu().numpy()

        # 5. 유사도 필터 및 정렬
        matched = [
            {"event": gdelt_events[i], "similarity": float(similarities[i])}
            for i in range(len(similarities))
            if similarities[i] >= min_similarity
        ]

        matched.sort(key=lambda x: x["similarity"], reverse=True)
        return matched[:top_k]

def run_event_mapping_test():
    # 파일 경로
    article_path = os.path.join(BASE_DIR, "data/interpreted_articles.json")
    gdelt_path = os.path.join(BASE_DIR, "data/gdelt_filtered_events.json")

    # 데이터 로드
    articles = load_json(article_path)
    gdelt_events = load_json(gdelt_path)

    matcher = GDELTPolicyEventMatcher()

    # 첫 번째 기사 기준으로 테스트
    article = articles[0]
    print(f"📰 기사 제목: {article['title']}")
    print(f"📅 날짜: {article['date']}")
    print("🔗 URL:", article['url'])
    print("=" * 80)

    for block in article["text"]:
        intent = block.get("intent")
        keywords = block.get("keywords")
        text = block.get("text")
        uid = block.get("id")

        if not intent or not keywords:
            continue

        print(f"\n🧾 발언 #{uid}: {text}")
        print(f"👉 Intent: {intent}")
        print(f"🧩 Keywords: {keywords}")

        matched = matcher.match(
            utterance_info={
                "intent": intent,
                "keywords": keywords,
                "text": text,
                "id": uid
            },
            gdelt_events=gdelt_events,
            top_k=3,
            min_similarity=0.6
        )

        if not matched:
            print("❌ 유사한 사건 없음.")
        else:
            print("✅ 유사 GDELT 사건:")
            for i, m in enumerate(matched, 1):
                e = m["event"]
                print(f"  {i}. 🧠 유사도: {m['similarity']:.3f}")
                print(f"     📄 설명: {e.get('EventDescription')}")
                print(f"     🌍 지역: {e.get('ActionGeo_FullName')}")
                print(f"     🔢 코드: {e.get('EventCode')}")
                print("-" * 60)


if __name__ == "__main__":
    run_event_mapping_test()