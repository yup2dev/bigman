from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
import pandas as pd

# 📁 pipeline/event_mapping_pipeline.py

from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from data.gdelt_event_loader import GDELTEventLoader


class BeliefExtractor:
    def extract(self, text: str) -> str:
        """발언에서 발화자의 의도나 판단을 요약"""
        if "perfect" in text.lower():
            return "positive support for a specific plan"
        elif "unacceptable" in text.lower() or "never accept" in text.lower():
            return "opposition to an alternative plan"
        return "general statement"


class EventMatcher:
    def __init__(self, event_db: List[Dict], model_name: str = 'all-MiniLM-L6-v2'):
        self.event_db = event_db
        self.model = SentenceTransformer(model_name)
        self.event_embeddings = self.model.encode([e['summary'] for e in event_db], convert_to_tensor=True)

    def match(self, belief: str, date: str, id_num: int, top_k: int = 1) -> Dict:
        belief_emb = self.model.encode(belief, convert_to_tensor=True)
        similarities = util.cos_sim(belief_emb, self.event_embeddings)[0]

        best_idx = int(similarities.argmax())
        best_event = self.event_db[best_idx]
        score = float(similarities[best_idx])

        return {
            "event_id": best_event["event_id"],
            "type": best_event["type"],
            "relation": "aligned_with_belief",
            "confidence": round(score, 4)
        }


class EventMappingPipeline:
    def __init__(self, zip_url: str, keywords: List[str]):
        loader = GDELTEventLoader(zip_url)
        event_db = loader.filter_events(keywords=keywords)
        self.extractor = BeliefExtractor()
        self.matcher = EventMatcher(event_db)

    def map_utterances(self, utterances: List[Dict]) -> List[Dict]:
        results = []
        for u in utterances:
            belief = self.extractor.extract(u['text'])
            mapped_event = self.matcher.match(belief, u['date'], u['id'])
            results.append({
                **u,
                "belief": belief,
                "mapped_event": mapped_event
            })
        return results



class GDELTEventLoader:
    def __init__(self, zip_url: str):
        self.zip_url = zip_url
        self.columns_url = "https://gdeltproject.org/data/lookups/CSV.header.dailyupdates.txt"
        self.columns = self._load_column_names()
        self.df = self._load_data()

    def _load_column_names(self) -> List[str]:
        return pd.read_csv(self.columns_url, header=None)[0].tolist()

    def _load_data(self) -> pd.DataFrame:
        return pd.read_csv(
            self.zip_url,
            compression='zip',
            header=None,
            names=self.columns,
            sep='\t',
            low_memory=False
        )

    def filter_events(self,
                      min_date: int = None,
                      country_code: str = "US",
                      keywords: List[str] = None) -> List[Dict]:

        df = self.df.copy()

        if min_date:
            df = df[df["SQLDATE"] >= min_date]

        if country_code:
            df = df[df["ActionGeo_CountryCode"] == country_code]

        if keywords:
            pattern = '|'.join(keywords)
            df = df[df["EventDescription"].str.contains(pattern, case=False, na=False)]

        # 결과 포맷 정리
        events = []
        for _, row in df.iterrows():
            events.append({
                "event_id": f"GDELT_{row['SQLDATE']}_{row['GLOBALEVENTID']}",
                "date": str(row['SQLDATE']),
                "type": row['EventCode'],
                "summary": row['EventDescription'],
                "actor1": row['Actor1Name'],
                "actor2": row['Actor2Name'],
                "location": row['ActionGeo_FullName']
            })

        return events