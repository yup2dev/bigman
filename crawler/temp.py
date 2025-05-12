from typing import List, Dict, Optional
import re
import liwc

class LIWCAnalyzer:
    """Compute LIWC-based cognitive and emotion ratios using the liwc library."""
    def __init__(self, dict_path: str):
        """
        Initialize the LIWC analyzer by loading the LIWC dictionary file.
        :param dict_path: Path to the LIWC dictionary (.dic) file
        """
        # liwc.load_token_parser returns (categories, dictionary) parser
        self.categories, self.token_parser = liwc.load_token_parser(dict_path)

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze the text and return the proportion of tokens in cognitive and emotion categories.
        :param text: Input text to analyze
        :return: Dict with 'liwc_cognitive' and 'liwc_emotion' ratios
        """
        # Tokenize input text
        tokens = re.findall(r"\w+", text.lower())
        total = len(tokens) or 1
        # Parse tokens to LIWC categories
        counts = {'cognitive': 0, 'emotion': 0}
        for token in tokens:
            for cat in self.token_parser(token):
                # token_parser returns category strings
                if cat.lower() in self.categories:
                    # Check category hierarchy for cognitive and emotion
                    if cat.startswith('Cogmech') or cat.lower() == 'cognitive':
                        counts['cognitive'] += 1
                    if cat.startswith('Affect') or cat.lower() == 'emotion':
                        counts['emotion'] += 1
        return {
            'liwc_cognitive': counts['cognitive'] / total,
            'liwc_emotion': counts['emotion'] / total
        }

# The rest of the feature processors remain unchanged
class EmotionArousalScorer:
    """Compute emotion arousal score based on intensity keywords or lexicon."""
    def __init__(self, arousal_lexicon: Dict[str, float]):
        self.lexicon = arousal_lexicon

    def score(self, text: str) -> float:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return 0.0
        total = len(tokens)
        score_sum = sum(self.lexicon.get(t, 0.0) for t in tokens)
        return score_sum / total

class KeywordRarityScorer:
    """Compute rarity of keywords in a text against a corpus frequency dict."""
    def __init__(self, corpus_freq: Dict[str, int], total_tokens: int):
        self.corpus_freq = corpus_freq
        self.total_tokens = total_tokens

    def score(self, text: str) -> float:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return 0.0
        rarities = []
        for t in tokens:
            freq = self.corpus_freq.get(t, 0)
            # Rarity = 1 - normalized frequency
            rarities.append(1 - (freq / self.total_tokens))
        return sum(rarities) / len(rarities)

class StructuralEmphasisScorer:
    """Compute structural emphasis based on Q&A, repetition, or emphasis patterns."""
    def __init__(self):
        pass

    def score(self, text: str) -> float:
        score = 0.0
        # Q&A pattern
        if re.search(r"\?:", text):
            score += 0.2
        # repetition pattern
        if len(set(text.lower().split())) < len(text.split()):
            score += 0.2
        # emphasis words
        emphasis_words = ['really', 'very', 'absolutely']
        tokens = text.lower().split()
        score += sum(0.1 for t in tokens if t in emphasis_words)
        return min(score, 1.0)

class MemorabilityCalculator:
    """Combine arousal, rarity, structure into a single memorability score."""
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {'arousal': 0.4, 'rarity': 0.3, 'structure': 0.3}

    def compute(self, arousal: float, rarity: float, structure: float) -> float:
        return (self.weights['arousal'] * arousal +
                self.weights['rarity'] * rarity +
                self.weights['structure'] * structure)

class EventMapper:
    """Map an utterance to an event type using embedding similarity or classifier."""
    def __init__(self, event_db: List[Dict], embedder):
        self.event_db = event_db
        self.embedder = embedder

    def map(self, text: str, date: str) -> Dict:
        utter_emb = self.embedder.embed(text)
        best = {'event_id': None, 'type': None, 'confidence': 0.0}
        for evt in self.event_db:
            evt_emb = self.embedder.embed(evt['description'])
            sim = sum(u * v for u, v in zip(utter_emb, evt_emb))
            if sim > best['confidence']:
                best = {'event_id': evt['event_id'], 'type': evt['type'], 'confidence': sim}
        return best

class TranscriptProcessor:
    """Orchestrates feature extraction and event mapping for transcripts."""
    def __init__(
        self,
        liwc_dict_path: str,
        arousal_lexicon: Dict[str, float],
        corpus_freq: Dict[str, int],
        total_tokens: int,
        event_db: List[Dict],
        embedder
    ):
        self.liwc = LIWCAnalyzer(liwc_dict_path)
        self.arousal = EmotionArousalScorer(arousal_lexicon)
        self.rarity = KeywordRarityScorer(corpus_freq, total_tokens)
        self.structure = StructuralEmphasisScorer()
        self.memo = MemorabilityCalculator()
        self.mapper = EventMapper(event_db, embedder)

    def process(self, transcript: Dict) -> Dict:
        processed = {
            'url': transcript['url'],
            'date': transcript['date'],
            'title': transcript['title'],
            'text': []
        }
        for utt in transcript['text']:
            text = utt['text']
            liwc_feats = self.liwc.analyze(text)
            arousal_score = self.arousal.score(text)
            rarity_score = self.rarity.score(text)
            structure_score = self.structure.score(text)
            mem_score = self.memo.compute(arousal_score, rarity_score, structure_score)
            event = self.mapper.map(text, transcript['date'])

            entry = {
                'id': utt['id'],
                'speaker': utt.get('speaker'),
                'utterance': text,
                **liwc_feats,
                'stance_label': utt.get('stance_label'),
                'emotion_arousal': arousal_score,
                'keyword_rarity': rarity_score,
                'structural_emphasis': structure_score,
                'memorability_score': mem_score,
                'memory_id': f"M{transcript['date'][:10].replace('-', '')}_{utt['id']:02d}",
                'mapped_event': event
            }
            processed['text'].append(entry)
        return processed
