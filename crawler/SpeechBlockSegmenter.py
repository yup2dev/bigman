from typing import List, Dict


class SpeechBlockSegmenter:
    def __init__(self):
        pass

    def segment_as_qa_pairs(self, segments: List[Dict]) -> List[List[Dict]]:
        blocks = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            text = seg.get("text", "")
            block = [seg]

            # 질문이면 다음 발화까지 포함
            if text.strip().endswith("?") and i + 1 < len(segments):
                block.append(segments[i + 1])
                i += 1

            blocks.append(block)
            i += 1

        return blocks