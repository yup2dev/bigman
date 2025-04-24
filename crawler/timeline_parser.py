# -*- coding: utf-8 -*-
"""
Factba.se Transcript Loader → timeline_step JSONL
-------------------------------------------------
Fetches Donald Trump's public remarks (speeches, interviews, rallies) from
Factba.se API, extracts passage blocks, and converts them into the same
`timeline_step` schema used by the Wiki/Bio builder.

★ Usage (example – pull 2017 only) ★
    python factbase_transcript_loader.py \
        --year 2017 \
        --openai-key $OPENAI_KEY \
        -o data/timeline/factbase_2017.jsonl

* The script respects Factba.se free‑tier rate limits (20 req/min).
* Each transcript is sliced into ~250‑word passages and passed to GPT to
  produce `{date,event,motivation,action,result,tone}` steps.
* Deduplicates by (date,event).
"""

from __future__ import annotations
import argparse, os, json, re, textwrap, time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import requests
from tqdm import tqdm
import openai

BASE_DIR = Path(__file__).resolve().parents[1]

FACTBASE_ENDPOINT = "https://factba.se/api/v1/transcripts"
TRANSCRIPT_URL_TMPL = "https://factba.se/transcript/%s"
PASSAGE_LEN = 250  # words per chunk
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

PROMPT = textwrap.dedent(
    """Extract a structured timeline step for Donald Trump from the passage. Return JSON only: {"date":"...","event":"...","belief":"...","motivation":"...","action":"...","result":"...","tone":"..."}\nPassage:\n"{passage}"""
)

# ---------------------------------------------------------------------------
# Factbase helpers
# ---------------------------------------------------------------------------

def fetch_factbase(page:int=1,year:int|None=None)->List[Dict]:
    params = {"page":page,"format":"json"}
    if year:
        params["year"] = str(year)
    r=requests.get(FACTBASE_ENDPOINT, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("transcripts", [])


def iter_transcripts(year:int|None)->Dict[str,str]:
    """Yield (date, full_text) per transcript."""
    page=1
    while True:
        rows=fetch_factbase(page,year)
        if not rows:
            break
        for row in rows:
            date=row.get("date","")[:10]
            text=row.get("transcript","") or row.get("text","")
            if text:
                yield date,text
        page+=1; time.sleep(3)  # rudimentary rate‑limit sleep

# ---------------------------------------------------------------------------
# GPT struct & clean
# ---------------------------------------------------------------------------

def gpt_step(passage:str)->Optional[Dict]:
    msg=[{"role":"user","content":PROMPT.format(passage=passage)}]
    try:
        ret=openai.chat.completions.create(model="gpt-3.5-turbo-0125",messages=msg,temperature=0.1)
        txt=ret.choices[0].message.content.strip()
        return json.loads(txt[txt.index("{"): txt.rindex("}")+1])
    except Exception:
        return None


def split_passages(text:str,length:int=PASSAGE_LEN)->List[str]:
    words=text.split()
    return [" ".join(words[i:i+length]) for i in range(0,len(words),length)]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_from_factbase(year:int|None,outfile:Path,openai_key:str,max_transcripts:int=200):
    openai.api_key=openai_key
    outfile.parent.mkdir(parents=True,exist_ok=True)
    seen:Set[Tuple[str,str]]=set()

    with outfile.open("w",encoding="utf-8") as fout:
        for idx,(date,text) in enumerate(tqdm(iter_transcripts(year),desc="Factbase")):
            if idx>=max_transcripts:
                break
            for passage in split_passages(text):
                step=gpt_step(passage)
                if not step:
                    continue
                # simple date override by API date to ensure ISO
                step["date"] = f"{date}" if DATE_RE.match(date) else step.get("date","")
                key=(step.get("date",""),step.get("event",""))
                if key in seen or not key[0]:
                    continue
                seen.add(key)
                step["source_external"] = TRANSCRIPT_URL_TMPL % date
                fout.write(json.dumps(step,ensure_ascii=False)+"\n")
    print("✅",len(seen),"steps →",outfile)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--year",type=int,default=None,help="year filter (optional)")
    ap.add_argument("-o","--output",type=Path,required=True)
    ap.add_argument("--openai-key",default=os.getenv("OPENAI_API_KEY"))
    args=ap.parse_args()

    if not args.openai_key:
        raise SystemExit("OPENAI_API_KEY missing")

    build_from_factbase(args.year,args.output,args.openai_key)
