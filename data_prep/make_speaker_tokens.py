"""speaker_tokens.json: global speaker (ECAPA2 group, see make_speaker_map.py) -> a stable token name
"<iso>_<n>", for speakers with >= --min-hours of audio, numbered per language by hours (most first).
Frozen: names already in the file keep their number, new speakers are appended after the last one, so a
trained model's "speaker:pl_3" never changes meaning."""
import argparse
import json
import os
from collections import defaultdict

import pandas as pd

BUILD = "/data/dataset/build"
ISO = {"english": "en", "polish": "pl", "german": "de", "french": "fr", "russian": "ru", "italian": "it",
       "spanish": "es", "portuguese": "pt"}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--min-hours", type=float, default=0.5)
args = ap.parse_args()

d = json.load(open(f"{BUILD}/speaker_review/data.json", encoding="utf-8"))
m = pd.read_parquet(f"{BUILD}/speaker_map.parquet").set_index("key").global_speaker
hours = defaultdict(float)
for lang, L in d["languages"].items():
    for u in L["units"]:
        hours[m.get(u["id"], f"{lang}:{u['id']}")] += u["hours"]
path = f"{BUILD}/speaker_tokens.json"
tokens = json.load(open(path)) if os.path.exists(path) else {}
used = defaultdict(int)
for name in tokens.values():
    iso, n = name.rsplit("_", 1)
    used[iso] = max(used[iso], int(n) + 1)
new = 0
for spk, h in sorted(hours.items(), key=lambda x: -x[1]):
    if h < args.min_hours or spk in tokens:
        continue
    iso = ISO[spk.split(":", 1)[0]]
    tokens[spk] = f"{iso}_{used[iso]}"
    used[iso] += 1
    new += 1
json.dump(tokens, open(path, "w"), indent=0, ensure_ascii=False)
print(f"speaker tokens: {len(tokens)} ({new} new) | per language: {dict(used)}")
