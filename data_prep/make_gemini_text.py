#!/usr/bin/env python3
"""Gemini text (inline tags + spoken-form transcript) for training -> /data/dataset/build/gemini_text.parquet.

Reads /data/dataset/gemini_run/results/*.jsonl (no pilots) and keeps only valid rows of the current prompt
generations (old prompt 8579217c6f and earlier are excluded). One row per sentence (book_id, order): the newest
prompt generation wins. Columns: book_id, order, text_gemini, n_tags, model, prompt_version.
Re-run any time; training re-reads the file at every index reload (--reload-every).
"""
import glob
import json
import os
import re

import pandas as pd

# prompt generations, oldest first: v1 (tags) -> v2 (+[frustrated]) -> v3 (+audio_evidence_only)
GENERATIONS = [
    {"1295050331", "7b24b9bcb1", "2972844aa1", "499b3d5b27"},
    {"46ee5983d2", "516f7cdbf6", "72439858ca", "a94234405d"},
    {"7aeb8160b0", "29fe429d98", "5d40a2cac2", "4c93f31ad1"},
]
RANK = {v: g for g, vs in enumerate(GENERATIONS) for v in vs}
TAGS = json.load(open("/data/dataset/train/gemini/tags_spec.json", encoding="utf-8"))["inline_tags"]
TAG_RE = re.compile(r"\[(?:%s)\]" % "|".join(re.escape(t[1:-1]) for v in TAGS.values() for t in v))
OUT = "/data/dataset/build/gemini_text.parquet"


def main():
    rows = []
    for f in glob.glob("/data/dataset/gemini_run/results/*.jsonl"):
        if ".pilot" in f:
            continue
        for line in open(f, encoding="utf-8"):
            try:
                j = json.loads(line)
            except Exception:
                continue
            v = j.get("prompt_version")
            if v not in RANK or not j.get("valid") or not (j.get("gemini_text") or "").strip():
                continue
            t = j["gemini_text"].strip()
            rows.append((j["book_id"], int(j["order"]), t, len(TAG_RE.findall(t)), j.get("model", "gemini-3.8-flash"),
                         v, RANK[v]))
    df = pd.DataFrame(rows, columns=["book_id", "order", "text_gemini", "n_tags", "model", "prompt_version", "gen"])
    df = df.sort_values("gen").drop_duplicates(["book_id", "order"], keep="last").drop(columns="gen")
    df["order"] = df.order.astype("int32")
    tmp = OUT + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, OUT)
    print(f"{len(df)} sentences, {int((df.n_tags > 0).sum())} with tags, {int(df.n_tags.sum())} tags -> {OUT}")


if __name__ == "__main__":
    main()
