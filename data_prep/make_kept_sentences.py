"""kept_sentences.parquet: (parquet, row) of every sentence usable for training - text label ok
(render_v1_labels for audiobooks, meta.text_label for games), non-empty text, not in drop_clips.tsv.
Same filter as build_train_list.load_sentences, but file by file so it needs little memory; training
reads this small file instead of the whole metadata."""
import glob
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

BUILD, ROOT = "/data/dataset/build", "/data/dataset/full"
sys.path.insert(0, BUILD)
import drops  # noqa: E402

# ok clips per (superprod, book): small per-book sets, so each metadata file is checked cheaply
labels = {}
for f in glob.glob(f"{ROOT}/render_v1_labels/*_text_labels.parquet"):
    sp = os.path.basename(f).replace("_text_labels.parquet", "")
    t = pq.read_table(f, columns=["book_id", "file", "label"]).to_pandas()
    for book, g in t[t.label == "ok"].groupby("book_id"):
        labels[(sp, book)] = set(g.file)

parts = []
for f in sorted(glob.glob(f"{BUILD}/meta_cache/render_v1__*")):
    t = pq.read_table(f, columns=["superprod", "book_id", "speaker", "kind", "text", "file", "meta", "parquet", "row"]).to_pandas()
    t = t[(t.kind == "sentence") & t.speaker.notna() & t.text.notna()]
    t = t[t.text.str.strip() != ""]
    if t.empty:
        continue
    if t.superprod.iloc[0] == "superprod_4":
        ok = t.meta.fillna("").str.contains('"text_label": "ok"', regex=False)
    else:
        sp = t.superprod.iloc[0]
        ok = pd.Series([f in labels.get((sp, b), ()) for b, f in zip(t.book_id, t.file)], index=t.index)
    parts.append(t[ok][["parquet", "row"]])
kept = drops.apply(pd.concat(parts, ignore_index=True))
kept.to_parquet(f"{BUILD}/kept_sentences.parquet", index=False)
print(f"kept sentences: {len(kept)}")
