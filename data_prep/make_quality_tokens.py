"""quality_tokens.json: recording (quality) key -> stable token "q<n>": one per book (root after folder_map.json,
quality_groups.json joins volumes recorded together, e.g. Monte-Cristo T1+T2), one per game and language.
Numbered by hours (most first); frozen like speaker_tokens.json: existing names never change.
quality_key() builds the key for token index / metadata rows."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BUILD = "/data/dataset/build"
sys.path.insert(0, BUILD)
import folder_map  # noqa: E402

GROUPS = json.load(open(f"{BUILD}/quality_groups.json", encoding="utf-8"))


def quality_key(df):
    roots = folder_map.roots(df)
    game = (df.superprod == "superprod_4").to_numpy()
    return np.where(game, "superprod_4/" + df.book_id.to_numpy(dtype=object), [GROUPS.get(r, r) for r in roots])


if __name__ == "__main__":
    parts = []
    for f in glob.glob(f"{BUILD}/meta_cache/render_v1__*"):
        t = pq.read_table(f, columns=["superprod", "book_id", "kind", "num_samples", "sample_rate"]).to_pandas()
        t = t[t.kind == "sentence"]
        parts.append(t.assign(h=t.num_samples / t.sample_rate / 3600)[["superprod", "book_id", "h"]])
    t = pd.concat(parts, ignore_index=True)
    t["key"] = quality_key(t)
    hours = t.groupby("key").h.sum().sort_values(ascending=False)
    path = f"{BUILD}/quality_tokens.json"
    tokens = json.load(open(path)) if os.path.exists(path) else {}
    n = 1 + max([int(v[1:]) for v in tokens.values()], default=-1)
    for k in hours.index:
        if k not in tokens:
            tokens[k] = f"q{n}"
            n += 1
    json.dump(tokens, open(path, "w"), indent=0, ensure_ascii=False)
    print(f"quality tokens: {len(tokens)} recordings ({(hours.index.str.startswith('superprod_4')).sum()} game/language)")
