"""speaker_map.parquet: unit key -> global speaker (one per language: an ECAPA2 group from
speaker_review/data.json suggested_groups, else the unit itself). Keys are the speaker_review unit ids
plus one per extra speaker id of a unit: books <root>/spk<speaker> (root = <superprod>/<book_id> after
folder_map.json), games superprod_4/<base language>/<game>/<speaker>. speaker_key() builds the same key
for token index rows."""
import json
import os
import sys

import numpy as np
import pandas as pd

BUILD = "/data/dataset/build"
sys.path.insert(0, BUILD)
import folder_map  # noqa: E402


def speaker_key(df):
    """index rows (superprod, book_id, speaker) -> unit keys"""
    spk = df.speaker.astype(str).to_numpy(dtype=object)
    game = (df.superprod == "superprod_4").to_numpy()
    book = df.book_id.to_numpy(dtype=object)
    roots = folder_map.roots(df)
    out = np.empty(len(df), dtype=object)
    for i in np.flatnonzero(~game):
        out[i] = f"{roots[i]}/spk{spk[i]}"
    for i in np.flatnonzero(game):
        g, lang = book[i].split("/", 1)
        out[i] = f"superprod_4/{lang.split('_')[0]}/{g}/{spk[i]}"
    return out


if __name__ == "__main__":
    d = json.load(open(f"{BUILD}/speaker_review/data.json", encoding="utf-8"))
    rows = []
    for lang, L in d["languages"].items():
        group = {u: f"{lang}:g{i}" for i, g in enumerate(L["suggested_groups"]) for u in g}
        for u in L["units"]:
            spk = group.get(u["id"], f"{lang}:{u['id']}")
            rows.append((u["id"], spk))
            if not u["id"].startswith("superprod_4"):
                base = u["id"].rsplit("/", 1)[0]
                rows += [(f"{base}/spk{s}", spk) for s in u["speaker_ids"]]
    m = pd.DataFrame(rows, columns=["key", "global_speaker"]).drop_duplicates("key")
    m.to_parquet(f"{BUILD}/speaker_map.parquet", index=False)
    print(f"speaker map: {len(m)} keys -> {m.global_speaker.nunique()} global speakers")
