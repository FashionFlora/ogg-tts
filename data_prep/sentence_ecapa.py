#!/usr/bin/env python3
"""Speaker check of every sentence in the multispeaker productions (book_id contains "multispeaker"):
diarization sometimes gives a line to the wrong character, and a span then switches voice mid-way.

Every sentence gets an ECAPA2 embedding (16 kHz, first --max-seconds). Per book: a robust centroid per
speaker (mean, then again without the samples < 0.5 from it), then for each sentence own = cosine to its
speaker's centroid and other = the best other speaker of the book (speakers with >= --min-speaker-seconds).
A sentence is a mismatch when it sounds clearly like another character (other - own > margin, other >= 0.5;
the margin is doubled under 1.5 s, where ECAPA2 is unreliable; for speakers too small for a centroid,
other >= 0.65) or like no one of the book (own < --min-own,
lines >= 1.5 s). Writes /data/dataset/build/sentence_speaker_check.parquet (every sentence, with scores)
and speaker_mismatch.parquet (parquet, row of the mismatches): training keeps them, as speaker_x of their
language, each one a span on its own.
Streams parquet by parquet; only embeddings stay in memory.
"""
import argparse
import glob
import io
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from huggingface_hub import hf_hub_download

BUILD, ROOT = "/data/dataset/build", "/data/dataset/full"


def read_rows(parquet, rows, cap):
    """{row: 16 kHz mono tensor (first cap samples)}, reading only the row groups that hold the rows"""
    pf = pq.ParquetFile(os.path.join(ROOT, parquet))
    starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
    by_rg = defaultdict(list)
    for r in rows:
        by_rg[int(np.searchsorted(starts, r, side="right") - 1)].append(r)
    out = {}
    for rg, rs in by_rg.items():
        col = pf.read_row_group(rg, columns=["audio"]).column(0)
        for r in rs:
            blob = col[r - starts[rg]].as_py()
            if not blob:
                continue
            audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
            out[r] = AF.resample(torch.from_numpy(audio.mean(axis=1)[: int(cap * sr / 16000) + sr]), sr, 16000)[:cap]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--min-speaker-seconds", type=float, default=60.0)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--min-own", type=float, default=0.35,
                    help="no_speaker below this (lines >= 1.5 s); generous on purpose: a flagged line is kept, "
                    "as a span of its own")
    ap.add_argument("--books", type=int, default=0, help="test: only the first N books")
    ap.add_argument("--book", default="", help="test: only books whose book_id contains this")
    ap.add_argument("--limit", type=int, default=0, help="test: only the first N sentences")
    ap.add_argument("--out", default=BUILD)
    args = ap.parse_args()

    cols = ["superprod", "book_id", "speaker", "kind", "num_samples", "sample_rate", "parquet", "row"]
    parts = []
    for f in sorted(glob.glob(f"{BUILD}/meta_cache/render_v1__superprod_[0-3]*")):
        t = pq.read_table(f, columns=cols).to_pandas()
        t = t[(t.kind == "sentence") & t.book_id.str.contains("multispeaker") & t.speaker.notna()]
        if len(t):
            parts.append(t.drop(columns="kind"))
    df = pd.concat(parts, ignore_index=True)
    df["speaker"] = df.speaker.astype(str)
    df["dur"] = df.num_samples / df.sample_rate
    if args.book:
        df = df[df.book_id.str.contains(args.book, regex=False)]
    if args.books:
        df = df[df.book_id.isin(df.book_id.drop_duplicates().head(args.books))]
    if args.limit:
        df = df.head(args.limit)
    df = df.reset_index(drop=True)
    print(f"{len(df)} sentences, {df.dur.sum() / 3600:.0f} h, {df.book_id.nunique()} books", flush=True)

    path = hf_hub_download("Jenthe/ECAPA2", "ecapa2.pt", cache_dir=f"{BUILD}/models")
    model = torch.jit.load(path, map_location=args.device).eval()
    cap = int(args.max_seconds * 16000)
    emb = np.zeros((len(df), 192), dtype=np.float16)
    pos = {(p, r): i for i, (p, r) in enumerate(zip(df.parquet, df.row))}

    def embed(clips):  # [(index, wave)]; sorted by length, each batch cropped to its shortest
        clips.sort(key=lambda c: len(c[1]))
        for i in range(0, len(clips), args.batch):
            chunk = clips[i:i + args.batch]
            n = max(1600, min(len(c[1]) for c in chunk))
            x = torch.stack([F.pad(c[1], (0, max(0, n - len(c[1]))))[:n] for c in chunk]).to(args.device)
            with torch.inference_mode():
                e = F.normalize(model(x).float(), dim=-1).cpu().numpy()
            emb[[c[0] for c in chunk]] = e

    parquets = df.groupby("parquet").row.apply(list).to_dict()
    t0, done = time.time(), 0
    with ThreadPoolExecutor(6) as pool:
        names = list(parquets)
        for start in range(0, len(names), 6):
            part = names[start:start + 6]
            clips = []
            for p, got in zip(part, pool.map(lambda p: read_rows(p, sorted(parquets[p]), cap), part)):
                clips += [(pos[(p, r)], w) for r, w in got.items()]
            embed(clips)
            done += len(clips)
            el = time.time() - t0
            print(f"{start + len(part)}/{len(names)} parquets | {done} sentences | {el / 60:.0f} min, "
                  f"ETA {el / done * (len(df) - done) / 60:.0f} min", flush=True)

    # per book: speaker centroids, own / best-other similarity
    df["own"], df["other"], df["other_speaker"] = np.nan, np.nan, None
    E = emb.astype(np.float32)
    for book, g in df.groupby("book_id"):
        cents = {}
        for spk, s in g.groupby("speaker"):
            if s.dur.sum() < args.min_speaker_seconds:
                continue
            x = E[s.index[s.dur >= 1.5]] if (s.dur >= 1.5).sum() >= 5 else E[s.index]
            c = x.mean(0)
            c /= np.linalg.norm(c) + 1e-9
            keep = x @ c >= 0.5
            if keep.sum() >= 3:
                c = x[keep].mean(0)
                c /= np.linalg.norm(c) + 1e-9
            cents[spk] = c
        if not cents:
            continue
        names = list(cents)
        S = E[g.index] @ np.stack([cents[n] for n in names]).T  # [sentences, speakers]
        own_i = np.array([names.index(s) if s in cents else -1 for s in g.speaker])
        own = np.where(own_i >= 0, S[np.arange(len(g)), np.maximum(own_i, 0)], np.nan)
        S2 = S.copy()
        S2[np.arange(len(g))[own_i >= 0], own_i[own_i >= 0]] = -1
        best = S2.argmax(1)
        df.loc[g.index, "own"] = own
        df.loc[g.index, "other"] = S2.max(1)
        df.loc[g.index, "other_speaker"] = [names[b] for b in best]
    margin = np.where(df.dur < 1.5, 2 * args.margin, args.margin)
    wrong = (df.other - df.own > margin) & (df.other >= 0.5)
    # speakers too small for a centroid: a line that clearly sounds like a main character of the book
    wrong |= df.own.isna() & (df.other >= 0.65)
    nobody = (df.own < args.min_own) & (df.dur >= 1.5)
    df["mismatch"] = np.where(wrong, "other_speaker", np.where(nobody, "no_speaker", ""))
    out = df.drop(columns=["num_samples", "sample_rate"])
    out.to_parquet(f"{args.out}/sentence_speaker_check.parquet", index=False)
    mm = out[out.mismatch != ""][["parquet", "row"]]
    mm.to_parquet(f"{args.out}/speaker_mismatch.parquet", index=False)
    checked = out.own.notna()
    print(f"checked {checked.sum()} sentences ({out[checked].dur.sum() / 3600:.0f} h) | "
          f"other speaker: {wrong.sum()} ({out[wrong].dur.sum() / 3600:.1f} h) | "
          f"no speaker: {(nobody & ~wrong).sum()} ({out[nobody & ~wrong].dur.sum() / 3600:.1f} h)", flush=True)


if __name__ == "__main__":
    main()
