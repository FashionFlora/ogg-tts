#!/usr/bin/env python3
"""Fish codec tokens for render_v1 (the 48 kHz files are only read; 44.1 kHz exists only in memory).

Audiobooks: the book is re-assembled in `order` (sentences + real pauses/non_text/lead/tail, pads stripped,
room_tone and overlap rows left out - render_books.py: this gives the master back) and streamed: only the
current ~40 s are in memory. It is resampled 48 -> 44.1 kHz window by window (window starts on multiples of
160 input samples, so every window lands on the same output grid as one resample of the whole book) and
encoded in 30 s chunks with 8 s of left context (the codec is causal), many chunks per GPU batch.
Neighbouring sentences can then be cut out of one continuous token stream.
Games: lines are encoded on their own, many per batch (right zero padding is harmless: causal codec).

Output per book: <out>/tokens/<key>.npy  int16 [10, frames]
                 <out>/index/<key>.parquet  one row per sentence: parquet, row, order, speaker, text,
                 language, book_id, superprod, start_frame, end_frame (end exclusive; games: whole line)
Usage: encode_tokens.py [--limit-books N] [--books substring] [--out DIR]
"""
import argparse
import glob
import hashlib
import io
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fish_codec as fc  # noqa: E402

ROOT = "/data/dataset/full"
META = "/data/dataset/build/meta_cache"
SR_IN, SR = 48000, fc.SAMPLE_RATE
UP, DOWN = 147, 160                     # 44100 / 48000 reduced
CHUNK = 30 * SR // fc.HOP               # frames encoded per chunk
CONTEXT = 8 * SR // fc.HOP              # frames of left context per chunk
MARGIN = 10 * DOWN                      # input samples around a resample window (filter support is ~80)


def resample(x):
    return AF.resample(x, SR_IN, SR, lowpass_filter_width=64, rolloff=0.9475,
                       resampling_method="sinc_interp_kaiser", beta=14.769656459379492)


class RowReader:
    """Decoded audio of (parquet, row), one cached row group per parquet (rows of a book come in order)."""

    def __init__(self):
        self.files, self.cache = {}, {}

    def __call__(self, parquet, row, strip_pads):
        if parquet not in self.files:
            pf = pq.ParquetFile(f"{ROOT}/{parquet}")
            starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
            self.files[parquet] = (pf, starts)
        pf, starts = self.files[parquet]
        rg = int(np.searchsorted(starts, row, "right") - 1)
        if self.cache.get(parquet, (None,))[0] != rg:
            self.cache[parquet] = (rg, pf.read_row_group(rg, columns=["audio", "pad_left", "pad_right"]))
        t, i = self.cache[parquet][1], row - starts[rg]
        blob = t.column(0)[i].as_py()
        a = sf.read(io.BytesIO(blob), dtype="float32")[0] if blob else np.zeros(0, np.float32)
        if strip_pads:
            a = a[(t.column(1)[i].as_py() or 0): len(a) - (t.column(2)[i].as_py() or 0)]
        return a


def encode_batch(model, segments):
    """segments: list of 1-D float tensors at 44.1 kHz on GPU -> list of codes [10, frames(segment)]"""
    longest = max(len(s) for s in segments)
    batch = torch.zeros(len(segments), longest, device="cuda")
    for i, s in enumerate(segments):
        batch[i, :len(s)] = s
    codes, _ = fc.encode(model, batch)
    return [codes[i, :, : -(-len(s) // fc.HOP)] for i, s in enumerate(segments)]


def encode_book(model, rows, reader, batch):
    rows = rows[(rows.order >= 0) & (rows.num_samples > 0)].sort_values("order")
    total_in = 0
    bounds = []
    buf, buf_start, out, pending = np.zeros(0, np.float32), 0, [], []
    frame = 0  # next output frame to encode

    def window(c0, f1, final):
        """44.1 kHz samples [c0*HOP, f1*HOP) cut from a resample of an aligned 48 kHz window"""
        lo44, hi44 = c0 * fc.HOP, f1 * fc.HOP
        s = max(0, (lo44 * DOWN // UP - MARGIN) // DOWN * DOWN)
        e = hi44 * DOWN // UP + MARGIN
        x = buf[s - buf_start: e - buf_start]
        if final and len(x) < e - s:
            x = np.pad(x, (0, e - s - len(x)))
        y = resample(torch.from_numpy(np.ascontiguousarray(x)).cuda())
        off = lo44 - s * UP // DOWN
        return y[off: off + hi44 - lo44]

    def flush(final=False):
        nonlocal frame, buf, buf_start
        total = -(-(total_in * UP // DOWN) // fc.HOP) if final else None
        while True:
            f1 = frame + CHUNK if total is None else min(total, frame + CHUNK)
            if f1 <= frame:
                break
            need_end = f1 * fc.HOP * DOWN // UP + MARGIN
            if not final and buf_start + len(buf) < need_end:
                break
            c0 = max(0, frame - CONTEXT)
            pending.append((frame, c0, f1, window(c0, f1, final)))
            frame = f1
            if len(pending) >= batch:
                run()
            keep_from = max(0, (max(0, frame - CONTEXT) * fc.HOP * DOWN // UP - MARGIN) // DOWN * DOWN)
            if keep_from > buf_start:
                buf, buf_start = buf[keep_from - buf_start:], keep_from
        if final and pending:
            run()

    def run():
        codes = encode_batch(model, [w for *_, w in pending])
        for (f0, c0, f1, _), c in zip(pending, codes):
            out.append(c[:, f0 - c0: f0 - c0 + (f1 - f0)])
        pending.clear()

    for p, r in zip(rows.parquet, rows.row):
        a = reader(p, int(r), strip_pads=True)
        bounds.append((total_in, total_in + len(a)))
        total_in += len(a)
        buf = np.concatenate([buf, a])
        flush()
    flush(final=True)
    codes = torch.cat(out, dim=1)
    scale = UP / DOWN / fc.HOP
    rows = rows.assign(start_frame=[int(s * scale) for s, _ in bounds],
                       end_frame=[min(codes.shape[1], int(np.ceil(e * scale))) for _, e in bounds])
    return codes, rows[rows.kind == "sentence"]


def encode_lines(model, rows, reader, batch):
    """games: every line on its own. Lines are read 2000 at a time (bounded memory), sorted by length and
    packed into GPU batches by padded size (batch x 38 s of audio), so hundreds of short lines go together."""
    rows = rows[rows.kind == "sentence"].sort_values("order")
    codes = [None] * len(rows)
    budget = batch * (CHUNK + CONTEXT) * fc.HOP

    def run(group):
        for i, c in zip([i for i, _ in group], encode_batch(model, [w for _, w in group])):
            codes[i] = c

    keys = list(zip(rows.parquet, rows.row))
    for s in range(0, len(keys), 2000):
        part = sorted(((i, resample(torch.from_numpy(reader(p, int(r), strip_pads=False)).cuda()))
                       for i, (p, r) in enumerate(keys[s:s + 2000], s)), key=lambda x: len(x[1]))
        group = []
        for item in part:
            if group and len(item[1]) * (len(group) + 1) > budget:
                run(group)
                group = []
            group.append(item)
        if group:
            run(group)
    ends = np.cumsum([c.shape[1] for c in codes])
    return torch.cat(codes, dim=1), rows.assign(start_frame=np.concatenate([[0], ends[:-1]]), end_frame=ends)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/dataset/tokens_fish")
    ap.add_argument("--books", default="", help="only books whose id contains this")
    ap.add_argument("--limit-books", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16, help="30 s chunks (or game lines) per GPU batch")
    ap.add_argument("--shard", default="0/1", help="i/n: this process encodes every n-th book starting at i")
    ap.add_argument("--fp32", action="store_true", help="exact fp32 (default TF32: ~1.6x faster, sem. tokens 98.5% equal)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = not args.fp32
    os.makedirs(f"{args.out}/tokens", exist_ok=True)
    os.makedirs(f"{args.out}/index", exist_ok=True)

    cols = ["superprod", "language", "book_id", "order", "kind", "text", "speaker", "num_samples", "parquet", "row"]
    books = defaultdict(list)
    for f in sorted(glob.glob(f"{META}/render_v1__*")):
        t = pq.read_table(f, columns=cols).to_pandas()
        for key, g in t.groupby(["superprod", "book_id"]):
            books[key].append(g)
    keys = sorted(k for k in books if args.books in k[1])
    # fixed shuffle: the first encoded books already cover every language and source (pilot training)
    keys = [keys[i] for i in np.random.default_rng(0).permutation(len(keys))]
    if args.limit_books:
        keys = keys[: args.limit_books]
    shard, nshards = map(int, args.shard.split("/"))
    keys = keys[shard::nshards]
    model = fc.load_codec()
    for n, (sp, book) in enumerate(keys, 1):
        name = f"{sp}__{book.replace('/', '__')}"
        name = name if len(name.encode()) < 200 else f"{sp}__{hashlib.md5(book.encode()).hexdigest()}"
        if os.path.exists(f"{args.out}/index/{name}.parquet"):
            continue
        rows = pd.concat(books[(sp, book)])
        encode = encode_lines if sp == "superprod_4" else encode_book
        codes, idx = encode(model, rows, RowReader(), args.batch)
        np.save(f"{args.out}/tokens/{name}.npy", codes.cpu().numpy().astype(np.int16))
        idx = idx.drop(columns=["num_samples", "kind"]).assign(tokens=f"{name}.npy")
        idx.to_parquet(f"{args.out}/index/{name}.parquet.tmp", index=False)
        os.replace(f"{args.out}/index/{name}.parquet.tmp", f"{args.out}/index/{name}.parquet")
        print(f"[{n}/{len(keys)}] {sp}/{book}: {codes.shape[1]} frames ({codes.shape[1] / 21.53 / 3600:.2f} h), "
              f"{len(idx)} sentences", flush=True)


if __name__ == "__main__":
    main()
