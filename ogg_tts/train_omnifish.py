#!/usr/bin/env python3
"""Train OmniFish (OmniVoice backbone + Fish codec tokens) on /data/dataset/tokens_fish.

OmniFish v2 (semantic predictor + MTP decoder, see omnifish.py). Samples: audiobooks - 1..--max-sentences successive
kept sentences of one speaker in one book (other speakers' lines in between are skipped), spliced from the book's
continuous token stream: each sentence with the plain pause that follows it in the book
(/data/dataset/build/sentence_pauses.parquet), at most --max-pause seconds (a longer pause loses its middle, both
edges stay), the last one without; up to --max-seconds.
Music, effects and other untranscribed audio never enter a span. Games - single lines.
Only sentences in /data/dataset/build/kept_sentences.parquet (text label ok, not in drop_clips.tsv) are used.
Multispeaker lines whose voice does not match their label (/data/dataset/build/speaker_mismatch.parquet, from
sentence_ecapa.py) stay in, prompt only (no speaker token), each a span on its own, balanced together as speaker_x
of their language.
Text context (audiobooks): up to 2 sentences before the span and up to 2 after it (any speaker, text only, never
spoken), as <|object_ref_start|>before<|object_ref_end|> ... <|box_start|>after<|box_end|> around the text; how many
is drawn per sample (--context-before / --context-after probabilities), so the model also works without it.
Speaker token (instruct field): speakers with >= 30 min (/data/dataset/build/speaker_tokens.json, frozen names)
get "speaker:pl_3"; everyone else - small speakers, game extras - the bag "speaker:pl_x". Per
sample (--speaker-mode weights): token + prompt / token only (no prompt frames) / prompt only (instruct None);
bag samples always carry their bag token and their prompt.
Scene windows (multispeaker books, once sentence_ecapa.py has checked their lines): 1..--max-sentences consecutive
lines as they come; one voice -> an ordinary span, several -> a scene, text "<|quad_start|>pl_3<|quad_end|>line
<|quad_start|>pl_7<|quad_end|>line", instruct "dialogue", no prompt; --dialogue = share of the windows used.
--dialogue-prompt of the scenes name the voices S1, S2, ... and carry one prompt per voice (another 2-6 s line of
that speaker, spoken first, text "<|quad_start|>S1<|quad_end|>its line ..." before the scene): dialogue of any voices.
Recording quality: "quality:q35" (/data/dataset/build/quality_tokens.json: one per book, volumes recorded together
joined, one per game+language) appended to the instruct in --quality-ratio of the samples.
Speaker balance: each index round keeps a span with probability min(1, --speaker-cap / hours of its global
speaker) (ECAPA2 groups, /data/dataset/build/speaker_map.parquet), so the most recorded narrators cannot
dominate; rounds are re-drawn every --reload-every steps, so all their audio still comes up over time.
Block diffusion (LLaDA2.0 / BD3-LM, see omnifish.py): prompt = first 0..30 % of the frames (no loss); the rest
is the target - its semantic ids, then END and the Rainbow PAD cycle up to a multiple of the block size -
present twice, clean and noisy; every noisy block is masked at its own uniform random ratio. 10 %
unconditional (no text/language/prompt), language tag in 80 % of the rest, instruct = speaker token or "None".
Gemini text (/data/dataset/build/gemini_text.parquet, make_gemini_text.py) replaces a sentence's text where it has one:
inline audio tags ([sighing], [shouting], ...) plus the checked spoken-form transcript; spans with tags come up
--tag-oversample times, and in --tag-drop of the samples the tags are stripped, so they stay optional.
The block size follows --block-schedule (large blocks first, as LLaDA2.0's decay to small blocks).
Tokens are memory-mapped, never loaded.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/dataset/build/scripts")
sys.path.insert(0, "/data/dataset/build")
import make_quality_tokens  # noqa: E402
import make_speaker_map  # noqa: E402
import omnifish as of  # noqa: E402

FPS = 44100 / 2048
# inline audio tags of the Gemini pass (closed list, /data/dataset/train/gemini/tags_spec.json)
_TAGS = json.load(open("/data/dataset/train/gemini/tags_spec.json", encoding="utf-8"))["inline_tags"]
TAG_RE = re.compile(r"\[(?:%s)\]" % "|".join(re.escape(t[1:-1]) for v in _TAGS.values() for t in v))


def strip_tags(text):
    return re.sub(r"\s{2,}", " ", TAG_RE.sub("", text)).strip()


ISO = {"en": "en", "english": "en", "pl": "pl", "polish": "pl", "de": "de", "german": "de", "fr": "fr",
       "french": "fr", "es": "es", "spanish_es": "es", "spanish_latam": "es", "it": "it", "italian": "it",
       "pt": "pt", "portuguese_pt": "pt", "portuguese_br": "pt", "ru": "ru", "russian": "ru"}


def load_index(tokens_dir, val_books):
    """sentence index of every encoded book, with a `keep` flag; split by book into train / val"""
    idx = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{tokens_dir}/index/*.parquet")], ignore_index=True)
    kept = pd.read_parquet("/data/dataset/build/kept_sentences.parquet").assign(keep=True)
    idx = idx.merge(kept, on=["parquet", "row"], how="left").fillna({"keep": False})
    pauses = pd.read_parquet("/data/dataset/build/sentence_pauses.parquet")
    idx = idx.merge(pauses, on=["parquet", "row"], how="left").fillna({"pause_after": 0.0})
    # Gemini text (make_gemini_text.py): inline tags + checked spoken-form transcript, where the sentence has one
    gt = "/data/dataset/build/gemini_text.parquet"
    if os.path.exists(gt):
        idx = idx.merge(pd.read_parquet(gt, columns=["book_id", "order", "text_gemini"]), on=["book_id", "order"], how="left")
        idx["text"] = idx.text_gemini.fillna(idx.text)
        idx = idx.drop(columns="text_gemini")
    # global speaker (ECAPA2 group); rows without a speaker_review unit are their own speaker
    idx["key"] = make_speaker_map.speaker_key(idx)
    idx = idx.merge(pd.read_parquet("/data/dataset/build/speaker_map.parquet"), on="key", how="left")
    idx["global_speaker"] = idx.global_speaker.fillna(idx.key)
    idx["lang"] = idx.language.map(ISO)
    # multispeaker lines whose voice does not match their label (sentence_ecapa.py): kept, prompt only (no
    # speaker token: the voice has no name), each its own span (a unique speaker id, so nothing joins them),
    # balanced together as speaker_x of their language
    mm = "/data/dataset/build/speaker_mismatch.parquet"
    x = np.zeros(len(idx), dtype=bool)
    if os.path.exists(mm):
        x = idx.merge(pd.read_parquet(mm).assign(odd=True), on=["parquet", "row"], how="left").odd.fillna(False).to_numpy()
        idx.loc[x, "global_speaker"] = idx.lang[x] + ":speaker_x"
        idx.loc[x, "speaker"] = "x" + idx.order[x].astype(str)
    tokens = json.load(open("/data/dataset/build/speaker_tokens.json"))
    idx["spk_token"] = idx.global_speaker.map(tokens).fillna(idx.lang + "_x")
    idx.loc[x, "spk_token"] = ""
    quality = json.load(open("/data/dataset/build/quality_tokens.json"))
    idx["q_token"] = pd.Series(make_quality_tokens.quality_key(idx), index=idx.index).map(quality).fillna("")
    # stable split: a book is validation iff its name hashes into 1/val_books, so it never moves as books arrive
    import hashlib
    val = {b for b in idx.tokens.unique() if int(hashlib.md5(b.encode()).hexdigest(), 16) % val_books == 0}
    return idx[~idx.tokens.isin(val)], idx[idx.tokens.isin(val)]


def pause_pieces(r):
    """the kept part of the pause after line r: [end, pause_end) and, when its middle was cut, [tail_start, tail_end)"""
    return (int(r.end_frame), int(r.pause_end)), (int(r.tail_start), int(r.tail_end))


def pause_len(r):
    return sum(b - a for a, b in pause_pieces(r))


def splice(rows):
    """frame segments of successive lines, each but the last with its kept pause; touching segments merged"""
    merged = []
    for j, r in enumerate(rows):
        for a, b in ((int(r.start_frame), int(r.end_frame)),) + (pause_pieces(r) if j < len(rows) - 1 else ()):
            if b <= a:
                continue
            if merged and merged[-1][1] == a:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
    return tuple(merged)


def build_spans(idx, max_sentences, max_frames, max_gap, max_skip, seed=0, dialogue=0.0, dialogue_prompt=0.0):
    """(tokens file, ((start, end), ...) frame segments, text, lang, global speaker, (2 sentences before, 2 after))
    for spans of 1..max_sentences kept
    sentences of one speaker; every segment but the last carries the sentence's own following pause (<= max_gap);
    successive sentences lie at most max_skip frames apart in the book (other speakers' lines in between).
    dialogue > 0 (multispeaker books): also windows of 1..max_sentences consecutive lines as they come, each kept with
    probability `dialogue`: one voice -> an ordinary span, several -> a scene with turn markers ("dialogue")."""
    rng = random.Random(seed)
    spans = []
    for tok, g in idx.sort_values("order").groupby("tokens", sort=False):
        game = g.superprod.iloc[0] == "superprod_4"
        nxt = g.start_frame.shift(-1, fill_value=2**62).to_numpy()  # next sentence of any speaker: pause ends there
        # a pause longer than max_gap loses its middle: half of max_gap stays after the sentence, half before the
        # pause ends (the decay and the next breath are kept); [end, pause_end) + [tail_start, tail_end)
        p = np.minimum((g.pause_after.fillna(0) * FPS).round().to_numpy(), nxt - g.end_frame.to_numpy()).clip(0).astype(np.int64)
        e, h, cut = g.end_frame.to_numpy(), max_gap // 2, p > max_gap
        g = g.assign(pause_end=np.where(cut, e + h, e + p), tail_start=np.where(cut, e + p - (max_gap - h), e + p),
                     tail_end=e + p)
        # text context: the book's neighbouring sentences (any speaker); none for games (line order is no dialogue)
        t = g.text.fillna("").map(strip_tags)  # context is plain text
        g = g.assign(**({k: "" for k in ("b2", "b1", "a1", "a2")} if game else
                        dict(b2=t.shift(2, fill_value=""), b1=t.shift(1, fill_value=""),
                             a1=t.shift(-1, fill_value=""), a2=t.shift(-2, fill_value=""))))
        g = g.assign(pos=np.arange(len(g)))  # position among all the book's lines: consecutive lines in a scene
        g = g[g.keep & (g.end_frame > g.start_frame)]
        if dialogue > 0 and not game and "multispeaker" in g.book_id.iloc[0]:
            spans += dialogue_spans(tok, list(g.itertuples(index=False)), rng, dialogue, max_sentences, max_frames,
                                    dialogue_prompt)
        for _, sg in g.groupby("speaker", sort=False):
            rows = list(sg.itertuples(index=False))
            i = 0
            while i < len(rows):
                k = 1 if game else rng.randint(1, max_sentences)
                seg, text, total = [], [], 0
                for r in rows[i:i + k]:
                    n = r.end_frame - r.start_frame
                    pause = pause_len(seg[-1]) if seg else 0  # the previous sentence keeps its pause
                    if seg and (total + pause + n > max_frames or r.start_frame - seg[-1].end_frame > max_skip):
                        break
                    seg.append(r)
                    text.append(r.text.strip())
                    total += pause + n
                if total <= max_frames:
                    first, last = rows[i], rows[i + len(seg) - 1]
                    spans.append((tok, splice(seg), " ".join(text), first.lang, first.global_speaker,
                                  ((first.b2, first.b1), (last.a1, last.a2)), first.spk_token, first.q_token))
                i += len(seg)
    return spans


def dialogue_spans(tok, rows, rng, keep, max_lines, max_frames, prompted=0.0):
    """windows of 1..max_lines consecutive kept lines (no line of the book skipped), whoever speaks; a window stops
    at a speaker without an own token (bag, speaker_x: no voice to name). Each line keeps its own pause (at most
    max_gap, middle cut), the last none. One speaker in the window -> an ordinary span; several -> a scene with turn markers.
    `prompted` of the scenes instead name their voices S1, S2, ... (first appearance) and carry one prompt per voice:
    another 2..6 s line of that speaker from the same book, outside the window, spoken first; text
    "<|quad_start|>S1<|quad_end|>prompt line <|quad_start|>S2<|quad_end|>prompt line" + the scene with S markers,
    instruct "dialogue", spk_token "dialogue_p" and the prompt length in frames as 9th field."""
    pool = {}
    for r in rows:  # prompt candidates per speaker token
        if r.spk_token and not r.spk_token.endswith("_x") and 2 * FPS <= r.end_frame - r.start_frame <= 6 * FPS:
            pool.setdefault(r.spk_token, []).append(r)
    out, i = [], 0
    while i < len(rows):
        k = rng.randint(1, max_lines)
        seg, turns, total = [], [], 0
        for r in rows[i:i + k]:
            if not r.spk_token or r.spk_token.endswith("_x") or (seg and r.pos != prev.pos + 1):
                break
            n = r.end_frame - r.start_frame
            pause = pause_len(seg[-1]) if seg else 0
            if seg and total + pause + n > max_frames:
                break
            seg.append(r)
            turns.append((r.spk_token, r.text.strip()))
            total += pause + n
            prev = r
        if turns and total <= max_frames and rng.random() < keep:
            merged = splice(seg)
            first, last = rows[i], rows[i + len(seg) - 1]
            ctx = ((first.b2, first.b1), (last.a1, last.a2))
            if len({t for t, _ in turns}) == 1:
                out.append((tok, merged, " ".join(t for _, t in turns), first.lang, first.global_speaker,
                            ctx, first.spk_token, first.q_token))
            else:
                scene = None
                if rng.random() < prompted:
                    voices = list(dict.fromkeys(t for t, _ in turns))
                    used = {r.pos for r in seg}
                    picks = [[c for c in pool.get(v, []) if c.pos not in used] for v in voices]
                    if all(picks):
                        picks = [rng.choice(p) for p in picks]
                        plen = sum(int(p.end_frame - p.start_frame) for p in picks)
                        if total + plen <= max_frames:
                            name = {v: f"S{j + 1}" for j, v in enumerate(voices)}
                            ptext = " ".join(f"<|quad_start|>{name[v]}<|quad_end|>{p.text.strip()}" for v, p in zip(voices, picks))
                            stext = of.dialogue_text([(name[v], t) for v, t in turns])
                            psegs = tuple((int(p.start_frame), int(p.end_frame)) for p in picks)
                            scene = (tok, psegs + merged, f"{ptext} {stext}", first.lang, f"dialogue:{tok}", ctx,
                                     "dialogue_p", first.q_token, plen)
                out.append(scene or (tok, merged, of.dialogue_text(turns), first.lang, f"dialogue:{tok}", ctx,
                                     "dialogue", first.q_token))
        i += max(1, len(seg))
    return out


def span_frames(span):
    return sum(e - s for s, e in span[1])


def balance_speakers(spans, cap_hours, seed):
    """keep each span with probability min(1, cap_hours / its speaker's hours)"""
    if cap_hours <= 0:
        return spans
    hours = {}
    for x in spans:
        hours[x[4]] = hours.get(x[4], 0.0) + span_frames(x) / FPS / 3600
    rng = random.Random(seed)
    return [x for x in spans if hours[x[4]] <= cap_hours or rng.random() < cap_hours / hours[x[4]]]


class SpanDataset(IterableDataset):
    def __init__(self, spans, tokens_dir, tokenizer, args, train=True):
        self.spans, self.dir, self.tok, self.args, self.train = spans, tokens_dir, tokenizer, args, train
        self.mm = {}

    def tokens(self, name):
        if name not in self.mm:
            self.mm[name] = np.load(f"{self.dir}/tokens/{name}", mmap_mode="r")
        return self.mm[name]

    def process(self, span, rng):
        """-> ids [10, L], semantic labels, codes [10, L], audio mask, MTP target mask, MTP block, kind, block,
        positions; layout [style + text][prompt][clean target][noisy target], kind 0 / 0 / 1 / 2"""
        name, segments, text, lang, _, (before, after), spk_token, q_token, *fixed = span  # fixed: prompt frames of a "dialogue_p" scene
        if rng.random() < self.args.tag_drop:  # tags stay optional: the same text without them
            text = strip_tags(text)
        t = self.tokens(name)
        audio = torch.from_numpy(np.concatenate([np.asarray(t[:, s:e], dtype=np.int64) for s, e in segments], 1))  # [10, T]
        a, T, B, C = self.args, audio.shape[1], self.args.block_size, of.NUM_CODEBOOKS
        drop_cond = rng.random() < a.drop_cond_ratio and not fixed  # a prompted scene keeps its prompts
        # speaker: token + prompt / token only / prompt only; the per-language bag (_x) always with its prompt
        # dialogue scenes: voices from the turn markers only (instruct "dialogue", no prompt)
        if spk_token == "dialogue":
            mode, instruct = "token", "dialogue"
        elif spk_token == "dialogue_p":  # voices S1, S2, ... from their prompts
            mode, instruct = "fixed", "dialogue"
        elif not spk_token:  # speaker_x line (voice does not match its label): prompt only
            mode, instruct = "prompt", ""
        else:
            mode = "both" if spk_token.endswith("_x") else rng.choices(("both", "token", "prompt"), weights=a.speaker_mode)[0]
            instruct = f"speaker:{spk_token}"
        instruct = "" if mode == "prompt" else instruct
        # recording quality (one id per book / game+language), in --quality-ratio of the samples
        if q_token and rng.random() < a.quality_ratio:
            instruct = f"{instruct} quality:{q_token}".strip()
        instruct = "None" if drop_cond or not instruct else instruct
        plen = fixed[0] if fixed else 0 if drop_cond or mode == "token" else int(T * rng.uniform(*a.prompt_ratio))
        language = lang if (not drop_cond and rng.random() < a.language_ratio) else "None"
        # target: speech, END, PAD cycle to the block boundary
        n = T - plen
        nt = (n + 1 + B - 1) // B * B
        tgt = torch.tensor(of.PAD_IDS * ((nt - n - 1) // len(of.PAD_IDS) + 1))[:nt - n - 1]
        tgt = torch.cat([audio[0, plen:], torch.tensor([of.END_ID]), tgt])
        blk = torch.arange(nt) // B
        clean = torch.full((C, nt), of.MASK_ID)
        clean[0] = tgt
        g = torch.Generator().manual_seed(rng.randrange(2**31))
        ratio = torch.rand(nt // B, generator=g) * (a.mask_ratio[1] - a.mask_ratio[0]) + a.mask_ratio[0]
        m = torch.rand(nt, generator=g) < ratio[blk]
        if not m.any():
            m[torch.randint(nt, (1,), generator=g)] = True
        noisy = clean.clone()
        noisy[0][m] = of.MASK_ID
        if drop_cond:
            prefix = torch.zeros(C, 0, dtype=torch.long)
        else:
            style = self.tok(f"<|lang_start|>{language}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>",
                             return_tensors="pt").input_ids
            nb = rng.choices((0, 1, 2), weights=a.context_before)[0]
            na = rng.choices((0, 1, 2), weights=a.context_after)[0]
            txt = self.tok(of.context_text(text, [x for x in before[2 - nb:] if x], [x for x in after[:na] if x]),
                           return_tensors="pt").input_ids
            prefix = torch.cat([style, txt], 1).repeat(C, 1)
        S, ctx = prefix.shape[1], prefix.shape[1] + plen
        ids = torch.cat([prefix, audio[:, :plen], clean, noisy], 1)
        L = ids.shape[1]
        sem = torch.full((L,), -100)
        sem[ctx + nt:][m] = tgt[m]
        codes = torch.zeros(C, L, dtype=torch.long)
        codes[:, ctx:ctx + n] = audio[:, plen:]
        amask = torch.ones(L, dtype=torch.bool)
        amask[:S] = False
        tmask = torch.zeros(L, dtype=torch.bool)
        tmask[ctx:ctx + n] = True
        kind = torch.cat([torch.zeros(ctx, dtype=torch.long), torch.ones(nt, dtype=torch.long), torch.full((nt,), 2)])
        block = torch.cat([torch.zeros(ctx, dtype=torch.long), blk, blk])
        pos = torch.cat([torch.arange(ctx), ctx + torch.arange(nt), ctx + torch.arange(nt)])
        return ids, sem, codes, amask, tmask, block, kind, pos

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(1000 * wid + (os.getpid() if self.train else 0))
        order = list(range(wid, len(self.spans), nw))
        while True:
            if self.train:
                rng.shuffle(order)
            # token-budget batches from length-sorted buckets of 512 spans
            for b0 in range(0, len(order), 512):
                bucket = [self.process(self.spans[i], rng) for i in order[b0:b0 + 512]]
                bucket.sort(key=lambda x: x[0].shape[1])
                batch, longest = [], 0
                for item in bucket:
                    L = item[0].shape[1]
                    if batch and max(longest, L) * (len(batch) + 1) > self.args.batch_tokens:
                        yield collate(batch, self.tok.pad_token_id)
                        batch, longest = [], 0
                    batch.append(item)
                    longest = max(longest, L)
                if batch:
                    yield collate(batch, self.tok.pad_token_id)
            if not self.train:
                return


def collate(items, pad_id):
    L = max(i[0].shape[1] for i in items)
    pad = lambda x, v: torch.nn.functional.pad(x, (0, L - x.shape[-1]), value=v)
    ids = torch.stack([pad(i[0], pad_id) for i in items])
    sem = torch.stack([pad(i[1], -100) for i in items])
    codes = torch.stack([pad(i[2], 0) for i in items])
    am = torch.stack([pad(i[3], False) for i in items])
    tm = torch.stack([pad(i[4], False) for i in items])
    block = torch.stack([pad(i[5], 0) for i in items])
    kind = torch.stack([pad(i[6], -1) for i in items])
    pos = torch.stack([pad(i[7], 0) for i in items])
    return ids, am, of.block_attention(kind, block), pos, sem, codes, tm, block


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", default="/data/dataset/tokens_fish")
    ap.add_argument("--out", default="/data/dataset/train/ckpt/pilot")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-new", type=float, default=1e-3, help="lr for the new audio embeddings/heads")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--max-sentences", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=25.0)
    ap.add_argument("--max-pause", type=float, default=3.5, help="pause kept after each spliced sentence (s); a longer one loses its middle")
    ap.add_argument("--context-before", type=float, nargs=3, default=(0.3, 0.3, 0.4),
                    help="probabilities of 0 / 1 / 2 context sentences before the span")
    ap.add_argument("--context-after", type=float, nargs=3, default=(0.5, 0.25, 0.25),
                    help="probabilities of 0 / 1 / 2 context sentences after the span")
    ap.add_argument("--speaker-mode", type=float, nargs=3, default=(0.45, 0.3, 0.25),
                    help="weights of speaker token + prompt / token only / prompt only (speakers with a token)")
    ap.add_argument("--dialogue", type=float, default=0.5,
                    help="share of the scene windows used (multispeaker books; 0 = off)")
    ap.add_argument("--tag-oversample", type=int, default=3, help="spans with inline tags this many times")
    ap.add_argument("--tag-drop", type=float, default=0.25, help="share of the samples with their tags stripped")
    ap.add_argument("--dialogue-prompt", type=float, default=0.5,
                    help="share of the scenes with one prompt per voice (S1, S2, ...) instead of speaker tokens")
    ap.add_argument("--quality-ratio", type=float, default=0.7, help="share of samples that name their recording")
    ap.add_argument("--speaker-cap", type=float, default=30.0,
                    help="hours per global speaker per index round (0 = no balancing)")
    ap.add_argument("--max-skip", type=float, default=30.0, help="spliced sentences at most this far apart in the book (s)")
    ap.add_argument("--prompt-ratio", type=float, nargs=2, default=(0.0, 0.3))
    ap.add_argument("--mask-ratio", type=float, nargs=2, default=(0.0, 1.0))
    ap.add_argument("--drop-cond-ratio", type=float, default=0.1)
    ap.add_argument("--language-ratio", type=float, default=0.8)
    ap.add_argument("--val-books", type=int, default=40, help="1 in N books (by name hash) is validation")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-grad-ckpt", action="store_true", help="keep activations (faster, more VRAM)")
    ap.add_argument("--max-train-spans", type=int, default=0, help="debug: cap the number of training spans")
    ap.add_argument("--init", default="", help="start from these model weights (e.g. the pilot)")
    ap.add_argument("--resume-from", default="", help="no latest.pt in --out yet: resume model, optimizer and step "
                    "from this checkpoint (older vocab upgraded)")
    ap.add_argument("--block-size", type=int, default=16, help="frames per diffusion block (16 = 0.74 s)")
    ap.add_argument("--block-schedule", default="0:64,4000:32,8000:16",
                    help="step:size,... block size from each step on (OmniVoice is bidirectional: large blocks first); "
                    "empty = --block-size throughout. Sizes change when the index is re-read (--reload-every)")
    ap.add_argument("--reload-every", type=int, default=2000, help="re-read the token index (new books) every N steps")
    args = ap.parse_args()
    args.prompt_ratio, args.mask_ratio = tuple(args.prompt_ratio), tuple(args.mask_ratio)
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    model, tokenizer = of.OmniFish.from_omnivoice()
    if not args.no_grad_ckpt:
        model.llm.gradient_checkpointing_enable()
    model = model.cuda()
    is_new = lambda n: n.startswith(("audio_embeddings", "semantic_head", "mtp."))
    new = [p for n, p in model.named_parameters() if is_new(n)]
    old = [p for n, p in model.named_parameters() if not is_new(n)]
    opt = torch.optim.AdamW([{"params": old, "lr": args.lr}, {"params": new, "lr": args.lr_new}],
                            betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    base = [g["lr"] for g in opt.param_groups]
    step = 0
    latest = f"{args.out}/latest.pt"
    src = latest if os.path.exists(latest) else args.resume_from
    if src:  # resume: model, optimizer, step (weights / moments of an older vocab upgraded)
        ck = torch.load(src, map_location="cpu", weights_only=False)
        model.load_state_dict(of.OmniFish.upgrade_state(ck["model"]))
        names = [n for n, _ in model.named_parameters() if not is_new(n)] + [n for n, _ in model.named_parameters() if is_new(n)]
        for i, st in ck["opt"]["state"].items():
            for key in ("exp_avg", "exp_avg_sq"):
                st[key] = of.OmniFish.upgrade_moment(names[i], st[key])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed from {src} at step {step}", flush=True)
        del ck
    elif args.init:
        model.load_state_dict(of.OmniFish.upgrade_state(torch.load(args.init, map_location="cpu", weights_only=False)["model"]))
        print(f"initialised from {args.init}", flush=True)

    def set_lr(step):
        f = step / args.warmup if step < args.warmup else \
            0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * max(f, 0.02 if step >= args.warmup else f)

    def save(path):
        tmp = path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, tmp)
        os.replace(tmp, path)

    log = open(f"{args.out}/train.log", "a")
    max_frames, max_gap, max_skip = int(args.max_seconds * FPS), int(args.max_pause * FPS), int(args.max_skip * FPS)
    schedule = sorted((int(a), int(b)) for a, b in (x.split(":") for x in args.block_schedule.split(",") if x))
    fixed_block = args.block_size

    def block_size_at(step):
        return next((b for s0, b in reversed(schedule) if s0 <= step), fixed_block)

    model.train()
    while step < args.steps:
        # (re)read the token index: books encoded since the last round join the training set
        tr_idx, va_idx = load_index(args.tokens, args.val_books)
        # dialogue scenes only once the multispeaker lines are checked (speaker_mismatch.parquet)
        dlg = args.dialogue if os.path.exists("/data/dataset/build/speaker_mismatch.parquet") else 0.0
        tr = build_spans(tr_idx, args.max_sentences, max_frames, max_gap, max_skip, seed=step, dialogue=dlg,
                         dialogue_prompt=args.dialogue_prompt)
        va = build_spans(va_idx, args.max_sentences, max_frames, max_gap, max_skip, 1, dialogue=dlg,
                         dialogue_prompt=args.dialogue_prompt)
        all_hours = sum(span_frames(x) for x in tr) / FPS / 3600
        tr = balance_speakers(tr, args.speaker_cap, seed=step)
        if args.max_train_spans:
            tr = random.Random(0).sample(tr, min(args.max_train_spans, len(tr)))
        tagged = [x for x in tr if TAG_RE.search(x[2])]
        tr = tr + tagged * (args.tag_oversample - 1)  # spans with inline tags come up more often
        hours = sum(span_frames(x) for x in tr) / FPS / 3600
        args.block_size = block_size_at(step)
        msg = (f"step {step}: train spans {len(tr)} ({hours:.0f} h of {all_hours:.0f} h after the speaker cap, "
               f"{len(tagged)} with tags x{args.tag_oversample}) "
               f"from {tr_idx.tokens.nunique()} books, {tr_idx.global_speaker.nunique()} speakers | "
               f"val spans {len(va)} | block {args.block_size} frames")
        print(msg, flush=True)
        log.write(msg + "\n")
        dl = DataLoader(SpanDataset(tr, args.tokens, tokenizer, args), batch_size=None, num_workers=args.workers,
                        prefetch_factor=4)
        t0, seen = time.time(), 0
        for batch in dl:
            step += 1
            set_lr(step)
            batch = [x.cuda(non_blocking=True) for x in batch]
            with torch.autocast("cuda", torch.bfloat16):
                out = model(*batch)
            out["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            seen += int(batch[6].sum())  # target speech frames
            if step % 50 == 0:
                pc = " ".join(f"{x:.2f}" for x in out["residual"].tolist())
                msg = (f"step {step} loss {out['loss'].item():.3f} | sem {out['semantic'].item():.3f} | res {pc} | "
                       f"gn {gn:.2f} | lr {opt.param_groups[0]['lr']:.2e} | "
                       f"{seen / FPS / 3600 / ((time.time() - t0) / 3600):.0f}x audio/h | "
                       f"GPU {torch.cuda.max_memory_allocated() / 1e9:.0f} GB")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if step % args.val_every == 0 and va:
                model.eval()
                losses = []
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    for n, vb in enumerate(DataLoader(
                            SpanDataset(va, args.tokens, tokenizer, args, train=False), batch_size=None)):
                        vo = model(*(x.cuda() for x in vb))
                        losses.append((vo["loss"].item(), vo["semantic"].item()))
                        if n >= 50:
                            break
                msg = f"VAL step {step} loss {np.mean([l for l, _ in losses]):.3f} sem {np.mean([x for _, x in losses]):.3f}"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                model.train()
            if step % args.save_every == 0 or step >= args.steps:
                save(latest)
                torch.save({"model": model.state_dict(), "step": step}, f"{args.out}/step{step}.pt")
            if step >= args.steps or step % args.reload_every == 0:
                break
        del dl


if __name__ == "__main__":
    main()
