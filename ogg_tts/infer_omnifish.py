#!/usr/bin/env python3
"""OmniFish inference: voice-cloning TTS by block diffusion (see omnifish.py).

Input sequence: [lang/instruct style] [prompt text + target text] [prompt frames (10 codes)] [generated
blocks] [current block (MASK)]. Blocks of --block frames are produced left to right, each unmasked over
--steps iterations (most confident first, cosine schedule, classifier-free guidance against the sequence
without text and prompt); generation ends with the block holding END, so the length is the model's own.
Then the MTP decoder fills codebooks 1..9 and the Fish codec renders 44.1 kHz audio.

  infer_omnifish.py --ckpt CKPT --prompt-tokens FILE.npy:START:END --prompt-text "..." --text "..." --lang pl --out out.wav
  infer_omnifish.py --ckpt CKPT --demo DIR    (prompts and texts taken from held-out encoded books)
"""
import argparse
import math
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fish_codec as fc  # noqa: E402
import omnifish as of  # noqa: E402

FPS = 44100 / 2048


def prefix_ids(tok, text, lang, before=(), after=(), speaker=None, quality=None):
    """before / after: up to 2 sentences of text context each (not spoken); speaker: token name ("pl_3"), or the
    language bag ("pl_x", only with a prompt), or "dialogue" (text from of.dialogue_text, no prompt), or None"""
    C = of.NUM_CODEBOOKS
    instruct = "dialogue" if speaker == "dialogue" else f"speaker:{speaker}" if speaker else ""
    instruct = f"{instruct} quality:{quality}".strip() if quality else instruct  # recording id, e.g. q35
    instruct = instruct or "None"
    style = tok(f"<|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>", return_tensors="pt").input_ids
    txt = tok(of.context_text(text, list(before), list(after)), return_tensors="pt").input_ids
    return torch.cat([style, txt], 1).repeat(C, 1)


def sequence(prefix, prompt, clean, noisy, block):
    """[text][prompt (10 codes)][clean blocks (semantic)][noisy block] -> ids [1, 10, L], audio mask, attention,
    positions, slice of the clean frames, slice of the noisy frames"""
    C, dev = of.NUM_CODEBOOKS, clean.device
    S, P, nc, nn_ = prefix.shape[1], prompt.shape[1], clean.shape[0], noisy.shape[0]
    tgt = torch.full((C, nc + nn_), of.MASK_ID, dtype=torch.long, device=dev)
    tgt[0] = torch.cat([clean, noisy])
    ids = torch.cat([prefix.to(dev), prompt.to(dev), tgt], 1)[None]
    L, ctx = ids.shape[-1], S + P
    am = torch.ones(1, L, dtype=torch.bool, device=dev)
    am[:, :S] = False
    kind = torch.cat([torch.zeros(ctx), torch.ones(nc), torch.full((nn_,), 2)]).long().to(dev)[None]
    blk = torch.cat([torch.zeros(ctx), torch.arange(nc) // block, torch.full((nn_,), nc // block)]).long().to(dev)[None]
    pos = torch.arange(L, device=dev)[None]
    return ids, am, of.block_attention(kind, blk), pos, slice(ctx, ctx + nc), slice(ctx + nc, L)


@torch.no_grad()
def semantic_blocks(model, tok, prompt_codes, prompt_text, text, lang, block=16, steps=8, cfg=2.0, temperature=0.8,
                    max_frames=None, seed=0, before=(), after=(), speaker=None, quality=None):
    """block diffusion: blocks of `block` frames left to right, each unmasked in `steps` iterations (most
    confident first, cosine schedule, CFG against the sequence without text and prompt). Yields (prefix ids,
    semantic ids of every frame so far) after each block; the last block holds END (or a PAD), or the safety
    cap is reached."""
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    full_text = f"{prompt_text} {text}".strip()
    if prompt_codes is None:  # speaker token only, no voice prompt
        prompt_codes, prompt_text = torch.zeros(of.NUM_CODEBOOKS, 0, dtype=torch.long), ""
    if max_frames is None:  # safety cap only: twice the prompt's speaking rate (~1.6 frames/char without one)
        rate = prompt_codes.shape[1] / len(prompt_text) if prompt_text else 1.6
        max_frames = int(2 * rate * len(text)) + 2 * block
    prefix = prefix_ids(tok, full_text, lang, before, after, speaker, quality)
    none = torch.zeros(of.NUM_CODEBOOKS, 0, dtype=torch.long)
    clean = torch.zeros(0, dtype=torch.long, device=dev)

    def logits_of(pre, pro, cur):
        ids, am, att, pos, _, ns = sequence(pre, pro, clean, cur, block)
        with torch.autocast(dev.type, torch.bfloat16):
            h = model.backbone(ids, am, att, pos)
            return model.semantic_head(h[:, ns]).float()[0]

    while clean.shape[0] < max_frames:
        cur = torch.full((block,), of.MASK_ID, dtype=torch.long, device=dev)
        for s in range(steps):
            logits = logits_of(prefix, prompt_codes, cur)
            if cfg > 0:
                logits = logits + cfg * (logits - logits_of(none, none, cur))
            logits[:, of.MASK_ID] = -float("inf")
            masked = cur == of.MASK_ID
            probs = F.softmax(logits / max(temperature, 1e-5), -1)
            sample = torch.multinomial(probs, 1, generator=g)[:, 0] if temperature > 0 else logits.argmax(-1)
            conf = probs.gather(1, sample[:, None])[:, 0]
            conf = torch.where(masked, conf, torch.full_like(conf, -1.0))
            keep_masked = int(math.floor(block * math.cos(math.pi / 2 * (s + 1) / steps)))
            n_fix = int(masked.sum()) - keep_masked
            if n_fix > 0:
                idx = conf.topk(n_fix).indices
                cur[idx] = sample[idx]
        clean = torch.cat([clean, cur])
        yield prefix, clean
        if (cur >= of.END_ID).any():
            return


def speech_frames(clean):
    end = (clean >= of.END_ID).nonzero()
    return int(end[0]) if len(end) else clean.shape[0]


@torch.no_grad()
def residuals(model, prefix, prompt_codes, clean, n, block, mtp_temperature=0.0):
    """codes [10, n] of the first n frames: the clean pass over every generated frame, then the MTP
    (block-causal over frames, so frames of finished blocks never change as more blocks come)"""
    dev = clean.device
    if n == 0:
        return torch.zeros(of.NUM_CODEBOOKS, 0, dtype=torch.long, device=dev)
    ids, am, att, pos, cs, _ = sequence(prefix, prompt_codes, clean, clean[:0], block)
    with torch.autocast(dev.type, torch.bfloat16):
        h = model.backbone(ids, am, att, pos)
    ht = h[:, cs][:, :n].float()
    fpos = torch.arange(n, device=dev)[None]
    valid = torch.ones(1, n, dtype=torch.bool, device=dev)
    attn = model.frame_attention(fpos // block, valid)
    with torch.autocast(dev.type, torch.bfloat16):
        return model.mtp.generate(ht, clean[None, :n], fpos, valid, temperature=mtp_temperature, attn=attn)[0]


@torch.no_grad()
def generate(model, tok, prompt_codes, prompt_text, text, lang, block=16, steps=8, cfg=2.0, temperature=0.8,
             mtp_temperature=0.0, max_frames=None, seed=0, before=(), after=(), speaker=None, quality=None):
    """whole utterance -> codes [10, frames]; the speech is everything before END. before / after: text context;
    speaker: token name; prompt_codes may be None with a speaker token"""
    for prefix, clean in semantic_blocks(model, tok, prompt_codes, prompt_text, text, lang, block, steps, cfg,
                                         temperature, max_frames, seed, before, after, speaker, quality):
        pass
    if prompt_codes is None:
        prompt_codes = torch.zeros(of.NUM_CODEBOOKS, 0, dtype=torch.long)
    return residuals(model, prefix, prompt_codes, clean, speech_frames(clean), block, mtp_temperature)


@torch.no_grad()
def stream(model, tok, codec, prompt_codes, prompt_text, text, lang, block=16, steps=8, cfg=2.0, temperature=0.8,
           mtp_temperature=0.0, max_frames=None, seed=0, chunks=(1, 2, 4), before=(), after=(), speaker=None, quality=None):
    """streaming: yields 44.1 kHz audio chunks (numpy). A chunk is emitted after chunks[0] blocks, then every
    chunks[1], ... blocks (the last size repeats), and at END. The codec is causal but has a long memory, so each
    chunk is decoded from the start of the utterance (codes of emitted frames frozen) and only its new samples
    are emitted: the joined chunks equal decoding the streamed codes at once, seamless at every join."""
    if prompt_codes is None:
        prompt_codes, prompt_text = torch.zeros(of.NUM_CODEBOOKS, 0, dtype=torch.long), ""
    sizes, emitted, since = list(chunks), 0, 0
    done = None  # codes of the frames already emitted: frozen, so a later MTP pass (numerics) cannot change them

    def new_audio(prefix, clean):
        nonlocal emitted, done
        codes = residuals(model, prefix, prompt_codes, clean, speech_frames(clean), block, mtp_temperature)
        if done is not None:
            codes = torch.cat([done, codes[:, done.shape[1]:]], 1)
        done = codes
        wav = fc.decode(codec, codes[None])[0].float().cpu().numpy() if codes.shape[1] else np.zeros(0, np.float32)
        out, emitted = wav[emitted:], max(emitted, len(wav))
        return out

    for prefix, clean in semantic_blocks(model, tok, prompt_codes, prompt_text, text, lang, block, steps, cfg,
                                         temperature, max_frames, seed, before, after, speaker, quality):
        since += 1
        if since < sizes[0] and speech_frames(clean) == clean.shape[0]:  # chunk not full and no END yet
            continue
        since = 0
        if len(sizes) > 1:
            sizes.pop(0)
        chunk = new_audio(prefix, clean)
        if len(chunk):
            yield chunk
    if since:  # safety cap reached inside a chunk
        chunk = new_audio(prefix, clean)
        if len(chunk):
            yield chunk


def load_model(ckpt, device="cuda"):
    model, tok = of.OmniFish.from_omnivoice()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state)
    return model.to(device).eval(), tok


def demo(model, tok, codec, out_dir, n=4, seed=0, sentences=1, gen_kw=None):
    """prompts and targets from held-out books of the token store: real audio of the target is decoded too"""
    import glob

    import pandas as pd
    idx = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/data/dataset/tokens_fish/index/*.parquet"))[:200]])
    idx = idx[(idx.end_frame - idx.start_frame).between(int(3 * FPS), int(8 * FPS))]
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    books = idx.groupby("tokens")
    names = rng.choice(list(books.groups), size=min(n, books.ngroups), replace=False)
    iso = {"english": "en", "polish": "pl", "german": "de", "french": "fr", "russian": "ru", "italian": "it",
           "spanish_es": "es", "spanish_latam": "es", "portuguese_br": "pt", "portuguese_pt": "pt"}
    for k, name in enumerate(names):
        g = books.get_group(name).sample(2, random_state=k)
        p, t = g.iloc[0], g.iloc[1]
        toks = np.load(f"/data/dataset/tokens_fish/tokens/{name}", mmap_mode="r")
        prompt = torch.from_numpy(np.asarray(toks[:, p.start_frame:p.end_frame], dtype=np.int64))
        end = t.end_frame
        if sentences > 1:  # target = this sentence + the following ones of the same speaker, as training spans
            nxt = books.get_group(name).sort_values("order")
            nxt = nxt[(nxt.order > t.order) & (nxt.speaker == t.speaker)].head(sentences - 1)
            nxt = nxt[nxt.end_frame - t.start_frame <= int(25 * FPS)]
            if len(nxt):
                end = int(nxt.end_frame.iloc[-1])
                t = t.copy()
                t["text"] = " ".join([t.text.strip()] + [x.strip() for x in nxt.text])
        real = torch.from_numpy(np.asarray(toks[:, t.start_frame:end], dtype=np.int64))
        lang = iso.get(p.language, p.language)
        codes = generate(model, tok, prompt, p.text, t.text, lang, seed=k, **(gen_kw or {}))
        for tag, c in (("prompt", prompt), ("real", real), ("generated", codes.cpu())):
            wav = fc.decode(codec, c[None].to(next(codec.parameters()).device))[0].float().cpu().numpy()
            sf.write(f"{out_dir}/{k}_{lang}_{tag}.wav", wav, 44100)
        print(f"{k}: [{lang}] prompt: {p.text[:60]!r}\n   target (real {real.shape[1] / FPS:.1f} s, "
              f"generated {codes.shape[1] / FPS:.1f} s): {t.text!r}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--demo", help="output dir for held-out demo samples")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--prompt-tokens", help="tokens .npy:start_frame:end_frame")
    ap.add_argument("--prompt-text", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--lang", default="None")
    ap.add_argument("--speaker", default=None, help="speaker token, e.g. pl_3 (speaker_tokens.json); without "
                    "--prompt-tokens the voice comes from the token alone")
    ap.add_argument("--quality", default=None, help="recording id, e.g. q35 (quality_tokens.json)")
    ap.add_argument("--before", nargs="*", default=[], help="up to 2 sentences of text context before (not spoken)")
    ap.add_argument("--after", nargs="*", default=[], help="up to 2 sentences of text context after (not spoken)")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--steps", type=int, default=8, help="unmasking iterations per block")
    ap.add_argument("--block", type=int, default=16, help="frames per block (as trained)")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sentences", type=int, default=1, help="demo: consecutive sentences per target (<= 25 s)")
    args = ap.parse_args()
    model, tok = load_model(args.ckpt, args.device)
    codec = fc.load_codec(args.device)
    if args.demo:
        demo(model, tok, codec, args.demo, args.n, sentences=args.sentences,
             gen_kw=dict(block=args.block, steps=args.steps, cfg=args.cfg, temperature=args.temperature))
        return
    prompt = None
    if args.prompt_tokens:
        path, s, e = args.prompt_tokens.rsplit(":", 2)
        prompt = torch.from_numpy(np.asarray(np.load(path, mmap_mode="r")[:, int(s):int(e)], dtype=np.int64))
    codes = generate(model, tok, prompt, args.prompt_text, args.text, args.lang, block=args.block, steps=args.steps,
                     cfg=args.cfg, temperature=args.temperature, before=args.before, after=args.after,
                     speaker=args.speaker, quality=args.quality)
    wav = fc.decode(codec, codes[None].to(args.device))[0].float().cpu().numpy()
    sf.write(args.out, wav, 44100)
    print(f"saved {args.out} ({len(wav) / 44100:.1f} s)")


if __name__ == "__main__":
    main()
