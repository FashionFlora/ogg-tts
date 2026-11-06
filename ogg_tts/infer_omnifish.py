#!/usr/bin/env python3
"""OmniFish inference: voice-cloning TTS by block diffusion (see omnifish.py).

Input sequence: [lang/instruct style] [prompt text + target text] [prompt frames (10 codes)] [generated
blocks] [current block (MASK)]. Blocks of --block frames are produced left to right, each unmasked over
--steps iterations (most confident first, cosine schedule, classifier-free guidance against the sequence
without text and prompt); generation ends with the block holding END, so the length is the model's own.
Then the MTP decoder fills codebooks 1..9 and the Fish codec renders 44.1 kHz audio.

  infer_omnifish.py --ckpt CKPT --prompt-tokens FILE.npy:START:END --prompt-text "..." --text "..." --lang pl --out out.wav
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



def voice_context(tok, voices, text, lang, before=(), after=(), quality=None):
    """dialogue with defined voices: voices [(name, transcript, codes [10, T]), ...], text from
    of.dialogue_text([(name, line), ...]) -> context (ids [10, L], audio mask [L]) laid out as in training:
    [style][S1: transcript][S1 audio][S2: transcript][S2 audio]...[scene text]"""
    C = of.NUM_CODEBOOKS
    instruct = f"dialogue quality:{quality}" if quality else "dialogue"
    parts = [(tok(f"<|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>", return_tensors="pt").input_ids.repeat(C, 1), False)]
    for name, transcript, codes in voices:
        parts += [(tok(f"<|quad_start|>{name}<|quad_end|>{transcript.strip()}", return_tensors="pt").input_ids.repeat(C, 1), False),
                  (codes.cpu().long(), True)]
    parts.append((tok(of.context_text(text, list(before), list(after)), return_tensors="pt").input_ids.repeat(C, 1), False))
    return (torch.cat([x for x, _ in parts], 1),
            torch.cat([torch.full((x.shape[1],), f, dtype=torch.bool) for x, f in parts]))


class _Stream:
    """KV-cached sequences, one batch row per context ((ids [10, L], audio mask [L]): the conditional one and the empty CFG one), left-padded to a
    common length: the context (text + prompt) and the finished clean blocks are computed once; the current block
    is a query of `block` frames against the cache. Every allowed key of a block query is in the cache or in the
    block itself (clean/noisy block b sees context, clean blocks < b and itself), so a row only masks its padding.
    One forward per unmasking iteration serves both CFG branches."""

    def __init__(self, model, contexts, dev):
        from transformers import DynamicCache
        self.model, self.dev, self.cache, self.done = model, dev, DynamicCache(), 0
        C, B = of.NUM_CODEBOOKS, len(contexts)
        self.ctx = [c.shape[1] for c, _ in contexts]
        T = max(self.ctx)
        self.pad = [T - c for c in self.ctx]
        if T == 0:
            return
        ids = torch.zeros(B, C, T, dtype=torch.long, device=dev)
        am = torch.zeros(B, T, dtype=torch.bool, device=dev)
        mask = torch.zeros(B, 1, T, T, dtype=torch.bool, device=dev)
        pos = torch.zeros(B, T, dtype=torch.long, device=dev)
        for r, (c, a) in enumerate(contexts):
            k = self.pad[r]
            ids[r, :, k:] = c.to(dev)
            am[r, k:] = a.to(dev)
            mask[r, 0, k:, k:] = True  # context sees context
            mask[r, 0, :k, :k] = torch.eye(k, dtype=torch.bool, device=dev)  # padding rows: themselves only
            pos[r, k:] = torch.arange(self.ctx[r], device=dev)
        self.ctx_h = self._run(ids, am, mask, pos)  # context states (the MTP reads the prompt frames' ones)

    def _run(self, ids, am, mask, pos):
        with torch.autocast(self.dev.type, torch.bfloat16):
            return self.model.llm(inputs_embeds=self.model.embed(ids, am), attention_mask=mask, position_ids=pos,
                                  past_key_values=self.cache, use_cache=True, return_dict=True)[0]

    def _block(self, semantic, full=None):
        B, L, past = len(self.ctx), semantic.shape[0], self.cache.get_seq_length()
        ids = torch.full((B, of.NUM_CODEBOOKS, L), of.MASK_ID, dtype=torch.long, device=self.dev)
        ids[:, 0] = semantic
        if full is not None:  # a decided block with its residual codes (MASK where END / PAD)
            ids[:] = full
        am = torch.ones(B, L, dtype=torch.bool, device=self.dev)
        mask = torch.ones(B, 1, L, past + L, dtype=torch.bool, device=self.dev)
        pos = torch.empty(B, L, dtype=torch.long, device=self.dev)
        for r in range(B):
            mask[r, 0, :, :self.pad[r]] = False
            pos[r] = self.ctx[r] + self.done + torch.arange(L, device=self.dev)
        return ids, am, mask, pos

    def logits(self, cur):
        """semantic logits [rows, frames, vocab] of the noisy block `cur`; the cache is left as it was"""
        h = self._run(*self._block(cur))
        self.cache.crop(-cur.shape[0])  # drop the noisy block again
        head = self.model.semantic_head
        return head(h.to(head.weight.dtype)).float()

    def peek(self, cur):
        """row 0's hidden states of the decided block with semantic ids only (the MTP input); cache unchanged"""
        h = self._run(*self._block(cur))[0]
        self.cache.crop(-cur.shape[0])
        return h

    def commit(self, full):
        """append the decided block with all 10 codes [10, block] as context of the later blocks"""
        self._run(*self._block(full[0], full))
        self.done += full.shape[1]


@torch.no_grad()
def blocks_cached(model, tok, prompt_codes, prompt_text, text, lang, block=16, steps=6, cfg=1.5, temperature=0.8,
                  max_frames=None, seed=0, before=(), after=(), speaker=None, quality=None, mtp_temperature=0.0,
                  mtp_top_k=0, mtp_rounds=4, full_context=True, voices=None, min_frames=0, semantic_left=(), mtp_prompt=True,
                  target_prefix=None):
    """block diffusion with KV caches (one batch row for the conditional and one for the CFG branch). Per block:
    unmask its semantic ids in `steps` iterations, run the MTP on it (semantic-only states), append it with all
    10 codes (full_context, for models trained so) or its semantic ids only (default) to the cache as context of the
    later blocks. Yields (semantic ids of every frame so far, codes
    [10, frames]) after each block; the last block holds END (or a PAD), or the safety cap is reached."""
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    if mtp_temperature > 0:
        torch.manual_seed(seed)
    if voices:  # a dialogue with defined voices (name, transcript, codes): prompt_codes / prompt_text unused
        frames, chars = sum(v[2].shape[1] for v in voices), sum(len(v[1]) for v in voices)
        context = voice_context(tok, voices, text, lang, before, after, quality)
    else:
        if prompt_codes is None:
            prompt_codes, prompt_text = torch.zeros(of.NUM_CODEBOOKS, 0, dtype=torch.long), ""
        frames, chars = prompt_codes.shape[1], len(prompt_text)
        if semantic_left:
            instruct = f"speaker:{speaker}" if speaker else ""
            instruct = f"{instruct} quality:{quality}".strip() if quality else instruct
            instruct = instruct or "None"
            style = tok(f"<|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>",
                        return_tensors="pt").input_ids.repeat(of.NUM_CODEBOOKS, 1)
            parts, audio_parts = [style], [torch.zeros(style.shape[1], dtype=torch.bool)]
            # as in training: the left lines as text only, in one <|object_ref|> span (their sound comes with the
            # prompt); entries are texts or (text, codes) pairs, codes unused; as many lines as the model was trained with
            left_text = " ".join(x if isinstance(x, str) else x[0] for x in semantic_left)
            left_ids = tok(f"<|object_ref_start|>{left_text}<|object_ref_end|>",
                           return_tensors="pt").input_ids.repeat(of.NUM_CODEBOOKS, 1)
            parts.append(left_ids)
            audio_parts.append(torch.zeros(left_ids.shape[1], dtype=torch.bool))
            # the target text starts with the prompt's transcript (training: "prompt text target text"); no text-only
            # `before` (training has none next to the left context)
            target_ids = tok(of.context_text(f"{prompt_text} {text}".strip(), (), list(after)),
                             return_tensors="pt").input_ids.repeat(of.NUM_CODEBOOKS, 1)
            parts.append(target_ids)
            audio_parts.append(torch.zeros(target_ids.shape[1], dtype=torch.bool))
            if prompt_codes.shape[1]:
                parts.append(prompt_codes.cpu().long())
                audio_parts.append(torch.ones(prompt_codes.shape[1], dtype=torch.bool))
            context = (torch.cat(parts, 1), torch.cat(audio_parts))
        else:
            full_text = f"{prompt_text} {text}".strip()
            prefix = prefix_ids(tok, full_text, lang, before, after, speaker, quality)
            context = (torch.cat([prefix, prompt_codes.cpu().long()], 1),
                       torch.cat([torch.zeros(prefix.shape[1], dtype=torch.bool), torch.ones(prompt_codes.shape[1], dtype=torch.bool)]))
    if max_frames is None:
        rate = frames / chars if chars else 1.6
        max_frames = int(2 * rate * len(text)) + 2 * block
    none = (torch.zeros(of.NUM_CODEBOOKS, 0, dtype=torch.long), torch.zeros(0, dtype=torch.bool))
    st = _Stream(model, [context] + ([none] if cfg > 0 else []), dev)
    mtp_dtype = next(model.mtp.parameters()).dtype
    # the MTP sees the prompt (all 10 codes) as decided frames before the target, as in training
    P = prompt_codes.shape[1] if (prompt_codes is not None and not voices and mtp_prompt) else 0
    S = model.mtp.new_state(P + max_frames + block, model.llm.config.hidden_size, dev, mtp_dtype)
    if P:
        S["codes"][:, :P] = prompt_codes.to(dev).long()
        S["states"][0][0, :P] = st.ctx_h[0, -P:].to(mtp_dtype)
        with torch.autocast(dev.type, torch.bfloat16):
            model.mtp.prime(S, P)
    clean = torch.zeros(0, dtype=torch.long, device=dev)
    if target_prefix is not None:  # continuation: earlier audio [10, n·block] of the target as decided blocks
        tp = target_prefix.to(dev).long()
        for lo in range(0, tp.shape[1] - tp.shape[1] % block, block):
            full = tp[:, lo:lo + block]
            h = st.peek(full[0])
            S["codes"][:, P + lo:P + lo + block], S["states"][0][0, P + lo:P + lo + block] = full, h.to(mtp_dtype)
            st.commit(full)
            clean = torch.cat([clean, full[0]])
    while clean.shape[0] < max_frames:
        cur = torch.full((block,), of.MASK_ID, dtype=torch.long, device=dev)
        for s in range(steps):
            lg = st.logits(cur)
            logits = lg[0] + cfg * (lg[0] - lg[1]) if cfg > 0 else lg[0]
            logits[:, of.MASK_ID] = -float("inf")
            if min_frames:
                local = torch.arange(block, device=dev) + clean.shape[0]
                logits[local < min_frames, of.END_ID:] = -float("inf")
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
        lo = clean.shape[0]
        end = (cur >= of.END_ID).nonzero()
        nb = int(end[0]) if len(end) else block  # speech frames of this block
        full = torch.full((of.NUM_CODEBOOKS, block), of.MASK_ID, dtype=torch.long, device=dev)
        full[0] = cur
        if nb:
            h = st.peek(cur)
            S["codes"][0, P + lo:P + lo + nb], S["states"][0][0, P + lo:P + lo + nb] = cur[:nb], h[:nb].to(mtp_dtype)
            with torch.autocast(dev.type, torch.bfloat16):
                model.mtp.decode_block(S, P + lo, P + lo + nb, block, mtp_rounds, mtp_temperature, mtp_top_k, prefix=P)
            full[1:, :nb] = S["codes"][1:, P + lo:P + lo + nb]
        clean = torch.cat([clean, cur])
        yield clean, S["codes"][:, P:P + lo + nb]
        if len(end):
            return
        st.commit(full if full_context else full[:1].expand(of.NUM_CODEBOOKS, -1).clone().masked_fill(
            torch.arange(of.NUM_CODEBOOKS, device=dev)[:, None] > 0, of.MASK_ID))  # semantic-only context


@torch.no_grad()
def generate_cached(model, tok, prompt_codes, prompt_text, text, lang, block=16, steps=6, cfg=1.5, temperature=0.8,
                    mtp_temperature=0.0, mtp_top_k=0, mtp_rounds=4, max_frames=None, seed=0, before=(), after=(),
                    speaker=None, quality=None, full_context=True, voices=None, min_frames=0, semantic_left=(), mtp_prompt=True,
                    target_prefix=None):
    """whole utterance -> codes [10, speech frames]; voices [(name, transcript, codes), ...]: a dialogue with
    defined voices (text from of.dialogue_text with those names)"""
    for clean, codes in blocks_cached(model, tok, prompt_codes, prompt_text, text, lang, block, steps, cfg,
                                      temperature, max_frames, seed, before, after, speaker, quality,
                                      mtp_temperature, mtp_top_k, mtp_rounds, full_context, voices, min_frames, semantic_left, mtp_prompt,
                                      target_prefix):
        pass
    return codes[:, :speech_frames(clean)].clone()


generate = generate_cached


def speech_frames(clean):
    end = (clean >= of.END_ID).nonzero()
    return int(end[0]) if len(end) else clean.shape[0]


@torch.no_grad()
def stream(model, tok, codec, prompt_codes, prompt_text, text, lang, block=16, steps=6, cfg=1.5, temperature=0.8,
           mtp_temperature=0.0, max_frames=None, seed=0, chunks=(1, 2, 4), before=(), after=(), speaker=None, quality=None,
           voices=None):
    """streaming: yields 44.1 kHz audio chunks (numpy). A chunk is emitted after chunks[0] blocks, then every
    chunks[1], ... blocks (the last size repeats), and at END. The codec is causal but has a long memory, so each
    chunk is decoded from the start of the utterance and only its new samples are emitted: the joined chunks equal
    decoding the codes at once, seamless at every join. Decided blocks never change."""
    sizes, emitted, since = list(chunks), 0, 0
    for clean, codes in blocks_cached(model, tok, prompt_codes, prompt_text, text, lang, block, steps, cfg,
                                      temperature, max_frames, seed, before, after, speaker, quality, mtp_temperature,
                                      voices=voices):
        since += 1
        last = bool((clean[-block:] >= of.END_ID).any())
        if since < sizes[0] and not last:
            continue
        since = 0
        if len(sizes) > 1:
            sizes.pop(0)
        n = speech_frames(clean)
        wav = fc.decode(codec, codes[None, :, :n])[0].float().cpu().numpy() if n else np.zeros(0, np.float32)
        if len(wav) > emitted:
            yield wav[emitted:]
            emitted = len(wav)


def load_model(ckpt, device="cuda"):
    """-> model, tokenizer: checkpoints with the ID table get the ID tokenizer (speaker / recording IDs as single
    tokens), older ones the plain one (IDs as text, as they were trained)"""
    model, tok = of.OmniFish.from_omnivoice()
    if str(ckpt).endswith(".safetensors"):  # released weights (huggingface.co/FashionFlora/ogg-tts)
        from safetensors.torch import load_file
        state = load_file(ckpt)
    else:
        state = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)["model"]
    fresh = model.load_weights(state, tok)
    return model.to(device).eval(), (tok if fresh else of.IdTokenizer(tok, model.text_vocab))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt-tokens", help="tokens .npy:start_frame:end_frame")
    ap.add_argument("--prompt-text", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--lang", default="None")
    ap.add_argument("--speaker", default=None, help="speaker token, e.g. pl_3 (speaker_tokens.json); without "
                    "--prompt-tokens the voice comes from the token alone")
    ap.add_argument("--quality", default=None, help="recording id, e.g. q35 (quality_tokens.json)")
    ap.add_argument("--before", nargs="*", default=[], help="preceding lines as text context (not spoken), as trained")
    ap.add_argument("--after", nargs="*", default=[], help="up to 2 sentences of text context after (not spoken)")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--steps", type=int, default=6, help="unmasking iterations per block")
    ap.add_argument("--block", type=int, default=16, help="frames per block (as trained)")
    ap.add_argument("--cfg", type=float, default=1.5, help="guidance; higher = faster, more pronounced delivery")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--mtp-temperature", type=float, default=0.0, help="sampling temperature of the acoustic codes (0 = argmax)")
    ap.add_argument("--mtp-top-k", type=int, default=0, help="acoustic codes sampled among the k most likely (0 = all)")
    ap.add_argument("--mtp-rounds", type=int, default=4, help="unmasking rounds per codebook and block")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model, tok = load_model(args.ckpt, args.device)
    codec = fc.load_codec(args.device)
    prompt = None
    if args.prompt_tokens:
        path, s, e = args.prompt_tokens.rsplit(":", 2)
        prompt = torch.from_numpy(np.asarray(np.load(path, mmap_mode="r")[:, int(s):int(e)], dtype=np.int64))
    codes = generate_cached(model, tok, prompt, args.prompt_text, args.text, args.lang, block=args.block,
                            steps=args.steps, cfg=args.cfg, temperature=args.temperature,
                            mtp_temperature=args.mtp_temperature, mtp_top_k=args.mtp_top_k, mtp_rounds=args.mtp_rounds,
                            semantic_left=tuple(args.before), after=args.after, speaker=args.speaker,
                            quality=args.quality)
    wav = fc.decode(codec, codes[None].to(args.device))[0].float().cpu().numpy()
    sf.write(args.out, wav, 44100)
    print(f"saved {args.out} ({len(wav) / 44100:.1f} s)")


if __name__ == "__main__":
    main()
