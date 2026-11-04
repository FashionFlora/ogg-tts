"""OmniFish v2: OmniVoice (k2-fsa, Qwen3-0.6B, masked discrete diffusion, bidirectional) predicts the
Fish Audio Drama 3 / S2 codec's semantic codebook; a small codebook-MTP decoder predicts the 9 residual
codebooks, chained like Drama 3's audio decoder (layer k sees layer k-1's state and code k, predicts
code k+1), but ~170M params, trained from scratch, attending over all frames in both directions
(non-autoregressive: at inference every layer runs on all frames at once).

Block diffusion (as LLaDA2.0 / BD3-LM): the target is generated in blocks of frames, left to right; inside
a block, masked iterative unmasking. Training sequence:
  [<|lang_start|>pl<|lang_end|><|instruct_start|>None<|instruct_end|>] [<|text_start|>text<|text_end|>]
  [prompt frames] [clean target blocks, semantic ids] [clean target blocks, all 10 codes] [noisy target blocks]
attention (see block_attention): context (text + prompt) sees only context; a full block (10 codes) sees context
and full blocks <= b; a semantic-only block and a noisy block b see context, full blocks < b and themselves. The
three copies share positions. So the backbone predicts a block from the full codes (timbre included) of the
blocks before it, and the MTP reads the semantic-only copy. Length is the model's own call (Rainbow Padding): the speech is followed by one
END frame and the PAD_IDS cycle up to the block boundary; generation stops at the block holding END.
Audio frame inputs (all 10 codebook embeddings summed): prompt frames and full target frames carry their 10
codes; semantic-only and noisy target frames carry only the semantic id (or MASK_ID) and MASK_ID for codebooks
1..9 - the same at training and inference. Losses: semantic CE on masked noisy frames (END/PAD included) + residual CE of the MTP, trained as masked
prediction over frames (see mtp_unmask_loss): it decodes a block codebook by codebook, unmasking the frames of
each codebook in a few rounds while seeing every decided code of the earlier blocks.
"""
import glob
import json
import math
import os
import re

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

# OmniVoice weights: $OMNIVOICE_PATH, else the local HF cache of the training machine
OMNIVOICE_PATH = os.environ.get("OMNIVOICE_PATH") or next(iter(glob.glob("/data/dataset/train/hf/models--k2-fsa--OmniVoice/snapshots/*")), None)
CODEBOOK_SIZES = [4096] + [1024] * 9
NUM_CODEBOOKS = len(CODEBOOK_SIZES)
MASK_ID = max(CODEBOOK_SIZES)          # 4096, shared input mask id for every codebook
END_ID = MASK_ID + 1                   # 4097, semantic id of the frame after the speech
PAD_IDS = list(range(END_ID + 1, END_ID + 8))  # 4098..4104, Rainbow Padding cycle after END
VOCAB = PAD_IDS[-1] + 1                # 4105: per-codebook input embedding width, semantic head width
OLD_VOCAB = MASK_ID + 1                # checkpoints before block diffusion (no END/PAD)
RESIDUAL_WEIGHTS = [8.0, 6.35, 5.11, 4.22, 3.63, 3.29, 3.08, 3.01, 3.0]
LANGUAGES = ("None", "en", "pl", "de", "fr", "es", "it", "pt", "ru")
LANG_DIM = 64
MTP_KV_HEADS = int(os.environ.get("OMNIFISH_MTP_KV_HEADS", "2"))
MTP_DIM = int(os.environ.get("OMNIFISH_MTP_DIM", "768"))      # MTP width (the backbone is 1024)
MTP_BLOCKS = int(os.environ.get("OMNIFISH_MTP_BLOCKS", "2"))  # transformer blocks per codebook
MTP_ARCH = os.environ.get("OMNIFISH_MTP_ARCH", "axial")
AXIAL_CKPT = os.environ.get("OMNIFISH_AXIAL_CKPT", "0") == "1"  # activation checkpointing of the AxialMTP layers
AXIAL_PROMPT_X = os.environ.get("OMNIFISH_AXIAL_PROMPT_X", "0") == "1"  # extra attention from every token to all prompt tokens  # "axial": AxialMTP (frame x level grid), "chain": MTPDecoder


def rotary(x, positions, base=10000.0):
    """x [B, H, T, D], positions [B, T]: rotate-half RoPE"""
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = positions.float()[:, None, :, None] * inv
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., : d // 2].float(), x[..., d // 2:].float()
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).to(x.dtype)


# speaker / recording IDs: one learned vector each (OmniFish.spk_embeddings / q_embeddings + projections), one token in the sequence instead of the
# sub-word pieces of "pl_20" / "q35"; id_vocab.json is append-only (a new ID gets the next row)
ID_VOCAB = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "id_vocab.json")))
ID_INDEX = {n: i for i, n in enumerate(ID_VOCAB)}
N_SPK = sum(1 for n in ID_VOCAB if not re.fullmatch(r"q\d+", n))  # speakers (and language bags) first, then recordings
SPK_DIM, Q_DIM = 256, 64
_ID_RE = re.compile(r"(?<=speaker:)(?:[A-Za-z]{2}_\w+|spk_\w+)|(?<=quality:)(q\d+)|(?<=<\|quad_start\|>)(?:[A-Za-z]{2}_\w+|spk_\w+)(?=<\|quad_end\|>)")


def encode_text(tok, s, text_vocab, ids=True):
    """tokenize s -> input_ids [1, L]; with ids, every known speaker / recording ID after "speaker:", "quality:" or
    inside a turn marker is one token text_vocab + its row in ID_VOCAB (unknown names stay text)"""
    if not ids:
        return tok(s, return_tensors="pt").input_ids
    out, last = [], 0
    for m in _ID_RE.finditer(s):
        name = m.group(0)
        if name not in ID_INDEX:
            continue
        if m.start() > last:
            out += tok(s[last:m.start()], add_special_tokens=False).input_ids
        out.append(text_vocab + ID_INDEX[name])
        last = m.end()
    if last < len(s):
        out += tok(s[last:], add_special_tokens=False).input_ids
    return torch.tensor([out], dtype=torch.long)



class IdTokenizer:
    """the text tokenizer with speaker / recording IDs as single tokens (encode_text); everything else (decode,
    pad_token_id, ...) is the wrapped tokenizer's"""

    def __init__(self, tok, text_vocab):
        self.tok, self.text_vocab = tok, text_vocab

    def __call__(self, s, return_tensors="pt", **kw):
        class _Out:
            pass
        out = _Out()
        out.input_ids = encode_text(self.tok, s, self.text_vocab)
        return out

    def __getattr__(self, name):
        if name == "tok" or name.startswith("__"):  # not set yet (unpickling) / dunder lookups
            raise AttributeError(name)
        return getattr(self.tok, name)


def context_text(text, before=(), after=()):
    """<|object_ref_start|>before<|object_ref_end|><|text_start|>text<|text_end|><|box_start|>after<|box_end|>"""
    b = f"<|object_ref_start|>{' '.join(before)}<|object_ref_end|>" if before else ""
    a = f"<|box_start|>{' '.join(after)}<|box_end|>" if after else ""
    return f"{b}<|text_start|>{text}<|text_end|>{a}"


def dialogue_text(turns):
    """[(speaker token, line), ...] -> "<|quad_start|>pl_3<|quad_end|>line line <|quad_start|>pl_7<|quad_end|>line";
    a marker only where the speaker changes"""
    out, last = [], None
    for spk, line in turns:
        out.append(line if spk == last else f"<|quad_start|>{spk}<|quad_end|>{line}")
        last = spk
    return " ".join(out)


def block_attention(kind, block):
    """kind [B, L] (0 context, 1 clean target with semantic ids only, 3 clean target with all 10 codes, 2 noisy
    target, -1 padding), block [B, L] -> [B, 1, L, L]. Without kind 3 (the default, --full-context off): context
    sees context, clean block b sees clean blocks <= b, noisy block b sees clean blocks < b and itself; earlier
    blocks are seen by their semantic ids only. With kind 3: context sees context; full block b (3) sees context and
    full blocks <= b; semantic-only block b (1, the MTP's input states) and noisy block b (2) see context, full
    blocks < b and themselves. So a block's semantic is predicted from everything decided before it, residual
    codes of earlier blocks included, and the MTP never sees the codes it predicts. Padding rows see every real
    position (no empty row)."""
    kq, kk, bq, bk = kind[:, :, None], kind[:, None, :], block[:, :, None], block[:, None, :]
    if (kind == 3).any():  # full-code context (3 copies)
        att = ((kk == 0) & (kq >= 0)) | ((kq == 3) & (kk == 3) & (bk <= bq)) \
            | ((kq == 1) & (kk == 3) & (bk < bq)) | ((kq == 1) & (kk == 1) & (bk == bq)) \
            | ((kq == 2) & (kk == 3) & (bk < bq)) | ((kq == 2) & (kk == 2) & (bk == bq)) | ((kq == -1) & (kk >= 0))
    else:  # semantic-only context (2 copies): clean block b sees clean blocks <= b, noisy block b clean < b and itself
        att = ((kk == 0) & (kq >= 0)) | ((kq == 1) & (kk == 1) & (bk <= bq)) \
            | ((kq == 2) & (kk == 1) & (bk < bq)) | ((kq == 2) & (kk == 2) & (bk == bq)) | ((kq == -1) & (kk >= 0))
    return att[:, None]


class MTPBlock(nn.Module):
    """one bidirectional transformer block over frames (pre-norm attention + SwiGLU)"""

    def __init__(self, dim, n_head, n_kv, head_dim, ffn, eps=1e-6, causal=False):
        super().__init__()
        self.n_head, self.n_kv, self.head_dim, self.causal = n_head, n_kv, head_dim, causal
        self.attn_norm, self.ffn_norm = nn.RMSNorm(dim, eps=eps), nn.RMSNorm(dim, eps=eps)
        self.wq = nn.Linear(dim, n_head * head_dim, bias=False)
        self.wkv = nn.Linear(dim, 2 * n_kv * head_dim, bias=False)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=False)
        self.w1, self.w3 = nn.Linear(dim, ffn, bias=False), nn.Linear(dim, ffn, bias=False)
        self.w2 = nn.Linear(ffn, dim, bias=False)

    def forward(self, x, positions, valid, attn):
        B, T, _ = x.shape
        a = self.attn_norm(x)
        q = self.wq(a).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.wkv(a).view(B, T, 2, self.n_kv, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, positions), rotary(k, positions)
        rep = self.n_head // self.n_kv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        mask = valid[:, None, None, :] if attn is None else attn
        if attn is None and self.causal:  # autoregressive models: a frame sees only earlier frames
            mask = mask & torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, -1))
        return x + self.w2(F.silu(self.w1(self.ffn_norm(x))) * self.w3(self.ffn_norm(x)))


class MTPLayer(nn.Module):
    """Mix code k's embedding (and the already decided code k+1 of other frames, MASK where not decided) into the
    running state (in_dim: the backbone width for the first layer, the MTP width after it), MTP_BLOCKS
    bidirectional transformer blocks over frames, predict code k+1."""

    def __init__(self, dim, vocab_in, vocab_out, in_dim=None, n_head=8, n_kv=2, head_dim=None, ffn=None, blocks=1,
                 eps=1e-6, causal=False):
        super().__init__()
        in_dim = in_dim or dim
        head_dim = head_dim or dim // n_head
        ffn = ffn or int(dim * 3.5)
        self.causal = causal
        self.embedding = nn.Embedding(vocab_in, dim)
        self.revealed = nn.Embedding(vocab_out + 1, dim)  # code k+1 where already decided; index vocab_out = MASK
        self.mask_id = vocab_out
        self.enorm, self.hnorm = nn.RMSNorm(dim, eps=eps), nn.RMSNorm(in_dim, eps=eps)
        self.eh_proj = nn.Linear(dim + in_dim, dim, bias=False)
        self.blocks = nn.ModuleList(MTPBlock(dim, n_head, n_kv, head_dim, ffn, eps, causal) for _ in range(blocks))
        self.out_norm = nn.RMSNorm(dim, eps=eps)
        self.head = nn.Linear(dim, vocab_out, bias=False)

    def forward(self, h, codes, positions, valid, attn=None, revealed=None):
        """h [B, T, in_dim], codes [B, T], positions [B, T], valid [B, T], attn [B, 1, T, T] (optional, overrides
        valid/causal), revealed [B, T] code k+1 or mask_id (None: all masked) -> (state [B, T, dim], logits)"""
        if revealed is None:
            revealed = torch.full_like(codes, self.mask_id)
        e = self.embedding(codes) + self.revealed(revealed)
        x = self.eh_proj(torch.cat((self.enorm(e), self.hnorm(h)), -1))
        for b in self.blocks:
            x = b(x, positions, valid, attn)
        x = self.out_norm(x)
        return x, self.head(x)


class PromptCross(nn.Module):
    """attention from every grid token to all prompt tokens (every prompt frame, every level); output projection
    starts at zero, so a model trained without it is unchanged until this is learned"""

    def __init__(self, dim, n_head, n_kv, head_dim, eps=1e-6):
        super().__init__()
        self.n_head, self.n_kv, self.head_dim = n_head, n_kv, head_dim
        self.qnorm, self.knorm = nn.RMSNorm(dim, eps=eps), nn.RMSNorm(dim, eps=eps)
        self.wq = nn.Linear(dim, n_head * head_dim, bias=False)
        self.wkv = nn.Linear(dim, 2 * n_kv * head_dim, bias=False)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=False)
        nn.init.zeros_(self.wo.weight)

    def forward(self, x, p, pmask):
        """x [B, N, D] queries, p [B, M, D] prompt tokens, pmask [B, M] valid prompt tokens"""
        B, N, _ = x.shape; M = p.shape[1]
        q = self.wq(self.qnorm(x)).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.wkv(self.knorm(p)).view(B, M, 2, self.n_kv, self.head_dim).permute(2, 0, 3, 1, 4)
        rep = self.n_head // self.n_kv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        has = pmask.any(1)
        m = (pmask | ~has[:, None])[:, None, None, :]  # rows without a prompt: any key, output dropped below
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return x + self.wo(o.transpose(1, 2).reshape(B, N, -1)) * has[:, None, None].to(x.dtype)


class AxialMTP(nn.Module):
    """codebooks 1..9 by attention over a (frame x level) grid: token (t, j) carries code j of frame t (+ the frame's
    backbone state, + a level embedding) and predicts code j+1. Every layer: attention over the levels of a frame
    (causal: j sees <= j; a prompt frame sees all its levels, it is fully known), attention over frames at the same
    level (a frame sees the prompt, earlier blocks and its own block), SwiGLU. Decoding: 9 steps per block, all frames
    at once (level j+1 of the block from the tokens up to level j). Same interface as MTPDecoder (new_state, prime,
    decode_block)."""

    def __init__(self, in_dim=1024, dim=768, layers=6, n_head=8, n_kv=2, ffn=2688, eps=1e-6, ctx_frames=None):
        super().__init__()
        self.dim, self.in_dim, self.ctx_frames = dim, in_dim, ctx_frames
        self.sizes = [VOCAB] + list(CODEBOOK_SIZES[1:])
        self.register_buffer("offsets", torch.tensor([0] + list(torch.tensor(self.sizes).cumsum(0)[:-1].tolist())), persistent=False)
        self.code_emb = nn.Embedding(sum(self.sizes), dim)
        self.level_emb = nn.Embedding(NUM_CODEBOOKS, dim)
        self.in_norm = nn.RMSNorm(in_dim, eps=eps)
        self.in_proj = nn.Linear(in_dim, dim, bias=False)
        hd = dim // n_head
        self.lvl = nn.ModuleList(MTPBlock(dim, n_head, n_kv, hd, ffn, eps) for _ in range(layers))  # level attention + FFN
        self.frm = nn.ModuleList(MTPBlock(dim, n_head, n_kv, hd, ffn, eps) for _ in range(layers))  # frame attention + FFN
        self.px = nn.ModuleList(PromptCross(dim, n_head, n_kv, hd, eps) for _ in range(layers)) if AXIAL_PROMPT_X else None
        self.out_norm = nn.RMSNorm(dim, eps=eps)
        self.heads = nn.ModuleList(nn.Linear(dim, CODEBOOK_SIZES[j + 1], bias=False) for j in range(NUM_CODEBOOKS - 1))

    def _tokens(self, h, codes):
        """h [B, T, in_dim], codes [B, 10, T] -> [B, T, 10, dim]"""
        c = codes.clamp(min=0) + self.offsets.view(1, -1, 1)
        x = self.code_emb(c).permute(0, 2, 1, 3) + self.level_emb.weight[None, None]
        return x + self.in_proj(self.in_norm(h))[:, :, None]

    def _run(self, x, pos, valid, block):
        """x [B, T, L, D]; masks from block (-1 prompt) / valid -> [B, T, L, D]"""
        B, T, L, D = x.shape
        lv = torch.arange(L, device=x.device)
        causal = lv[None, :, None] >= lv[None, None, :]                                       # [1, L, L]
        lmask = (causal | (block < 0).reshape(B * T, 1, 1))[:, None]                            # prompt frames: all levels
        fmask = (block[:, None, :] <= block[:, :, None]) & valid[:, None, :]                   # [B, T(q), T(k)]
        fmask = (fmask | torch.eye(T, dtype=torch.bool, device=x.device)[None])[:, None]       # a row always sees itself
        fmask = fmask.repeat_interleave(L, 0)                                                   # [B*L, 1, T, T]
        lpos = lv[None].expand(B * T, L)
        fpos = pos.repeat_interleave(L, 0)
        Pm = int((block < 0).sum(1).max()) if self.px is not None else 0  # prompt frames lead every row
        pmask = ((block[:, :Pm] < 0) & valid[:, :Pm])[:, :, None].expand(B, Pm, L).reshape(B, Pm * L) if Pm else None

        def layer(x, la, fa, px):
            x = la(x.reshape(B * T, L, D), lpos, None, lmask).reshape(B, T, L, D)
            y = fa(x.permute(0, 2, 1, 3).reshape(B * L, T, D), fpos, None, fmask)
            x = y.reshape(B, L, T, D).permute(0, 2, 1, 3)
            if px is not None and Pm:  # every token also reads the whole prompt (all frames, all levels)
                x = px(x.reshape(B, T * L, D), x[:, :Pm].reshape(B, Pm * L, D), pmask).reshape(B, T, L, D)
            return x
        for n, (la, fa) in enumerate(zip(self.lvl, self.frm)):
            px = self.px[n] if self.px is not None else None
            if AXIAL_CKPT and self.training and torch.is_grad_enabled():  # optional: recompute in backward (less memory)
                x = torch.utils.checkpoint.checkpoint(layer, x, la, fa, px, use_reentrant=False)
            else:
                x = layer(x, la, fa, px)
        return self.out_norm(x)

    def loss(self, h, codes, pos, valid, block):
        """teacher forcing: h [B, T, in_dim], codes [B, 10, T] true codes, block [B, T] (-1: prompt frames, context
        only) -> 9 losses (codebooks 1..9) on the valid target frames"""
        x = self._run(self._tokens(h, codes), pos, valid, block)
        tgt = valid & (block >= 0)
        res = []
        for j, head in enumerate(self.heads):
            logits = head(x[:, :, j][tgt])
            res.append(F.cross_entropy(logits.float(), codes[:, j + 1][tgt]))
        return torch.stack(res)

    def new_state(self, T, dim, device, dtype):
        return {"codes": torch.zeros(NUM_CODEBOOKS, T, dtype=torch.long, device=device),
                "states": [torch.zeros(1, T, dim, dtype=dtype, device=device)]}

    def prime(self, S, P):
        pass  # the prompt frames are read straight from S at every step

    @torch.no_grad()
    def decode_block(self, S, lo, hi, block, rounds=4, temperature=0.0, top_k=0, prefix=0):
        """fill codebooks 1..9 of frames lo:hi (S["states"][0] backbone states, S["codes"][0] semantic ids set;
        frames < lo decided; the first `prefix` frames are the prompt). Context: the prompt + every decided frame
        (as in training; ctx_frames caps it if set)."""
        codes, dev = S["codes"], S["codes"].device
        start = prefix if self.ctx_frames is None else max(prefix, lo - self.ctx_frames)  # default: all, as in training
        keep = torch.cat([torch.arange(prefix, device=dev), torch.arange(start, hi, device=dev)])
        i = keep
        blk = torch.where(i < prefix, -1, (i - prefix) // block)[None]
        h = S["states"][0][:, keep]
        pos = i[None]
        valid = torch.ones_like(blk, dtype=torch.bool)
        cur = (i >= lo)
        for j in range(NUM_CODEBOOKS - 1):
            c = codes[:, keep][None].clone()
            x = self._run(self._tokens(h, c), pos, valid, blk)
            lg = self.heads[j](x[0, cur, j]).float()
            if temperature > 0:
                if top_k > 0:
                    lg = lg.masked_fill(lg < lg.topk(top_k, -1).values[..., -1:], -float("inf"))
                nxt = torch.distributions.Categorical(logits=lg / temperature).sample()
            else:
                nxt = lg.argmax(-1)
            codes[j + 1, lo:hi] = nxt


class MTPDecoder(nn.Module):
    """codebooks 1..9 from the backbone state and the semantic codes: layer k embeds code k, predicts k+1"""

    def __init__(self, dim=1024, causal=False):
        """dim: the backbone width (the first layer's input); the MTP runs at MTP_DIM with MTP_BLOCKS blocks per
        codebook"""
        super().__init__()
        self.dim, self.in_dim = MTP_DIM, dim
        self.layers = nn.ModuleList(MTPLayer(MTP_DIM, CODEBOOK_SIZES[k], CODEBOOK_SIZES[k + 1], in_dim=dim if k == 0 else MTP_DIM,
                                             n_kv=MTP_KV_HEADS, blocks=MTP_BLOCKS, causal=causal)
                                    for k in range(NUM_CODEBOOKS - 1))

    def forward(self, h, codes, positions, valid, attn=None):
        """teacher forcing: codes [B, 10, T] true codes -> list of 9 logits [B, T, 1024] (codebooks 1..9)"""
        out = []
        for k, layer in enumerate(self.layers):
            h, logits = layer(h, codes[:, k], positions, valid, attn)
            out.append(logits)
        return out

    def new_state(self, T, dim, device, dtype):
        """buffers for block-by-block decoding of up to T frames: codes [10, T], clean input states of every layer"""
        return {"codes": torch.zeros(NUM_CODEBOOKS, T, dtype=torch.long, device=device),  # states[0]: backbone width
                "states": [torch.zeros(1, T, dim if k == 0 else self.dim, dtype=dtype, device=device) for k in range(len(self.layers) + 1)]}

    @torch.no_grad()
    def prime(self, S, P):
        """the P prompt frames at the start of S (backbone states + all 10 codes set): run every layer's clean pass
        over them once, so the target blocks see them as decided frames (block -1)"""
        codes, states, dev = S["codes"], S["states"], S["codes"].device
        pos = torch.arange(P, device=dev)[None]
        full = torch.ones(1, 1, P, P, dtype=torch.bool, device=dev)
        for k, layer in enumerate(self.layers):
            st, _ = layer(states[k][:, :P], codes[k, :P][None], pos, None, full, codes[k + 1, :P][None])
            states[k + 1][:, :P] = st.to(states[k + 1].dtype)

    def decode_block(self, S, lo, hi, block, rounds=4, temperature=0.0, top_k=0, prefix=0):
        """fill codebooks 1..9 of frames lo:hi (S["states"][0] = backbone states, S["codes"][0] = semantic ids
        already set there; frames < lo decided). Codebook by codebook, the frames of a codebook are unmasked over
        `rounds` iterations (most confident first, cosine schedule), each seeing the decided codes of the earlier
        blocks and of the frames revealed so far; layer k's input state is layer k-1's clean pass (all decided)."""
        codes, states, dev = S["codes"], S["states"], S["codes"].device
        pos = torch.arange(hi, device=dev)[None]
        i = torch.arange(hi, device=dev)
        bb = torch.where(i < prefix, -1, (i - prefix) // block)  # the first `prefix` frames: the prompt (block -1)
        kind = (torch.arange(hi, device=dev) >= lo).long()
        attn = (((bb[None, :, None] >= bb[None, None, :]) & (kind[None, None, :] == 0)) |
                ((kind[None, :, None] == 1) & (kind[None, None, :] == 1)))[:, None]
        cattn = (bb[None, :, None] >= bb[None, None, :])[:, None]
        for k, layer in enumerate(self.layers):
            hk = states[k][:, :hi]
            cur = torch.full((hi - lo,), layer.mask_id, dtype=torch.long, device=dev)
            for r in range(rounds):
                rev = torch.cat([codes[k + 1, :lo], cur])[None]
                _, logits = layer(hk, codes[k, :hi][None], pos, None, attn, rev)
                lg = logits[0, lo:hi].float()
                masked = cur == layer.mask_id
                probs = F.softmax(lg / max(temperature, 1e-5), -1)
                if temperature > 0:
                    if top_k > 0:
                        lg = lg.masked_fill(lg < lg.topk(top_k, -1).values[..., -1:], -float("inf"))
                    sample = torch.distributions.Categorical(logits=lg / temperature).sample()
                else:
                    sample = lg.argmax(-1)
                conf = torch.where(masked, probs.gather(1, sample[:, None])[:, 0], torch.full_like(probs[:, 0], -1.0))
                keep = int(math.floor((hi - lo) * math.cos(math.pi / 2 * (r + 1) / rounds)))
                n_fix = int(masked.sum()) - keep
                if n_fix > 0:
                    idx = conf.topk(n_fix).indices
                    cur[idx] = sample[idx]
            codes[k + 1, lo:hi] = cur
            st, _ = layer(hk, codes[k, :hi][None], pos, None, cattn, codes[k + 1, :hi][None])
            states[k + 1][:, lo:hi] = st[:, lo:hi].to(states[k + 1].dtype)

    @torch.no_grad()
    def generate(self, h, semantic, block, rounds=4, temperature=0.0, top_k=0):
        """semantic [T] codes of one utterance, h [T, dim] its backbone states -> codes [10, T], block by block"""
        T = semantic.shape[0]
        S = self.new_state(T, h.shape[-1], h.device, h.dtype)
        S["codes"][0], S["states"][0][0] = semantic, h
        for lo in range(0, T, block):
            self.decode_block(S, lo, min(T, lo + block), block, rounds, temperature, top_k)
        return S["codes"]


class OmniFish(nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        dim = llm.config.hidden_size
        self.audio_embeddings = nn.Embedding(NUM_CODEBOOKS * VOCAB, dim)
        self.semantic_head = nn.Linear(dim, VOCAB, bias=False)  # semantic codes, MASK (never a target), END, PADs
        self.mtp = AxialMTP(dim) if MTP_ARCH == "axial" else MTPDecoder(dim)
        self.text_vocab = llm.get_input_embeddings().weight.shape[0]
        # speaker / recording IDs (see encode_text): small vectors in a shared space, one projection each to the
        # model width (rare speakers learn 256 numbers, the projection learns from all of them)
        self.spk_embeddings = nn.Embedding(N_SPK, SPK_DIM)
        self.spk_proj = nn.Linear(SPK_DIM, dim)
        self.q_embeddings = nn.Embedding(len(ID_VOCAB) - N_SPK, Q_DIM)
        self.q_proj = nn.Linear(Q_DIM, dim)
        self.lang_embeddings = nn.Embedding(len(LANGUAGES), LANG_DIM)
        self.lang_proj = nn.Linear(LANG_DIM, dim)
        self.lang_start_id = None
        self.lang_token_to_index = {}
        std = float(llm.get_input_embeddings().weight.detach().std())
        nn.init.normal_(self.audio_embeddings.weight, std=std)
        nn.init.normal_(self.semantic_head.weight, std=1e-3)  # start from ~uniform semantic logits
        for head in (self.mtp.heads if MTP_ARCH == "axial" else [layer.head for layer in self.mtp.layers]):
            nn.init.normal_(head.weight, std=1e-3)
        self.register_buffer("codebook_offsets", torch.arange(NUM_CODEBOOKS) * VOCAB, persistent=False)
        w = torch.tensor(RESIDUAL_WEIGHTS, dtype=torch.float)
        self.register_buffer("residual_weights", w / w.sum(), persistent=False)

    @classmethod
    def from_omnivoice(cls, path=OMNIVOICE_PATH, dtype=torch.float32, attn="sdpa"):
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer, Qwen3Config

        cfg = json.load(open(os.path.join(path, "config.json")))
        llm_cfg = Qwen3Config(**{k: v for k, v in cfg["llm_config"].items() if k != "model_type"})
        llm_cfg._attn_implementation = attn
        llm = AutoModel.from_config(llm_cfg, dtype=dtype)
        state = load_file(os.path.join(path, "model.safetensors"))
        missing, _ = llm.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith("llm.")}, strict=False)
        if missing:
            raise RuntimeError(f"backbone weights missing: {missing[:5]}")
        tok = AutoTokenizer.from_pretrained(path)
        model = cls(llm).to(dtype)
        model.configure_languages(tok)
        return model, tok

    def configure_languages(self, tok):
        """Map the single token immediately after <|lang_start|> to the learned language table."""
        self.lang_start_id = tok.convert_tokens_to_ids("<|lang_start|>")
        self.lang_token_to_index = {}
        for i, lang in enumerate(LANGUAGES):
            ids = tok(lang, add_special_tokens=False).input_ids
            if len(ids) == 1:
                self.lang_token_to_index[int(ids[0])] = i

    def embed(self, input_ids, audio_mask):
        """input_ids [B, 10, L] (text ids repeated on every channel), audio_mask [B, L] -> [B, L, H]"""
        t = input_ids[:, 0]
        is_id = (t >= self.text_vocab) & ~audio_mask
        text = self.llm.get_input_embeddings()(torch.where(is_id, 0, t) * ~audio_mask)
        if is_id.any():
            k = (t - self.text_vocab).clamp(0, len(ID_VOCAB) - 1)
            spk = self.spk_proj(self.spk_embeddings(k.clamp(max=N_SPK - 1)))
            q = self.q_proj(self.q_embeddings((k - N_SPK).clamp(min=0)))
            text = torch.where((is_id & (k < N_SPK))[..., None], spk.to(text.dtype), text)
            text = torch.where((is_id & (k >= N_SPK))[..., None], q.to(text.dtype), text)
        if self.lang_start_id is not None and self.lang_token_to_index:
            after_start = torch.zeros_like(t, dtype=torch.bool)
            after_start[:, 1:] = t[:, :-1] == self.lang_start_id
            lang_mask = after_start & ~audio_mask
            lang_index = torch.zeros_like(t)
            for token_id, index in self.lang_token_to_index.items():
                lang_index = torch.where(t == token_id, index, lang_index)
            lang = self.lang_proj(self.lang_embeddings(lang_index))
            text = torch.where(lang_mask[..., None], lang.to(text.dtype), text)
        shifted = input_ids * audio_mask[:, None] + self.codebook_offsets.view(1, -1, 1)
        return torch.where(audio_mask[..., None], self.audio_embeddings(shifted).sum(1), text)

    def load_weights(self, state, tok):
        """load a (possibly older) state dict; a checkpoint from before the ID tables starts them from the text
        pieces of each ID (init_id_embeddings). -> True when the tables were new"""
        state = OmniFish.upgrade_state(state)
        fresh = "spk_embeddings.weight" not in state
        if fresh:
            state = dict(state, **{k: v.detach().cpu() for k, v in self.state_dict().items()
                                   if k.startswith(("spk_", "q_"))})
        own = self.state_dict()
        missing_px = [k for k in own if ".px." in k and k not in state]  # new prompt cross-attention: fresh init
        if missing_px:
            state = dict(state, **{k: own[k] for k in missing_px})
        self.load_state_dict(state)
        if fresh:
            self.init_id_embeddings(tok)
        return fresh

    @torch.no_grad()
    def init_id_embeddings(self, tok):
        """start the ID tables so that projection(vector) is the best rank-SPK_DIM / rank-Q_DIM fit (SVD) of the
        mean text embedding of each ID's sub-word pieces: the model keeps what it learned about "pl_20" as text"""
        emb = self.llm.get_input_embeddings().weight.float()
        for names, table, proj, r in ((ID_VOCAB[:N_SPK], self.spk_embeddings, self.spk_proj, SPK_DIM),
                                      (ID_VOCAB[N_SPK:], self.q_embeddings, self.q_proj, Q_DIM)):
            M = torch.stack([emb[tok(n, add_special_tokens=False).input_ids].mean(0) for n in names])  # [n, dim]
            mu = M.mean(0)
            U, S, Vh = torch.linalg.svd(M - mu, full_matrices=False)
            k = min(r, S.shape[0])
            table.weight.zero_(); proj.weight.zero_()
            table.weight[:, :k] = (U[:, :k] * S[:k].sqrt()).to(table.weight.dtype)
            proj.weight[:, :k] = (Vh[:k].T * S[:k].sqrt()).to(proj.weight.dtype)
            proj.bias.copy_(mu.to(proj.bias.dtype))

    def backbone(self, input_ids, audio_mask, attention_mask, position_ids):
        return self.llm(inputs_embeds=self.embed(input_ids, audio_mask), attention_mask=attention_mask,
                        position_ids=position_ids, return_dict=True)[0]

    @staticmethod
    def upgrade_state(state):
        """weights saved before END/PAD (input/semantic vocab 4097/4096) -> current shapes; new rows freshly
        initialised. Optimizer moments for those two tensors are remapped the same way (zeros for new rows)."""
        out = dict(state)
        emb = state["audio_embeddings.weight"]
        if emb.shape[0] == NUM_CODEBOOKS * VOCAB:
            return out
        dim = emb.shape[1]
        new = torch.randn(NUM_CODEBOOKS, VOCAB, dim, dtype=emb.dtype) * emb.std()
        new[:, :OLD_VOCAB] = emb.view(NUM_CODEBOOKS, OLD_VOCAB, dim)
        out["audio_embeddings.weight"] = new.view(-1, dim)
        head = state["semantic_head.weight"]
        newh = torch.randn(VOCAB, dim, dtype=head.dtype) * 1e-3
        newh[:head.shape[0]] = head
        out["semantic_head.weight"] = newh
        return out

    @staticmethod
    def upgrade_moment(name, t):
        """an AdamW moment of an old-shape parameter -> current shape, zeros for the new rows"""
        if t.dim() < 2:  # norms, biases: never reshaped
            return t
        dim = t.shape[1]
        if name == "audio_embeddings.weight" and t.shape[0] != NUM_CODEBOOKS * VOCAB:
            new = torch.zeros(NUM_CODEBOOKS, VOCAB, dim, dtype=t.dtype)
            new[:, :OLD_VOCAB] = t.view(NUM_CODEBOOKS, OLD_VOCAB, dim)
            return new.view(-1, dim)
        if name == "semantic_head.weight" and t.shape[0] != VOCAB:
            new = torch.zeros(VOCAB, dim, dtype=t.dtype)
            new[:t.shape[0]] = t
            return new
        return t

    @staticmethod
    def gather_targets(h, codes, target_mask):
        """target frames of every row, left-aligned: h [B, T, H], codes [B, 10, T], positions, valid [B, T]"""
        B = h.shape[0]
        n = target_mask.sum(1)
        T = int(n.max())
        order = torch.argsort((~target_mask).to(torch.int8), dim=1, stable=True)[:, :T]  # target positions first
        valid = torch.arange(T, device=h.device)[None] < n[:, None]
        ht = h.gather(1, order[..., None].expand(B, T, h.shape[-1]))
        ct = codes.gather(2, order[:, None].expand(B, codes.shape[1], T))
        positions = torch.arange(T, device=h.device)[None].expand(B, T)
        return ht, ct, positions, valid, order

    def mtp_unmask_loss(self, ht, ct, pos, valid, block):
        """MTP as masked prediction over frames: per layer, every block masks code k+1 of its frames at its own
        uniform random ratio; [clean frames | noisy frames]: clean frame i sees clean frames of blocks <= its block;
        noisy frame i sees clean frames of earlier blocks and noisy frames of its own block. Loss on masked noisy
        frames. -> 9 losses"""
        B, T = valid.shape
        kind = torch.cat([torch.zeros(B, T, dtype=torch.long, device=ht.device), torch.ones(B, T, dtype=torch.long, device=ht.device)], 1)
        bb = torch.cat([block, block], 1)
        v2 = torch.cat([valid, valid], 1)
        kq, kk, bq, bk = kind[:, :, None], kind[:, None, :], bb[:, :, None], bb[:, None, :]
        attn = ((kq == 0) & (kk == 0) & (bk <= bq)) | ((kq == 1) & (kk == 0) & (bk < bq)) | ((kq == 1) & (kk == 1) & (bk == bq))
        attn = (attn | ~v2[:, :, None]) & v2[:, None, :]  # padding rows see the valid frames (no empty row)
        attn = attn | torch.eye(2 * T, dtype=torch.bool, device=ht.device)[None]  # a row always sees itself
        attn = attn[:, None]
        pos2 = torch.cat([pos, pos], 1)
        nb = int(block.max()) + 1
        tgt_frame = valid & (block >= 0)  # block -1: the prompt's decided frames (context only, no loss)
        first = (block < 0).sum(1).clamp(max=T - 1)  # first target frame of each row
        state, res = ht, []
        for k, layer in enumerate(self.mtp.layers):
            ratio = torch.rand(B, nb, device=ht.device)
            m = (torch.rand(B, T, device=ht.device) < ratio.gather(1, block.clamp(min=0))) & tgt_frame
            none = ~m.any(1)  # at least one masked target frame per row
            m[torch.arange(B, device=ht.device)[none], first[none]] = tgt_frame[torch.arange(B, device=ht.device)[none], first[none]]
            tgt = ct[:, k + 1]
            rev = torch.cat([tgt, torch.where(m, torch.full_like(tgt, layer.mask_id), tgt)], 1)
            st, logits = layer(torch.cat([state, state], 1), torch.cat([ct[:, k], ct[:, k]], 1), pos2, v2, attn, rev)
            res.append(F.cross_entropy(logits[:, T:].float()[m], tgt[m]))
            state = st[:, :T]
        return torch.stack(res)

    @staticmethod
    def frame_attention(block, valid):
        """MTP mask over frames: frame i sees valid frame j iff block_j <= block_i -> [B, 1, T, T] (padding rows
        see the valid frames too, so no row is empty)"""
        return ((block[:, None, :, None] >= block[:, None, None, :]) | ~valid[:, None, :, None]) & valid[:, None, None, :]

    def forward(self, input_ids, audio_mask, attention_mask, position_ids, semantic_labels, codes, target_mask,
                target_block):
        """semantic_labels [B, L] (-100 except masked noisy frames), codes [B, 10, L] true codes, target_mask
        [B, L] speech frames of the clean copy (MTP targets), target_block [B, L] their block index"""
        h = self.backbone(input_ids, audio_mask, attention_mask, position_ids)
        sem_logits = self.semantic_head(h)
        sem_loss = F.cross_entropy(sem_logits.float().flatten(0, 1), semantic_labels.flatten(), ignore_index=-100)
        ht, ct, pos, valid, order = self.gather_targets(h, codes, target_mask)
        blk = target_block.gather(1, order)
        res = self.mtp.loss(ht, ct, pos, valid, blk) if MTP_ARCH == "axial" else self.mtp_unmask_loss(ht, ct, pos, valid, blk)
        loss = sem_loss + (res * self.residual_weights).sum()
        return {"loss": loss, "semantic": sem_loss.detach(), "residual": res.detach()}
