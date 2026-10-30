"""QwenFish: Qwen3-TTS-12Hz-0.6B-Base talker (autoregressive) predicting the Fish Audio Drama 3 / S2 codec's
semantic codebook frame by frame + a codebook-MTP decoder trained from scratch (the same ~164M design as
OmniFish, causal over frames) for the 9 residual codebooks - a small Drama 3.

Kept from Qwen3-TTS: the 28 talker layers + final norm, the text embedding (2048) and its text_projection to
1024. New: Fish codec embeddings (input), the semantic head, the MTP decoder.

Dual-track input, as Qwen3-TTS: position t of the audio track carries the sum of the 10 codec embeddings of
frame t-1 (frame -1 = BOS) plus the projected embedding of text token t; once the text is used up, tts_eos
then tts_pad is added instead. Position t predicts the semantic code of frame t, or EOS after the last frame.
Prefix before the audio track: the language tag as projected text.
"""
import glob
import json
import os

import torch
import torch.nn.functional as F
from torch import nn

from omnifish import CODEBOOK_SIZES, NUM_CODEBOOKS, RESIDUAL_WEIGHTS, MTPDecoder

QWEN_TTS_PATH = glob.glob("/data/dataset/train/hf/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/*")[0]
SEM = CODEBOOK_SIZES[0]
BOS, EOS = SEM, SEM + 1                 # extra ids of the semantic codebook (input BOS, output EOS)
TTS_PAD, TTS_BOS, TTS_EOS = 151671, 151672, 151673


class QwenFish(nn.Module):
    def __init__(self, llm, text_embedding, text_projection):
        super().__init__()
        self.llm, self.text_embedding, self.text_projection = llm, text_embedding, text_projection
        dim = llm.config.hidden_size
        sizes = [SEM + 2] + CODEBOOK_SIZES[1:]
        self.codec_embeddings = nn.ModuleList(nn.Embedding(n, dim) for n in sizes)
        self.semantic_head = nn.Linear(dim, SEM + 2, bias=False)  # semantic codes + BOS (never a target) + EOS
        self.mtp = MTPDecoder(dim, causal=True)
        std = 0.02
        for e in self.codec_embeddings:
            nn.init.normal_(e.weight, std=std)
        nn.init.normal_(self.semantic_head.weight, std=1e-3)
        for layer in self.mtp.layers:
            nn.init.normal_(layer.head.weight, std=1e-3)
        w = torch.tensor(RESIDUAL_WEIGHTS, dtype=torch.float)
        self.register_buffer("residual_weights", w / w.sum(), persistent=False)

    @classmethod
    def from_qwen3_tts(cls, path=QWEN_TTS_PATH, dtype=torch.float32):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

        t = json.load(open(os.path.join(path, "config.json")))["talker_config"]
        cfg = Qwen3Config(hidden_size=t["hidden_size"], num_hidden_layers=t["num_hidden_layers"],
                          num_attention_heads=t["num_attention_heads"], num_key_value_heads=t["num_key_value_heads"],
                          head_dim=t["head_dim"], intermediate_size=t["intermediate_size"], rms_norm_eps=t.get("rms_norm_eps", 1e-6),
                          rope_theta=t["rope_theta"], max_position_embeddings=t["max_position_embeddings"], vocab_size=1)
        cfg._attn_implementation = "sdpa"
        llm = Qwen3Model(cfg)
        state = load_file(os.path.join(path, "model.safetensors"))
        body = {k[len("talker.model."):]: v for k, v in state.items()
                if k.startswith("talker.model.layers.") or k == "talker.model.norm.weight"}
        missing, _ = llm.load_state_dict(body, strict=False)
        if [m for m in missing if not m.startswith("embed_tokens")]:
            raise RuntimeError(f"talker weights missing: {missing[:5]}")
        text_embedding = nn.Embedding(*state["talker.model.text_embedding.weight"].shape)
        text_embedding.weight.data.copy_(state["talker.model.text_embedding.weight"])
        th = t["text_hidden_size"]
        act = {"silu": nn.SiLU(), "gelu": nn.GELU()}[t["hidden_act"]]
        text_projection = nn.Sequential()
        text_projection.linear_fc1 = nn.Linear(th, th)
        text_projection.act = act
        text_projection.linear_fc2 = nn.Linear(th, t["hidden_size"])
        for n in ("linear_fc1", "linear_fc2"):
            getattr(text_projection, n).weight.data.copy_(state[f"talker.text_projection.{n}.weight"])
            getattr(text_projection, n).bias.data.copy_(state[f"talker.text_projection.{n}.bias"])
        tok = AutoTokenizer.from_pretrained(path)
        return cls(llm, text_embedding, text_projection).to(dtype), tok

    def text(self, ids):
        return self.text_projection(self.text_embedding(ids))

    def frame_sum(self, codes):
        """codes [B, 10, T] -> [B, T, H]: summed embeddings of every codebook of each frame"""
        return sum(e(codes[:, k]) for k, e in enumerate(self.codec_embeddings))

    def inputs(self, prefix_ids, text_ids, text_len, codes, n_frames):
        """prefix_ids [B, P], text_ids [B, N] (padded), text_len [B], codes [B, 10, T] (true frames),
        n_frames [B] -> embeddings [B, P + T + 1, H]; audio track position t = frame t-1 + text token t"""
        B, T = codes.shape[0], codes.shape[2]
        bos = torch.zeros(B, NUM_CODEBOOKS, 1, dtype=torch.long, device=codes.device)
        bos[:, 0] = BOS
        prev = torch.cat([bos, codes], 2)                                   # frame t-1 at position t
        audio = self.frame_sum(prev)                                        # [B, T+1, H]
        t = torch.arange(T + 1, device=codes.device)[None]
        n = text_ids.shape[1]
        tid = text_ids.gather(1, t.clamp(max=n - 1).expand(B, -1))
        tid = torch.where(t < text_len[:, None], tid, torch.where(t == text_len[:, None], TTS_EOS, TTS_PAD))
        return torch.cat([self.text(prefix_ids), audio + self.text(tid)], 1)

    def forward(self, prefix_ids, text_ids, text_len, codes, n_frames, n_prompt):
        """codes [B, 10, T] padded frames, n_frames [B] real frames, n_prompt [B] prompt frames (no loss)"""
        B, _, T = codes.shape
        P = prefix_ids.shape[1]
        x = self.inputs(prefix_ids, text_ids, text_len, codes, n_frames)
        L = x.shape[1]
        valid = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        valid[:, :P] = True
        valid[:, P:] = torch.arange(T + 1, device=x.device)[None] <= n_frames[:, None]
        att = (valid[:, None, None, :] & torch.ones(L, L, dtype=torch.bool, device=x.device).tril())
        pos = torch.arange(L, device=x.device)[None].expand(B, L)
        h = self.llm(inputs_embeds=x, attention_mask=att, position_ids=pos, return_dict=True)[0][:, P:]  # [B, T+1, H]
        t = torch.arange(T + 1, device=x.device)[None]
        target = torch.cat([codes[:, 0], torch.full((B, 1), EOS, device=x.device)], 1)
        target = torch.where(t == n_frames[:, None], EOS, target)
        keep = (t >= n_prompt[:, None]) & (t <= n_frames[:, None])
        sem_loss = F.cross_entropy(self.semantic_head(h).float()[keep], target[keep])
        # MTP: state at position t + semantic code of frame t -> residual codes of frame t (frames < n_frames)
        fvalid = torch.arange(T, device=x.device)[None] < n_frames[:, None]
        floss = fvalid & (torch.arange(T, device=x.device)[None] >= n_prompt[:, None])
        res = []
        for k, logits in enumerate(self.mtp(h[:, :T], codes, pos[:, :T], fvalid)):
            res.append(F.cross_entropy(logits.float()[floss], codes[:, k + 1][floss]))
        res = torch.stack(res)
        return {"loss": sem_loss + (res * self.residual_weights).sum(), "semantic": sem_loss.detach(), "residual": res.detach()}
