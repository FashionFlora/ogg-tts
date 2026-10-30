# OGG TTS

**Expressive multilingual text-to-speech with block diffusion over Fish Audio codec tokens.**

OGG TTS is named after the trolls of *The Witcher*. Its planned siblings are **ROGG** and **POGG**: the three
architectures compared in the technical report. Like its namesakes, OGG talks a lot.

- Model weights and model card: https://huggingface.co/FashionFlora/ogg-tts (published at checkpoint 20k).
- Status: pre-training in progress (400k steps planned).

Built with Fish Audio.

## What it does

- **Zero-shot voice cloning** from a few seconds of reference audio, or a built-in speaker identity without any audio prompt.
- **8 languages:** English, Polish, Spanish, German, Russian, French, Portuguese and Italian.
- **Emotions and vocal events as optional inline tags**:
  ```
  [frustrated] Nie ma chwili spokoju… [sighing] O co tu w ogóle chodzi?
  [scared] [whispering] Ktoś tu jest…
  [shouting] Run!
  ```
  The closed list has 24 tags in three groups:
  - emotion: `[angry]` `[frustrated]` `[excited]` `[tired]` `[playful]` `[sad]` `[scared]` `[shouting]`;
  - delivery: `[whispering]` `[softly]` `[voice breaking]` `[singing]` `[pause]` `[slowly]`;
  - vocal events: `[laughing]` `[sighing]` `[gasping]` `[panting]` `[coughing]` `[clears throat]` `[sniffling]` `[scoffing]` `[yawning]` `[humming]`.
- **Multi-speaker scenes** in one pass. Turn markers name either known speakers or arbitrary voices (S1, S2, …), each with its own short prompt.
- **Streaming:** audio is produced block by block (0.74 s blocks).

## Architecture

| Part | |
|---|---|
| Backbone | OmniVoice (Qwen3-0.6B, masked discrete diffusion), fine-tuned for **block diffusion** (LLaDA2.0 / BD3-LM style): bidirectional inside a block of 16 frames, block-causal across blocks |
| Acoustic decoder | MTP: 9 chained transformer layers, one per residual codebook (164M params), attending over frames |
| Codec | Fish Audio Drama 3 / S2 codec: 44.1 kHz, 21.53 frames/s, 1 semantic codebook (4096) + 9 residual (1024) |

**Generation of each block:**
1. The backbone unmasks the semantic tokens of the block in 8 confidence-ordered iterations, with classifier-free guidance.
2. The MTP fills the 9 residual codebooks of the whole block at once.
3. The codec decodes the block to audio.

Finished blocks never change, so they can be streamed and cached. The model decides the length itself (END token + Rainbow Padding).

**Training sequence:**

```
[lang + instruct] [text] [prompt frames (10 codes)] [clean target blocks] [noisy target blocks]
```

**Attention:**
- context sees context;
- clean block *b* sees the context and clean blocks ≤ *b*;
- noisy block *b* sees the context, clean blocks < *b*, and itself.

See `ogg_tts/omnifish.py`.

## Training run (v0)

- **Data:** ~8.2k hours of professional voice acting. Sources: multi-cast audiobook productions, single-narrator audiobooks, and voiced dialogue
  from four AAA games. 16.6k speakers, 8 languages.
- **Emotion tags:** produced by an audio-LLM pass (Gemini) that tags only what is audible.
- **Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell (AWS `g7e.2xlarge`, ~$3.36/h), ~320 h of audio per training hour.
- **Cost:** about $500 of GPU for data encoding, speaker checks and the full 400k-step pre-training (~103 h). Tagging used about 129M
  Gemini tokens, ~$180 at list price.

Full details are in the model card on Hugging Face.

## Repository

```
ogg_tts/
  omnifish.py         model: OmniVoice backbone + semantic head + MTP decoder, block attention
  train_omnifish.py   training (block diffusion + MTP)
  infer_omnifish.py   inference and demos
  fish_codec.py       Fish codec loading (encode / decode)
  encode_tokens.py    audio -> Fish tokens
  qwenfish.py         autoregressive baseline (Qwen + Fish tokens)
data_prep/            speaker map/tokens, quality tokens, kept sentences, sentence-level ECAPA2 check, Gemini text export
configs/tags_spec.json   the tag vocabulary and tagging rules
run_train.sh          the training command used for v0
```

**Setup:**
- The scripts expect the original paths of the training machine (`/data/dataset/...`). Adjust them for your setup.
- Fish Audio codec code goes in `ogg_tts/fish-speech`:
  ```
  git clone https://github.com/fishaudio/fish-speech ogg_tts/fish-speech
  ```
  The codec weights (`codec.pth`) come from the `fishaudio/drama-3` model on Hugging Face, under the Fish Audio license.
- OmniVoice weights: `k2-fsa/OmniVoice` on Hugging Face.

**Inference:**

```
python ogg_tts/infer_omnifish.py --ckpt step20000.pt \
  --prompt-tokens prompt.npy:0:120 --prompt-text "Text of the prompt." \
  --text "[sighing] Text to speak." --lang en --out out.wav
```

**Dialogue with known speakers:**

```
--speaker dialogue --text "<|quad_start|>pl_17<|quad_end|>Słyszałeś to? <|quad_start|>pl_23<|quad_end|>Nic nie słyszałem."
```

## Licenses and responsible use

- **This code:** Apache 2.0 (`LICENSE`).
- **Fish Audio codec and software** are not included and are under the Fish Audio Community License (`LICENSE-FISH-AUDIO.md`, `NOTICE`). Use of
  the model's audio output depends on that codec.
- **Do not clone or imitate the voice of a real person without their consent.** No deceptive impersonation, no fraud,
  no sexual content involving minors.
- The training data contains copyrighted recordings. The released weights are intended for research.
