# OGG TTS

**Expressive multilingual text-to-speech with block diffusion over Fish Audio codec tokens.**

OGG TTS is named after the trolls of *The Witcher*. Its planned siblings are **ROGG** and **POGG**: the three
architectures compared in the technical report. Like its namesakes, OGG talks a lot.

- Model weights and model card: https://huggingface.co/FashionFlora/ogg-tts (current release: checkpoint 120k).
- Project page (architecture, samples, evaluation): https://fashionflora.github.io/ogg-tts/
- Playground: https://huggingface.co/spaces/FashionFlora/ogg-tts
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
- **Dialogue and long text with context:** each line is generated from its speaker's prompt, with the neighbouring lines as text context (`--before`, `--after`), so the delivery fits the scene.
- **Streaming:** audio is produced block by block (0.74 s blocks).

## Architecture

| Part | |
|---|---|
| Backbone | OmniVoice (Qwen3-0.6B, masked discrete diffusion), fine-tuned for **block diffusion** (LLaDA2.0 / BD3-LM style): bidirectional inside a block of 16 frames, block-causal across blocks; earlier blocks are seen with all 10 codebooks |
| Acoustic decoder | Axial MTP (110M params): 6 layers over a (frame × codebook) grid, each with attention across the codebooks of a frame and across the frames of a codebook |
| Total | ~0.75B parameters (codec excluded) |
| Codec | Fish Audio Drama 3 / S2 codec: 44.1 kHz, 21.53 frames/s, 1 semantic codebook (4096) + 9 residual (1024) |

**Generation of each block:**
1. The backbone unmasks the semantic tokens of the block in 6 confidence-ordered iterations, with classifier-free guidance.
2. The axial MTP fills the 9 residual codebooks of the block, one codebook per step, all frames at once.
3. The codec decodes the block to audio.

Finished blocks never change, so they can be streamed and cached (KV cache). The model decides the length itself (END token).

**Guidance (`--cfg`)** also sets the pace: 1–1.5 gives the natural tempo of the voice, 2 speaks faster and more pointedly.

See `ogg_tts/omnifish.py`.

## Training

- **Data:** about 8,200 hours of studio-quality speech in 8 languages, a private collection (not publicly available).
- **Emotion tags:** produced by an audio-LLM pass (Gemini) that tags only what is audible.
- **Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell.

Evaluation results are in the model card on Hugging Face.

## Repository

```
ogg_tts/
  omnifish.py         model: OmniVoice backbone + semantic head + MTP decoder, block attention
  train_omnifish.py   training of the first release (v0)
  infer_omnifish.py   inference (KV-cached block decoding)
  fish_codec.py       Fish codec loading (encode / decode)
  encode_tokens.py    audio -> Fish tokens
  qwenfish.py         autoregressive baseline (Qwen + Fish tokens)
data_prep/            speaker map/tokens, quality tokens, kept sentences, sentence-level ECAPA2 check, Gemini text export
configs/tags_spec.json   the tag vocabulary and tagging rules
run_train.sh          the training command of the first release (v0)
```

**Setup:**
- The scripts expect the original paths of the training machine (`/data/dataset/...`). Adjust them for your setup.
  For inference, `OMNIVOICE_PATH` and `FISH_CODEC_PATH` can point to downloaded `k2-fsa/OmniVoice` and `fishaudio/drama-3/codec.pth`.
- Fish Audio codec code goes in `ogg_tts/fish-speech`:
  ```
  git clone https://github.com/fishaudio/fish-speech ogg_tts/fish-speech
  ```
  The codec weights (`codec.pth`) come from the `fishaudio/drama-3` model on Hugging Face, under the Fish Audio license.
- OmniVoice weights: `k2-fsa/OmniVoice` on Hugging Face.

**Inference** with the released weights (`model.safetensors` from Hugging Face):

```
huggingface-cli download FashionFlora/ogg-tts model.safetensors --local-dir .
python ogg_tts/infer_omnifish.py --ckpt model.safetensors \
  --prompt-tokens prompt.npy:0:120 --prompt-text "Text of the prompt." \
  --text "[sighing] Text to speak." --lang en --cfg 1.5 --out out.wav
```

`prompt.npy` holds the Fish codes of the reference audio (`ogg_tts/encode_tokens.py`, or `fish_codec.encode`).

**A line of dialogue with its context** (the neighbouring lines are not spoken, they set the delivery):

```
python ogg_tts/infer_omnifish.py --ckpt model.safetensors --prompt-tokens voice.npy:0:180 --prompt-text "..." \
  --before "Where were you last night?" --text "[frustrated] I told you already. I was at work." \
  --after "At three in the morning?" --lang en --out line.wav
```

## Licenses and responsible use

- **This code:** Apache 2.0 (`LICENSE`).
- **Fish Audio codec and software** are not included and are under the Fish Audio Community License (`LICENSE-FISH-AUDIO.md`, `NOTICE`). Use of
  the model's audio output depends on that codec.
- **Do not clone or imitate the voice of a real person without their consent.** No deceptive impersonation, no fraud,
  no sexual content involving minors.
- The released weights are intended for research.
