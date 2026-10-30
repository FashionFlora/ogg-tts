"""Fish Audio codec (Drama 3 / S2 codec.pth): 44.1 kHz, hop 2048 (~21.53 frames/s),
10 codebooks = 1 semantic (4096) + 9 residual (1024 each)."""
import glob
import os
import sys

import torch

FISH_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish-speech")
sys.path.insert(0, FISH_SRC)
# codec weights: $FISH_CODEC_PATH, else the local HF cache of the training machine
CODEC_PATH = os.environ.get("FISH_CODEC_PATH") or next(iter(glob.glob("/data/dataset/train/hf/models--fishaudio--drama-3/snapshots/*/codec.pth")), None)

SAMPLE_RATE = 44100
HOP = 2048
CODEBOOK_SIZES = [4096] + [1024] * 9


def load_codec(device="cuda"):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=os.path.join(FISH_SRC, "fish_speech/configs")):
        cfg = compose(config_name="modded_dac_vq")
    model = instantiate(cfg)
    state = torch.load(CODEC_PATH, map_location=device, mmap=True, weights_only=True)
    state = state.get("state_dict", state)
    if any("generator" in k for k in state):
        state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"codec weights missing: {missing[:5]} ...")
    return model.eval().to(device)


@torch.inference_mode()
def encode(model, audio, lengths=None):
    """audio [B, T] float at 44.1 kHz -> codes [B, 10, frames] (long), frame lengths [B]."""
    if lengths is None:
        lengths = torch.full((audio.shape[0],), audio.shape[-1], device=audio.device)
    codes, feature_lengths = model.encode(audio[:, None], lengths)
    return codes.long(), feature_lengths


@torch.inference_mode()
def decode(model, codes):
    """codes [B, 10, frames] -> audio [B, T] at 44.1 kHz."""
    return model.from_indices(codes.clone())[:, 0]  # decode clamps indices in place
