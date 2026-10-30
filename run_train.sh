#!/bin/bash
# OmniFish block-diffusion training from the OmniVoice weights, detached; resumes from ckpt/omnifish_bd/latest.pt.
# Stops itself (before the encoders' guard at 8 GB) if available RAM falls under 12 GB.
cd /data/dataset/train/src
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ../venv/bin/python -I -u train_omnifish.py \
  --out /data/dataset/train/ckpt/omnifish_bd \
  --steps 400000 --warmup 1000 --batch-tokens 16384 --max-seconds 40 --max-pause 3.5 --no-grad-ckpt --workers 4 --val-every 1000 --save-every 2000 --reload-every 2000 \
  >> /data/dataset/train/train_omnifish_bd.log 2>&1 &
PID=$!
echo $PID > /data/dataset/train/train.pid
while kill -0 $PID 2>/dev/null; do
  free_gb=$(awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo)
  if [ "$free_gb" -lt 12 ]; then echo "$(date) RAM GUARD: ${free_gb} GB free, stopping training" >> /data/dataset/train/train_omnifish_bd.log; kill $PID; fi
  sleep 5
done
echo "$(date) training exited" >> /data/dataset/train/train_omnifish_bd.log
