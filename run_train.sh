#!/bin/bash
# OmniFish v26 (axial frame x level decoder that reads the prompt), detached; resumes from $OUT/latest.pt.
# Prompt = the speaker's earlier lines (>= 6 s, each with its pause); context and target length grow by curriculum.
# Stops itself if available RAM falls under 6 GB.
OUT=/data/dataset/train/experiments/mtp_axial/fork_A; LOG=/data/dataset/train/experiments/mtp_axial/train.full_A.log
cd "$(dirname "$0")/ogg_tts"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /data/dataset/train/venv/bin/python -I -u train_omnifish.py \
  --out $OUT --steps 400000 --warmup 200 --batch-tokens 16384 --max-sentences 5 --max-seconds 40 \
  --max-pause 3.5 --dialogue 0 --dialogue-prompt 0 --cross-prompt 1 --prompt-min-seconds 6 \
  --context-curriculum 0:0,20000:1,60000:2,110000:3,170000:4 --sentence-curriculum 0:1,30000:2,80000:3,140000:4 \
  --speaker-mode 0.333 0.333 0.334 --quality-ratio 1 --drop-cond-ratio 0 --language-ratio 1 --tag-oversample 1 \
  --full-context --workers 4 --val-every 1000 --save-every 10000 --reload-every 2000 >> $LOG 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  f=$(awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo)
  if [ "$f" -lt 6 ]; then echo "$(date) RAM GUARD" >> $LOG; kill $PID; fi
  sleep 10
done
echo "$(date) training exited" >> $LOG
