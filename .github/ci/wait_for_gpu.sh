#!/usr/bin/env bash
# Wait until no other process holds GPU memory (the runner shares the GPU with the user's own
# runs; a CI job next to a long run could make either run out of memory). Processes using less
# than GPU_IDLE_MIB (default 300 MiB, e.g. the display server) do not count. Gives up after
# GPU_WAIT_MINUTES (default 30) with an error, so the job can be re-run later.
set -euo pipefail
idle=${GPU_IDLE_MIB:-300}
limit=$(( ${GPU_WAIT_MINUTES:-30} * 60 ))
waited=0
while true; do
  busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits \
         | awk -F', ' -v t="$idle" '$3 > t {print "pid " $1 " (" $2 "): " $3 " MiB"}')
  if [ -z "$busy" ]; then
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  if [ "$waited" -ge "$limit" ]; then
    echo "::error::GPU busy for ${GPU_WAIT_MINUTES:-30} min; re-run this job when it is free:"
    echo "$busy"
    exit 1
  fi
  [ $(( waited % 300 )) -eq 0 ] && { echo "GPU busy, waiting:"; echo "$busy"; }
  sleep 30; waited=$(( waited + 30 ))
done
