#!/usr/bin/env bash
# Auto GPU Selector and Single-Job Runner
# Usage: bash process.sh <BASE_DIR> <TGT_NAME> <OBJ>
# - If CUDA_VISIBLE_DEVICES is preset by the caller (e.g., run_masks.sh),
#   this script WILL NOT scan GPUs and will run on logical GPU 0 in the
#   visible set (i.e., just uses the preset).
# - If not preset, it will scan with nvidia-smi and pick an idle GPU.

set -Eeuo pipefail

# ===================== Config =====================
MEMORY_THRESHOLD=1000   # MB
CHECK_INTERVAL=5        # seconds
MAX_WAIT_TIME=300       # seconds

# ===================== Args =======================
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <BASE_DIR> <TGT_NAME> <OBJ>" 1>&2
  exit 2
fi
BASE_DIR="$1"           # e.g. /data/.../rich_07_img
TGT_NAME="$2"           # e.g. cam_06
OBJ="$3"                # e.g. chair

# ===================== Colors & log ================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "$@" 1>&2; }

# ===================== Helpers ====================
check_nvidia_smi() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "${RED}Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed.${NC}"
    exit 1
  fi
}

find_idle_gpu() {
  local idle_gpu=-1
  local min_memory=999999

  local num_gpus
  num_gpus=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')

  if [[ "$num_gpus" -eq 0 ]]; then
    log "${RED}No GPUs found on this system.${NC}"
    echo -1
    return
  fi

  log "${YELLOW}Checking ${num_gpus} GPU(s) for availability...${NC}"
  for gpu_id in $(seq 0 $((num_gpus - 1))); do
    local mem_used mem_total gpu_util
    mem_used=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    mem_total=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
    gpu_util=$(nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    log "  GPU ${gpu_id}: Memory used: ${mem_used}MB / ${mem_total}MB, Utilization: ${gpu_util}%"

    if [[ "$mem_used" -lt "$MEMORY_THRESHOLD" && "$mem_used" -lt "$min_memory" ]]; then
      min_memory=$mem_used
      idle_gpu=$gpu_id
    fi
  done

  echo "$idle_gpu"
}

wait_for_idle_gpu() {
  local wait_time=0
  while [[ "$wait_time" -lt "$MAX_WAIT_TIME" ]]; do
    local id
    id=$(find_idle_gpu)
    if [[ "$id" != "-1" ]]; then
      log "${GREEN}Found idle GPU: ${id}${NC}"
      echo "$id"
      return
    fi
    log "${YELLOW}No idle GPU found. Waiting ${CHECK_INTERVAL}s... (${wait_time}s / ${MAX_WAIT_TIME}s)${NC}"
    sleep "$CHECK_INTERVAL"
    wait_time=$((wait_time + CHECK_INTERVAL))
  done
  log "${RED}Timeout: No idle GPU found after ${MAX_WAIT_TIME} seconds.${NC}"
  exit 1
}

run_job() {
  # If CUDA_VISIBLE_DEVICES_OVERRIDE is set, we use it just for this command.
  # Otherwise, we assume caller already set CUDA_VISIBLE_DEVICES (or none needed).
  local env_prefix=()
  if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]]; then
    env_prefix=(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE}")
  fi

  log "${GREEN}========================================${NC}"
  if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]]; then
    log "${GREEN}Running job on GPU ${CUDA_VISIBLE_DEVICES_OVERRIDE}${NC}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    log "${GREEN}Running job with preset CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' (logical GPU 0)${NC}"
  else
    log "${GREEN}Running job (no CUDA_VISIBLE_DEVICES set)${NC}"
  fi
  log "${GREEN}========================================${NC}"
  log "Command: ${env_prefix[*]} python process.py --base_dir \"$BASE_DIR\" --tgt_name \"$TGT_NAME\" --obj \"$OBJ\""
  log ""

  "${env_prefix[@]}" python process.py \
    --base_dir "$BASE_DIR" \
    --tgt_name "$TGT_NAME" \
    --obj "$OBJ"

  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    log "${GREEN}Job completed successfully!${NC}"
  else
    log "${RED}Job failed with exit code: ${exit_code}${NC}"
    exit $exit_code
  fi
}

# ===================== Main =======================
main() {
  log "${GREEN}========================================${NC}"
  log "${GREEN}Auto GPU Selector and Job Runner (Single)${NC}"
  log "${GREEN}========================================${NC}"
  log ""

  # Basic sanity check (optional)
  if [[ ! -d "$BASE_DIR" ]]; then
    log "${RED}BASE_DIR not found: $BASE_DIR${NC}"
    exit 2
  fi

  # If caller already pinned a device, DO NOT scan all GPUs.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    log "${YELLOW}CUDA_VISIBLE_DEVICES is preset by caller: '${CUDA_VISIBLE_DEVICES}'. Skipping GPU scan.${NC}"
    # When a single physical GPU is exposed, use logical index 0.
    # We don't override the env; just run directly.
    run_job
    exit 0
  fi

  # Otherwise, self-select an idle GPU.
  check_nvidia_smi
  log "${YELLOW}Looking for an idle GPU...${NC}"
  local idle_gpu
  idle_gpu=$(find_idle_gpu)
  if [[ "$idle_gpu" = "-1" ]]; then
    log "${YELLOW}No idle GPU found immediately. Waiting for one to become available...${NC}"
    idle_gpu=$(wait_for_idle_gpu)
  fi

  # Use an override so we don't export globally.
  CUDA_VISIBLE_DEVICES_OVERRIDE="$idle_gpu" run_job
}

main
