#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_DIR:-$(dirname "${SCRIPT_DIR}")}"
SERVERS_YAML="${PROJECT_ROOT}/SERVERS.yaml"
LOCAL_CODE_ROOT="${PROJECT_ROOT}/AdaRES"

SERVER="${1:-}"
CFG="${2:-}"
GPU_ID="${3:-0}"
NUM_GPUS="${4:-1}"

if [[ -z "${SERVER}" || -z "${CFG}" ]]; then
    echo "Usage: $0 <server> <config> [gpu_id] [num_gpus]"
    exit 1
fi
if [[ "${NUM_GPUS}" != "1" || "${GPU_ID}" == *","* || "${GPU_ID}" == *" "* ]]; then
    echo "Error: AdaRES follow-up experiments are single-GPU only."
    exit 2
fi

yaml_value() {
    local section="$1" key="$2"
    python -c "
import yaml
with open(r'${SERVERS_YAML}', encoding='utf-8') as f:
    data = yaml.safe_load(f)
value = data.get('servers', {}).get('${section}', {}).get('${key}', '')
print('' if value is None else value)
"
}

HOST="$(yaml_value "${SERVER}" host)"
USER="$(yaml_value "${SERVER}" user)"
REMOTE_CODE_ROOT="$(yaml_value "${SERVER}" remote_code_root)"
PYTHON="$(yaml_value "${SERVER}" python)"
SITE_PACKAGES="$(yaml_value "${SERVER}" site_packages)"
POINTNET_EGG="$(yaml_value "${SERVER}" pointnet_egg)"

if [[ -z "${HOST}" || -z "${REMOTE_CODE_ROOT}" || -z "${PYTHON}" ]]; then
    echo "Error: incomplete server configuration for ${SERVER}."
    exit 1
fi

ALLOWED="$(python -c "
import yaml
with open(r'${SERVERS_YAML}', encoding='utf-8') as f:
    data = yaml.safe_load(f)
print(','.join(str(x) for x in data['servers']['${SERVER}'].get('allowed_gpus', [])))
")"
if [[ ",${ALLOWED}," != *",${GPU_ID},"* ]]; then
    echo "Error: GPU ${GPU_ID} is outside ${SERVER} allowed_gpus=[${ALLOWED}]."
    exit 2
fi
if [[ ! -f "${LOCAL_CODE_ROOT}/${CFG}" ]]; then
    echo "Error: config not found: ${LOCAL_CODE_ROOT}/${CFG}"
    exit 1
fi

SSH_TARGET="${USER}@${HOST}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CFG_BASENAME="$(basename "${CFG}" .yaml)"
REMOTE_LOG="${REMOTE_CODE_ROOT}/logs/train_${CFG_BASENAME}_${TIMESTAMP}.log"
SOURCE_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"

echo "[1/3] Syncing committed AdaRES code to ${SERVER}..."
git -C "${PROJECT_ROOT}" archive HEAD:AdaRES | ssh "${SSH_TARGET}" \
    "mkdir -p '${REMOTE_CODE_ROOT}' && tar -C '${REMOTE_CODE_ROOT}' -xf - && printf '%s\n' '${SOURCE_COMMIT}' > '${REMOTE_CODE_ROOT}/.codex_source_commit'"

echo "[2/3] Verifying GPU ${GPU_ID} is idle..."
GPU_USED="$(ssh "${SSH_TARGET}" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i '${GPU_ID}'")"
if [[ "${GPU_USED:-999999}" -ge 1024 ]]; then
    echo "Error: GPU ${GPU_ID} is not idle (${GPU_USED} MiB used)."
    exit 3
fi

echo "[3/3] Launching ${CFG_BASENAME} on ${SERVER} GPU ${GPU_ID}..."
ssh "${SSH_TARGET}" "
    set -e
    mkdir -p '${REMOTE_CODE_ROOT}/logs'
    cd '${REMOTE_CODE_ROOT}'
    export PYTHONPATH='${REMOTE_CODE_ROOT}:${REMOTE_CODE_ROOT}/ipdn/lib:${SITE_PACKAGES}:${POINTNET_EGG}'
    nohup '${PYTHON}' -S tools/train.py '${CFG}' --num_gpus 1 --gpu_ids '${GPU_ID}' \
        > '${REMOTE_LOG}' 2>&1 < /dev/null &
    pid=\$!
    echo \"\$pid\" > 'logs/.pid_${CFG_BASENAME}'
    sleep 5
    if ! kill -0 \"\$pid\" 2>/dev/null; then
        tail -n 30 '${REMOTE_LOG}'
        exit 1
    fi
    echo \"PID=\$pid\"
    echo 'Training started successfully'
"

echo "Started ${CFG_BASENAME} on ${SERVER} GPU ${GPU_ID}"
echo "Log: ${REMOTE_LOG}"
