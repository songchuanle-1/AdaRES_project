#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_DIR:-$(dirname "${SCRIPT_DIR}")}"
SERVERS_YAML="${PROJECT_ROOT}/SERVERS.yaml"
LOCAL_CODE_ROOT="${PROJECT_ROOT}/AdaRES"

SERVER="${1:-}"
CFG="${2:-}"
GPU_ID="${3:-0}"
RESUME_CKPT="${4:-}"

if [[ -z "${SERVER}" || -z "${CFG}" ]]; then
    echo "Usage: $0 <server> <config> <gpu_id> [checkpoint]"
    exit 1
fi
if [[ "${GPU_ID}" == *","* || "${GPU_ID}" == *" "* ]]; then
    echo "Error: AdaRES follow-up resumes are single-GPU only."
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
REMOTE_WORK_DIR="$(yaml_value "${SERVER}" work_dir)"
PYTHON="$(yaml_value "${SERVER}" python)"
SITE_PACKAGES="$(yaml_value "${SERVER}" site_packages)"
POINTNET_EGG="$(yaml_value "${SERVER}" pointnet_egg)"

if [[ -z "${HOST}" || -z "${REMOTE_CODE_ROOT}" || -z "${REMOTE_WORK_DIR}" || -z "${PYTHON}" ]]; then
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
SOURCE_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"

if [[ -z "${RESUME_CKPT}" ]]; then
    RESUME_CKPT="$(ssh "${SSH_TARGET}" \
        "ls -t '${REMOTE_WORK_DIR}/${CFG_BASENAME}'/*/last.pth 2>/dev/null | head -1")"
fi
if [[ -z "${RESUME_CKPT}" ]]; then
    echo "Error: no last.pth found for ${CFG_BASENAME}."
    exit 1
fi

REMOTE_LOG="${REMOTE_CODE_ROOT}/logs/resume_${CFG_BASENAME}_${TIMESTAMP}.log"

echo "[1/3] Syncing committed AdaRES code to ${SERVER}..."
git -C "${PROJECT_ROOT}" archive HEAD:AdaRES | ssh "${SSH_TARGET}" \
    "mkdir -p '${REMOTE_CODE_ROOT}' && tar -C '${REMOTE_CODE_ROOT}' -xf - && printf '%s\n' '${SOURCE_COMMIT}' > '${REMOTE_CODE_ROOT}/.codex_source_commit'"

echo "[2/3] Verifying GPU ${GPU_ID} is idle..."
GPU_USED="$(ssh "${SSH_TARGET}" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i '${GPU_ID}'")"
if [[ "${GPU_USED:-999999}" -ge 1024 ]]; then
    echo "Error: GPU ${GPU_ID} is not idle (${GPU_USED} MiB used)."
    exit 3
fi

echo "[3/3] Resuming ${CFG_BASENAME} on ${SERVER} GPU ${GPU_ID}..."
ssh "${SSH_TARGET}" "
    set -e
    mkdir -p '${REMOTE_CODE_ROOT}/logs'
    cd '${REMOTE_CODE_ROOT}'
    export PYTHONPATH='${REMOTE_CODE_ROOT}:${REMOTE_CODE_ROOT}/ipdn/lib:${SITE_PACKAGES}:${POINTNET_EGG}'
    nohup '${PYTHON}' -S tools/train.py '${CFG}' --num_gpus 1 --gpu_ids '${GPU_ID}' \
        --resume '${RESUME_CKPT}' > '${REMOTE_LOG}' 2>&1 < /dev/null &
    pid=\$!
    echo \"\$pid\" > 'logs/.pid_${CFG_BASENAME}'
    sleep 5
    if ! kill -0 \"\$pid\" 2>/dev/null; then
        tail -n 30 '${REMOTE_LOG}'
        exit 1
    fi
    echo \"PID=\$pid\"
    echo 'Resumed training started'
"

echo "Resumed ${CFG_BASENAME} on ${SERVER} GPU ${GPU_ID}"
echo "Checkpoint: ${RESUME_CKPT}"
echo "Log: ${REMOTE_LOG}"
