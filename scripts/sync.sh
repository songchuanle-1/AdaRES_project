#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVER="${1:?Usage: sync.sh <server>}"

read_value() {
    python -c "import yaml; d=yaml.safe_load(open(r'$PROJECT_ROOT/SERVERS.yaml', encoding='utf-8'))['servers']['$SERVER']; print(d['$1'])"
}

HOST="$(read_value host)"
USER="$(read_value user)"
REMOTE_ROOT="$(read_value remote_code_root)"
TARGET="${USER}@${HOST}"
HEAD="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
BASE="$(ssh -o BatchMode=yes "$TARGET" "cat '$REMOTE_ROOT/.codex_source_commit'")"

if [[ "$BASE" == "$HEAD" ]]; then
    exit 0
fi
if ! git -C "$PROJECT_ROOT" cat-file -e "$BASE^{commit}" ||
   ! git -C "$PROJECT_ROOT" merge-base --is-ancestor "$BASE" "$HEAD"; then
    echo "Remote marker is not a local ancestor of HEAD" >&2
    exit 1
fi
mapfile -t CHANGES < <(git -C "$PROJECT_ROOT" diff --name-status "$BASE" "$HEAD" -- AdaRES)
if (( ${#CHANGES[@]} == 0 )); then
    echo "Refusing marker-only sync without AdaRES additions" >&2
    exit 1
fi

FILES=()
for change in "${CHANGES[@]}"; do
    status="${change%%$'\t'*}"
    path="${change#*$'\t'}"
    if [[ "$status" != "A" || "$path" != AdaRES/* ]]; then
        echo "Refusing non-additive sync: $change" >&2
        exit 1
    fi
    FILES+=("${path#AdaRES/}")
done

for file in "${FILES[@]}"; do
    if ssh -o BatchMode=yes "$TARGET" "test -e '$REMOTE_ROOT/$file'"; then
        echo "Refusing to overwrite remote file: $file" >&2
        exit 1
    fi
done

git -C "$PROJECT_ROOT" archive "$HEAD:AdaRES" "${FILES[@]}" |
    ssh -o BatchMode=yes "$TARGET" "tar -C '$REMOTE_ROOT' -xkf -"

for file in "${FILES[@]}"; do
    expected="$(git -C "$PROJECT_ROOT" show "$HEAD:AdaRES/$file" | sha256sum | cut -d ' ' -f 1)"
    actual="$(ssh -o BatchMode=yes "$TARGET" "sha256sum '$REMOTE_ROOT/$file'" | cut -d ' ' -f 1)"
    if [[ "$expected" != "$actual" ]]; then
        echo "Remote hash mismatch: $file" >&2
        exit 1
    fi
done
ssh -o BatchMode=yes "$TARGET" "printf '%s\n' '$HEAD' > '$REMOTE_ROOT/.codex_source_commit'"
echo "Verified ${#FILES[@]} additive AdaRES files on $SERVER at $HEAD"
