#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVERS_YAML="$(cygpath -m "$PROJECT_ROOT/SERVERS.yaml")"
SERVER="${1:?Usage: sync.sh <server>}"

read_value() {
    python -c "import yaml; d=yaml.safe_load(open(r'$SERVERS_YAML', encoding='utf-8'))['servers']['$SERVER']; print(d['$1'])"
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
    ssh -o BatchMode=yes "$TARGET" "printf '%s\n' '$HEAD' > '$REMOTE_ROOT/.codex_source_commit'"
    echo "Verified no AdaRES changes; advanced source marker on $SERVER to $HEAD"
    exit 0
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

MISSING=()
for file in "${FILES[@]}"; do
    expected="$(git -C "$PROJECT_ROOT" -c core.autocrlf=false archive "$HEAD:AdaRES" "$file" | tar -xO "$file" | sha256sum | cut -d ' ' -f 1)"
    if ssh -o BatchMode=yes "$TARGET" "test -e '$REMOTE_ROOT/$file'"; then
        actual="$(ssh -o BatchMode=yes "$TARGET" "sha256sum '$REMOTE_ROOT/$file'" | cut -d ' ' -f 1)"
        if [[ "$expected" != "$actual" ]]; then
            echo "Refusing mismatched existing remote file: $file" >&2
            exit 1
        fi
    else
        MISSING+=("$file")
    fi
done

if (( ${#MISSING[@]} > 0 )); then
    git -C "$PROJECT_ROOT" -c core.autocrlf=false archive "$HEAD:AdaRES" "${MISSING[@]}" |
        ssh -o BatchMode=yes "$TARGET" "tar --warning=no-timestamp -C '$REMOTE_ROOT' -xkf -"
fi
for file in "${FILES[@]}"; do
    expected="$(git -C "$PROJECT_ROOT" -c core.autocrlf=false archive "$HEAD:AdaRES" "$file" | tar -xO "$file" | sha256sum | cut -d ' ' -f 1)"
    actual="$(ssh -o BatchMode=yes "$TARGET" "sha256sum '$REMOTE_ROOT/$file'" | cut -d ' ' -f 1)"
    [[ "$expected" == "$actual" ]] || { echo "Remote hash mismatch: $file" >&2; exit 1; }
done
ssh -o BatchMode=yes "$TARGET" "printf '%s\n' '$HEAD' > '$REMOTE_ROOT/.codex_source_commit'"
echo "Verified ${#FILES[@]} additive AdaRES files on $SERVER at $HEAD"
