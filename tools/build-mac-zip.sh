#!/usr/bin/env bash
# 打 Mac 包：git archive + 注入 version.json（版本号徽标/更新日志用）。
# 用法：bash tools/build-mac-zip.sh [输出路径] [分支]
# 默认：/home/hatch/workspace/nn-anatomy2-mac.zip，分支 main
set -euo pipefail
OUT="${1:-/home/hatch/workspace/nn-anatomy2-mac.zip}"
BRANCH="${2:-main}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TMPV="$(mktemp /tmp/nn-anatomy2-version.XXXXXX.json)"

python3 "$REPO/tools/gen-version.py" "$TMPV" "$BRANCH"
git -C "$REPO" archive --format=zip -o "$OUT" "$BRANCH"
python3 - "$OUT" "$TMPV" <<'EOF'
import sys, zipfile
out, ver = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "a", zipfile.ZIP_DEFLATED) as z:
    z.write(ver, "version.json")
print("injected version.json into", out)
EOF
rm -f "$TMPV"
ls -la "$OUT"
