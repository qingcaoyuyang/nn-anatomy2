#!/bin/bash
# nn-anatomy2 本地启动器（Mac 双击运行）。
# 起一个本地 HTTP 服务并用 Chrome 打开。关闭本窗口即停止服务。
set -u
cd "$(dirname "$0")/.." || { echo "找不到项目目录"; exit 1; }

PORT="${1:-8080}"

# 端口被占 → 依次试 8081..8090，实在不行报错退出
port_free() {
  ! python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', $1))" 2>/dev/null
}
if ! port_free "$PORT"; then
  for p in $(seq 8081 8090); do
    if port_free "$p"; then PORT="$p"; break; fi
  done
fi
if ! port_free "$PORT"; then
  echo "端口 8080-8090 都被占用，请先关掉占用端口的程序再试。"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "找不到 python3，请先安装 Python 3。"
  exit 1
fi

python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 1
if ! kill -0 $SRV 2>/dev/null; then
  echo "本地服务启动失败。"
  exit 1
fi

URL="http://127.0.0.1:$PORT/index.html"
echo "已启动：$URL（关闭本窗口即停止）"
if ! open -a "Google Chrome" "$URL" 2>/dev/null; then
  echo "找不到 Google Chrome，已为你打开默认浏览器。"
  open "$URL" 2>/dev/null || echo "请手动打开：$URL"
fi
wait $SRV
