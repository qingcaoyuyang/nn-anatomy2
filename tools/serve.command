#!/bin/bash
# nn-anatomy2 本地启动器（Mac 双击运行）。
# 起一个本地 HTTP 服务并用 Chrome 打开。关闭本窗口即停止服务。
set -u
cd "$(dirname "$0")/.." || { echo "找不到项目目录"; exit 1; }

PORT="${1:-8080}"

# 端口被占 → 依次试 8081..8090，实在不行报错退出
# 注：用 SO_REUSEADDR 做探测，与 http.server 自身行为一致；
# 否则刚被杀掉的服务留下的 TIME_WAIT 连接会造成"幽灵占用"误判。
# 返回值：0 = 端口空闲，1 = 被占用（直接取 python 退出的状态，不取反）。
port_free() {
  python3 -c "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', $1))" 2>/dev/null
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

# 清理上次异常退出留下的孤儿服务（比如直接关窗口没走 trap 的）
if [ -f /tmp/nn-anatomy2-serve.pid ]; then
  OLDSRV=$(cat /tmp/nn-anatomy2-serve.pid 2>/dev/null)
  if [ -n "$OLDSRV" ] && kill -0 "$OLDSRV" 2>/dev/null; then
    kill "$OLDSRV" 2>/dev/null
    sleep 0.5
  fi
  rm -f /tmp/nn-anatomy2-serve.pid
fi

python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
SRV=$!
echo "$SRV" > /tmp/nn-anatomy2-serve.pid
trap 'kill $SRV 2>/dev/null; rm -f /tmp/nn-anatomy2-serve.pid' EXIT
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
