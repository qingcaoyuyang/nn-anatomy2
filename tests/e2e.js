/**
 * 端到端验收驱动（Task 12）：一条命令跑完全链路。
 *
 *   node tests/e2e.js
 *
 * 链路（tests/browser/e2e.html，MemoryBackend —— headless 下 OPFS/IndexedDB 不可用）：
 *   导入 3300 张示例集 → newModel('e2e', seed=23) → batchTrain(15, Adam 0.001)
 *   → 断言测试集 acc ≥ 阈值 → save/load round-trip（权重 bit-identical，acc 不变）
 *
 * 确定性：init seed=23 + app 首轮 runSeed=1 → node 实测见 docs/rebuild-plan.md。
 * 截图：/tmp/e2e-import.png /tmp/e2e-trained.png /tmp/e2e-confusion.png
 * 退出码 0=通过，1=失败。
 */
import { spawn } from 'node:child_process';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';

const REPO = new URL('..', import.meta.url).pathname.replace(/\/$/, '');
const CHROME = `${process.env.HOME}/workspace/tools/chrome-headless-shell-linux64/chrome-headless-shell`;
const TIMEOUT_MS = 12 * 60 * 1000;

function freePort() {
  return new Promise((resolve) => {
    const s = createServer();
    s.listen(0, '127.0.0.1', () => {
      const p = s.address().port;
      s.close(() => resolve(p));
    });
  });
}

async function launchChrome(dbgPort, profile) {
  const env = { ...process.env };
  for (const k of ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy']) delete env[k];
  const proc = spawn(CHROME, [
    '--headless', '--no-sandbox', '--disable-gpu', '--no-proxy-server',
    `--remote-debugging-port=${dbgPort}`, `--user-data-dir=${profile}`, 'about:blank',
  ], { env, stdio: 'ignore' });
  const t0 = Date.now();
  while (Date.now() - t0 < 30000) {
    try {
      const r = await fetch(`http://127.0.0.1:${dbgPort}/json/list`);
      if (r.ok) return proc;
    } catch { /* retry */ }
    await delay(300);
  }
  throw new Error('chrome 启动超时');
}

async function main() {
  const httpPort = await freePort();
  const dbgPort = await freePort();
  const profile = mkdtempSync(join(tmpdir(), 'e2e-prof-'));
  const server = spawn('python3', ['-m', 'http.server', String(httpPort)], { cwd: REPO, stdio: 'ignore' });
  await delay(1000);
  const chrome = await launchChrome(dbgPort, profile);

  let ws = null;
  let id = 0;
  const pending = new Map();
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const mid = ++id;
    pending.set(mid, { resolve, reject });
    ws.send(JSON.stringify({ id: mid, method, params }));
  });
  const evaluate = async (expr) => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true });
    if (r.exceptionDetails) throw new Error('evaluate 异常：' + JSON.stringify(r.exceptionDetails).slice(0, 300));
    return r.result.value;
  };
  const screenshot = async (path) => {
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(path, Buffer.from(shot.data, 'base64'));
    console.log('截图已保存：', path);
  };

  try {
    const targets = await (await fetch(`http://127.0.0.1:${dbgPort}/json/list`)).json();
    const page = targets.find((t) => t.type === 'page');
    if (!page) throw new Error('找不到 page target');
    ws = new WebSocket(page.webSocketDebuggerUrl, { maxPayload: 256 * 1024 * 1024 });
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && pending.has(msg.id)) {
        const { resolve, reject } = pending.get(msg.id);
        pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
      }
    });
    await new Promise((res, rej) => {
      ws.addEventListener('open', res, { once: true });
      ws.addEventListener('error', rej, { once: true });
    });
    await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });
    await send('Page.navigate', { url: `http://127.0.0.1:${httpPort}/tests/browser/e2e.html` });

    const seen = new Set();
    const t0 = Date.now();
    let final = null;
    while (Date.now() - t0 < TIMEOUT_MS) {
      await delay(2000);
      const title = await evaluate('document.title').catch(() => '');
      let j = null;
      try { j = JSON.parse(title); } catch { /* not ready */ }
      if (!j || !j.phase) continue;
      const phases = Array.isArray(j.phases) ? j.phases : [j.phase];
      if (phases.includes('import-done') && !seen.has('import-done')) {
        seen.add('import-done');
        await screenshot('/tmp/e2e-import.png');
      }
      if (phases.includes('train-done') && !seen.has('train-done')) {
        seen.add('train-done');
        await screenshot('/tmp/e2e-trained.png');
        // 滚到混淆矩阵再截一张
        await evaluate(`document.getElementById('confusion').scrollIntoView({block:'center'})`);
        await delay(800);
        await screenshot('/tmp/e2e-confusion.png');
      }
      if (j.phase === 'done') { final = j; break; }
    }
    if (!final) throw new Error('e2e 超时：页面未到达 done');

    console.log('e2e 结果：', JSON.stringify({ pass: final.pass, fail: final.fail, acc: final.acc, errors: final.errors }));
    if (final.fail !== 0) {
      console.error('E2E 失败');
      process.exitCode = 1;
    } else {
      console.log('E2E 通过');
    }
  } catch (e) {
    console.error('e2e 驱动异常：', e.message);
    process.exitCode = 1;
  } finally {
    try { ws && ws.close(); } catch { /* ignore */ }
    try { chrome.kill('SIGKILL'); } catch { /* ignore */ }
    try { server.kill('SIGKILL'); } catch { /* ignore */ }
  }
}

main();
