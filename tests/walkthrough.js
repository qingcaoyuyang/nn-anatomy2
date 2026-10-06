/**
 * UI 走查驱动（Task 12）：一条命令跑完走查清单。
 *
 *   node tests/walkthrough.js
 *
 * 步骤：
 *  1. walkthrough.html（main）：空模型/空数据集/训练中按钮态/暂停继续/NaN 恢复
 *  2. walkthrough.html?mode=rm + reduced-motion 仿真：舞台动画瞬达
 *  3. walkthrough.html?mode=touch + dispatchTouchEvent：在画板上画一笔 → 预览有墨
 *  4. 390×844 viewport：无横向溢出 + 截图
 *  5. lesson stepper：notify 推进 → DOM 状态断言 + 截图
 *
 * 退出码 0=全过，1=有失败。截图落 /tmp（验收用，不进仓库）。
 */
import { spawn } from 'node:child_process';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';

const REPO = new URL('..', import.meta.url).pathname.replace(/\/$/, '');
const CHROME = `${process.env.HOME}/workspace/tools/chrome-headless-shell-linux64/chrome-headless-shell`;

function freePort() {
  return new Promise((resolve) => {
    const s = createServer();
    s.listen(0, '127.0.0.1', () => {
      const p = s.address().port;
      s.close(() => resolve(p));
    });
  });
}

class Cdp {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.id = 0;
    this.pending = new Map();
    this.events = [];
  }
  async connect() {
    this.ws = new WebSocket(this.wsUrl, { maxPayload: 256 * 1024 * 1024 });
    this.ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
      } else if (msg.method) {
        this.events.push(msg);
      }
    });
    await new Promise((res, rej) => {
      this.ws.addEventListener('open', res, { once: true });
      this.ws.addEventListener('error', rej, { once: true });
    });
  }
  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.id;
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async evaluate(expr) {
    const r = await this.send('Runtime.evaluate', { expression: expr, returnByValue: true });
    if (r.exceptionDetails) throw new Error('evaluate 异常：' + JSON.stringify(r.exceptionDetails).slice(0, 300));
    return r.result.value;
  }
  async title() {
    return this.evaluate('document.title');
  }
  async waitTitle(pred, timeoutMs, label) {
    const t0 = Date.now();
    let lastLog = 0;
    while (Date.now() - t0 < timeoutMs) {
      const t = await this.title().catch((e) => `ERR:${e.message.slice(0, 80)}`);
      if (t && pred(t)) return t;
      if (Date.now() - lastLog > 30000) {
        lastLog = Date.now();
        const href = await this.evaluate('location.href').catch(() => '?');
        console.log(`[waitTitle ${label}] ${(Date.now() - t0) / 1000 | 0}s title=`, String(t).slice(0, 120), 'href=', String(href).slice(0, 120));
      }
      await delay(500);
    }
    throw new Error(`等待 title 超时：${label}`);
  }
  async screenshot(path) {
    const shot = await this.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(path, Buffer.from(shot.data, 'base64'));
  }
  close() { try { this.ws.close(); } catch { /* ignore */ } }
}

async function launchChrome(dbgPort, profile) {
  const env = { ...process.env };
  for (const k of ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy']) delete env[k];
  const proc = spawn(CHROME, [
    '--headless', '--no-sandbox', '--disable-gpu', '--no-proxy-server',
    `--remote-debugging-port=${dbgPort}`, `--user-data-dir=${profile}`, 'about:blank',
  ], { env, stdio: 'ignore' });
  // 等待 DevTools 就绪
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

async function newPage(dbgPort) {
  const targets = await (await fetch(`http://127.0.0.1:${dbgPort}/json/list`)).json();
  const page = targets.find((t) => t.type === 'page');
  if (!page) throw new Error('找不到 page target');
  const cdp = new Cdp(page.webSocketDebuggerUrl);
  await cdp.connect();
  await cdp.send('Runtime.enable');
  return cdp;
}

const summary = { pass: 0, fail: 0, items: [] };
function check(cond, name, extra = '') {
  if (cond) { summary.pass++; summary.items.push(`ok   ${name}`); }
  else { summary.fail++; summary.items.push(`FAIL ${name} ${extra}`); }
}

async function main() {
  const httpPort = await freePort();
  const dbgPort = await freePort();
  const profile = mkdtempSync(join(tmpdir(), 'wt-prof-'));
  const server = spawn('python3', ['-m', 'http.server', String(httpPort)], { cwd: REPO, stdio: 'ignore' });
  await delay(1000);
  const chrome = await launchChrome(dbgPort, profile);
  const base = `http://127.0.0.1:${httpPort}`;
  let cdp = null;

  try {
    // —— 1. 主走查 ——
    cdp = await newPage(dbgPort);
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });
    await cdp.send('Page.navigate', { url: `${base}/tests/browser/walkthrough.html` });
    const t1 = await cdp.waitTitle(
      (t) => { try { const j = JSON.parse(t); return j.mode === 'main' && (j.pass + j.fail) > 0; } catch { return false; } },
      300000, 'walkthrough main');
    const j1 = JSON.parse(t1);
    check(j1.fail === 0, `walkthrough 主流程 ${j1.pass} 项`, JSON.stringify(j1.errors).slice(0, 400));
    await cdp.screenshot('/tmp/wt-main.png');

    // —— 5. lesson stepper 推进（在同一页上；先 reset，清掉主流程已触发的事件） ——
    const lessonStates = await cdp.evaluate(`(() => {
      const L = window.__nnanatomy.lesson;
      L.reset();
      const cls = () => Array.from(document.querySelectorAll('.lesson-step')).map(e => e.className);
      const before = cls();
      L.notify('drawn'); L.notify('inference');
      const mid = cls();
      L.notify('trained'); L.notify('evaluated');
      const after = cls();
      const cheer = !document.querySelector('#lesson-cheer').classList.contains('hidden');
      return { before, mid, after, cheer };
    })()`);
    check(lessonStates.before[0].includes('active'), 'lesson 初始：阶段1高亮');
    check(lessonStates.mid[0].includes('done') && lessonStates.mid[1].includes('done') && lessonStates.mid[2].includes('active'),
      'lesson 推进：完成2步后阶段3高亮');
    check(lessonStates.after.every((c) => c.includes('done')) && lessonStates.cheer, 'lesson 全完成：打勾+祝贺');
    check(lessonStates.before[0].includes('active') && !lessonStates.before[0].includes('done'), 'lesson reset 回到初始');
    await cdp.evaluate(`(() => { const L = window.__nnanatomy.lesson; L.notify('drawn'); L.notify('inference'); L.notify('trained'); })()`);
    await delay(300);
    await cdp.screenshot('/tmp/wt-lesson.png');
    cdp.close();

    // —— 2. reduced-motion ——
    cdp = await newPage(dbgPort);
    await cdp.send('Emulation.setEmulatedMedia', {
      features: [{ name: 'prefers-reduced-motion', value: 'reduce' }],
    });
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });
    await cdp.send('Page.navigate', { url: `${base}/tests/browser/walkthrough.html?mode=rm` });
    const t2 = await cdp.waitTitle(
      (t) => { try { const j = JSON.parse(t); return j.mode === 'rm' && (j.pass + j.fail) > 0; } catch { return false; } },
      120000, 'walkthrough rm');
    const j2 = JSON.parse(t2);
    check(j2.fail === 0, `reduced-motion ${j2.pass} 项`, JSON.stringify(j2.errors).slice(0, 400));
    await cdp.screenshot('/tmp/wt-rm.png');
    cdp.close();

    // —— 3. touch ——
    cdp = await newPage(dbgPort);
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1400, height: 900, deviceScaleFactor: 1, mobile: true });
    await cdp.send('Page.navigate', { url: `${base}/tests/browser/walkthrough.html?mode=touch` });
    await cdp.waitTitle(
      (t) => { try { return JSON.parse(t).ready === true; } catch { return false; } },
      60000, 'walkthrough touch ready');
    const rect = await cdp.evaluate(`(() => { const r = document.getElementById('board').getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; })()`);
    const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2;
    const touch = (type, points) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: points });
    await touch('touchStart', [{ x: cx - 60, y: cy, id: 1 }]);
    for (let i = 1; i <= 8; i++) {
      await touch('touchMove', [{ x: cx - 60 + i * 15, y: cy, id: 1 }]);
      await delay(30);
    }
    await touch('touchEnd', []);
    await delay(500);
    const ink = await cdp.evaluate(`(() => {
      const d = window.__nnanatomy.drawing.getPixels60();
      return Array.from(d).filter((v) => v > 0.05).length;
    })()`);
    check(ink > 0, `触屏画一笔 → 预览有墨（${ink} 像素）`);
    await cdp.screenshot('/tmp/wt-touch.png');
    cdp.close();

    // —— 4. 移动端 390px ——
    cdp = await newPage(dbgPort);
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
    await cdp.send('Page.navigate', { url: `${base}/tests/browser/walkthrough.html?mode=touch` });
    await cdp.waitTitle(
      (t) => { try { return JSON.parse(t).ready === true; } catch { return false; } },
      60000, 'mobile ready');
    await delay(1000);
    const overflow = await cdp.evaluate('document.documentElement.scrollWidth');
    check(overflow <= 390, `390px 无横向溢出（scrollWidth=${overflow}）`);
    await cdp.screenshot('/tmp/wt-mobile.png');
    cdp.close();
  } finally {
    try { chrome.kill('SIGKILL'); } catch { /* ignore */ }
    try { server.kill('SIGKILL'); } catch { /* ignore */ }
  }

  console.log(summary.items.join('\n'));
  console.log(`\nwalkthrough: pass=${summary.pass} fail=${summary.fail}`);
  process.exit(summary.fail === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error('walkthrough 驱动异常：', e.message);
  process.exit(1);
});
