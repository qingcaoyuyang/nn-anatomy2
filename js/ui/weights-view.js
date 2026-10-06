/**
 * 权重视图：modal 展示当前网络的全部权重。
 *
 * ES module，零 npm 依赖。样式通过一次性注入的 <style>（类名前缀 wv-）
 * 自包含，不依赖外部 CSS 文件。
 *
 * # 接口（见 docs/rebuild-plan.md「接口约定」）
 * initWeightsView(api) → { open() }；api.getModel() 返回 { net }。
 * net = { arch: { inputs: 3600, hidden: 32, outputs: 10 },
 *         w1: Float64Array(32*3600) 行主序 w1[h*3600+i],
 *         b1, w2: Float64Array(10*32) 行主序 w2[o*32+h], b2 }
 * （布局见 js/network.js：createNetwork 按 h 行主序填充 w1，
 * forward 用 row = h*nIn / row = o*nH 取行。）
 *
 * # 展示
 * - W1：32 张 60×60 热力图（canvas 60×60 像素，CSS 约 48×48，pixelated 放大），
 *   蓝=正 / 红=负 / 白≈0，按全局 max|w1| 归一化，标注 h0..h31。
 * - W2：10×32 矩阵热力图（一张 canvas；行=输出类 0..9，列=h0..h31），
 *   按全局 max|w2| 归一化。
 * - 右上角 × 关闭；点击遮罩关闭；Esc 关闭。
 */

const IN = 3600;   // 60×60
const NH = 32;
const NOUT = 10;
const GS = 60;     // 热力图边长（像素）

let styleInjected = false;
function ensureStyle() {
  if (styleInjected) return;
  styleInjected = true;
  const st = document.createElement('style');
  st.textContent = `
.wv-overlay{position:fixed;inset:0;background:rgba(15,23,42,.55);display:flex;
  align-items:center;justify-content:center;z-index:9999;padding:16px;box-sizing:border-box;}
.wv-box{background:#fff;border-radius:12px;max-width:780px;width:100%;max-height:92vh;
  overflow:auto;padding:18px 20px;position:relative;box-sizing:border-box;
  box-shadow:0 20px 60px rgba(0,0,0,.35);font-size:13px;color:#0f172a;}
.wv-close{position:absolute;top:10px;right:12px;width:30px;height:30px;border:none;
  background:#f1f5f9;border-radius:15px;font-size:16px;cursor:pointer;color:#475569;line-height:1;}
.wv-close:hover{background:#e2e8f0;}
.wv-title{font-size:16px;font-weight:800;margin:0 40px 6px 0;}
.wv-note{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:8px 10px;
  font-size:12.5px;line-height:1.7;color:#92400e;margin:8px 0 12px;}
.wv-sec{font-size:13px;font-weight:700;margin:14px 0 6px;}
.wv-sub{font-size:12px;color:#64748b;margin-bottom:8px;}
.wv-grid{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;}
.wv-cell{text-align:center;}
.wv-cell canvas{width:48px;height:48px;image-rendering:pixelated;border:1px solid #e2e8f0;
  border-radius:4px;display:block;margin:0 auto;background:#fff;}
.wv-cell div{font-size:11px;color:#475569;margin-top:2px;font-family:ui-monospace,Menlo,monospace;}
.wv-w2wrap{display:flex;gap:6px;align-items:flex-start;}
.wv-rows{display:flex;flex-direction:column;}
.wv-rows div{height:14px;line-height:14px;font-size:11px;color:#475569;text-align:right;
  padding-right:4px;font-family:ui-monospace,Menlo,monospace;}
.wv-cols{display:flex;margin-bottom:2px;}
.wv-cols div{width:14px;font-size:10px;color:#94a3b8;text-align:center;
  font-family:ui-monospace,Menlo,monospace;}
.wv-w2wrap canvas{image-rendering:pixelated;border:1px solid #e2e8f0;border-radius:4px;background:#fff;}
.wv-legend{display:flex;gap:14px;align-items:center;font-size:12px;color:#475569;margin-top:10px;flex-wrap:wrap;}
.wv-sw{display:inline-block;width:12px;height:12px;border-radius:3px;border:1px solid #cbd5e1;
  vertical-align:-1px;margin-right:4px;}
`;
  document.head.appendChild(st);
}

/** 发散配色：t∈[-1,1]，正→蓝，负→红，0→白。 */
function heat(t) {
  let r, g, bl;
  if (t >= 0) { r = g = Math.round(255 * (1 - t)); bl = 255; }
  else { r = 255; g = bl = Math.round(255 * (1 + t)); }
  return [r, g, bl];
}

function drawHeat(canvas, values, w, h, maxAbs) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  const d = img.data;
  const scale = maxAbs > 0 ? 1 / maxAbs : 0;
  for (let i = 0; i < w * h; i++) {
    let t = values[i] * scale;
    if (t > 1) t = 1; else if (t < -1) t = -1;
    const [r, g, b] = heat(t);
    d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function maxAbsOf(arr) {
  let m = 0;
  for (let i = 0; i < arr.length; i++) {
    const a = Math.abs(arr[i]);
    if (a > m) m = a;
  }
  return m;
}

function el(tag, cls, parent, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

export function initWeightsView(api) {
  function open() {
    ensureStyle();
    const model = api && typeof api.getModel === 'function' ? api.getModel() : null;
    const net = model && model.net;

    const overlay = el('div', 'wv-overlay', document.body);
    const box = el('div', 'wv-box', overlay);
    const closeBtn = el('button', 'wv-close', box, '×');
    closeBtn.setAttribute('aria-label', '关闭');

    const close = () => {
      document.removeEventListener('keydown', onKey);
      overlay.remove();
    };
    const onKey = (ev) => { if (ev.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    closeBtn.addEventListener('click', (ev) => { ev.stopPropagation(); close(); });
    overlay.addEventListener('click', (ev) => { if (ev.target === overlay) close(); });

    el('div', 'wv-title', box, '模型权重');
    el('div', 'wv-note', box,
      '新建模型的权重是随机小值（≈0），模型还什么都不会——训练会改变它们。' +
      '下面每一张热力图都是真实权重：蓝色=正权重（"看到墨就兴奋"），红色=负权重（"看到墨就抑制"）。');

    if (!net || !net.w1 || !net.w2 ||
        net.w1.length !== NH * IN || net.w2.length !== NOUT * NH) {
      el('div', 'wv-sub', box, '模型尚未创建（或权重形状不对），没有可展示的权重。');
      return;
    }

    // ---- W1：32 张 60×60 热力图 ----
    el('div', 'wv-sec', box, `W₁：32 个隐藏节点各自的 60×60 权重（w₁[h·3600+i]）`);
    const max1 = maxAbsOf(net.w1);
    el('div', 'wv-sub', box,
      `按全局最大 |w₁| = ${max1.toExponential(2)} 归一化；白≈0。每张图是一个隐藏节点"在找什么样的笔画"。`);
    const grid = el('div', 'wv-grid', box);
    for (let h = 0; h < NH; h++) {
      const cell = el('div', 'wv-cell', grid);
      const cv = el('canvas', '', cell);
      drawHeat(cv, net.w1.slice(h * IN, (h + 1) * IN), GS, GS, max1); // slice：普通 Array 与 TypedArray 通用
      el('div', '', cell, `h${h}`);
    }

    // ---- W2：10×32 矩阵 ----
    el('div', 'wv-sec', box, 'W₂：10×32（行=输出类 0..9，列=隐藏节点 h0..h31）');
    const max2 = maxAbsOf(net.w2);
    el('div', 'wv-sub', box,
      `按全局最大 |w₂| = ${max2.toExponential(2)} 归一化；第 o 行就是"判为数字 o 时，每个隐藏节点的投票权重"。`);
    // 列标
    const colWrap = el('div', '', box);
    colWrap.style.cssText = 'display:flex;margin-left:26px;';
    const cols = el('div', 'wv-cols', colWrap);
    for (let h = 0; h < NH; h++) {
      el('div', '', cols, h % 8 === 0 ? `h${h}` : '');
    }
    const wrap = el('div', 'wv-w2wrap', box);
    const rows = el('div', 'wv-rows', wrap);
    for (let o = 0; o < NOUT; o++) el('div', '', rows, String(o));
    const cv2 = el('canvas', '', wrap);
    const CELL = 14;
    cv2.style.width = `${NH * CELL}px`;
    cv2.style.height = `${NOUT * CELL}px`;
    // 把 w2[o*32+h] 排成 (列=h, 行=o) 的像素矩阵
    const pix = new Float64Array(NH * NOUT);
    for (let o = 0; o < NOUT; o++) {
      for (let h = 0; h < NH; h++) pix[o * NH + h] = net.w2[o * NH + h];
    }
    drawHeat(cv2, pix, NH, NOUT, max2);

    // 图例
    const legend = el('div', 'wv-legend', box);
    const sw = (color, label) => {
      const s = el('span', '', legend);
      const i = el('span', 'wv-sw', s);
      i.style.background = color;
      s.appendChild(document.createTextNode(label));
    };
    sw('#3b82f6', '正权重');
    sw('#ef4444', '负权重');
    sw('#ffffff', '≈ 0');
  }

  return { open };
}
