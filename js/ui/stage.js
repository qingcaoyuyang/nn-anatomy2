/**
 * 神经网络舞台：3600→32→10 MLP 的实时可视化（Canvas 2D）。
 *
 * ES module，零 npm 依赖，不碰页面 DOM（只用传入的 canvas）。布局（680×440）：
 *   左：6×6 分块（每块 = 10×10 像素的平均墨量，灰度）｜
 *   中：32 隐藏节点（2 列×16 行）｜
 *   右：10 条输出概率柱。
 * 连线：input→hidden 每隐藏节点只画 |w1| 最大的 top-8（共 256 条，
 *       起点为该像素在其分块内的近似位置）；
 *       hidden→output 32×10 全画（320 条）；线宽 ∝ |w|，蓝=正 / 红=负。
 *
 * 左下角恒画诚实标注：36 块只是帮人眼看清墨在哪，
 * 网络实际把 3600 像素一次性全连到每个隐藏节点。
 *
 * 推理动画：粒子沿边前向流动（分两段），隐藏节点辉光渐亮，
 *           输出柱从均匀 10% 收敛到 p。
 * 训练动画：红色误差粒子反向流动 + |Δw| top-24 的边发扩张脉冲 +
 *           输出柱 before（灰）/ after（蓝）双色对比。
 *
 * # 无障碍
 * matchMedia('(prefers-reduced-motion: reduce)') 为真时，所有动画退化为
 * 瞬达：直接渲染终态，Promise 立即 resolve。
 *
 * # 重入
 * 动画进行中再次调用 playInference / playTrainingStep → 返回进行中的
 * Promise（不排队、不打断）。动画结束后状态落在终态。
 * （参数校验先于重入检查：非法输入直接抛错。）
 *
 * # 状态语义（教学停留用）
 * - playTrainingStep 结束后，输出柱保持 before（灰）/ after（蓝）双色对比态，
 *   直到下一次 playInference / render；
 * - playInference 开始时清除热力图/分块详情与对比柱，回到"当前输入"态。
 *
 * # 点击交互
 * - 点击隐藏节点 → 左侧切换显示该节点 3600 权重的蓝红热力图；
 *   再点该节点 → 恢复分块显示；
 * - 点击分块 → 左侧显示该块 100 像素小图 + 该块 Σ|w| 最大的 top-3 隐藏节点；
 *   再点该块 → 恢复分块显示；
 * - 点击空白 → 恢复分块显示；
 * - 热力图模式下点分块 → 退出热力图并显示该分块详情。
 */

import { inferWithTrace } from '../trainer.js';
import { createParticleSystem, MAX_PARTICLES } from './particles.js';

const TAU = Math.PI * 2;

/* —— 逻辑画布尺寸 —— */
const W = 680;
const H = 440;

/* —— 布局 —— */
const BLOCKS = { x: 16, y: 56, cell: 30, n: 6, px: 10 }; // 6×6 块，每块 10×10 像素 → 180×180
const HIDDEN = { x0: 330, colGap: 56, y0: 56, y1: 388, r: 10, cols: 2, rows: 16 }; // 32 节点
const OUT = { labelX: 452, barX: 478, barMax: 120, y0: 64, rowH: 32, n: 10 };

const HONEST = '36 块只是帮人眼看清墨在哪；网络实际把 3600 像素一次性全连到每个隐藏节点';

/* —— 配色 —— */
const POS = '37,99,235'; // 蓝（正权重）
const NEG = '220,38,38'; // 红（负权重）
const INK = '#1c1917';
const MUTED = '#78716c';
const GRID_BG = '#fafaf9';

const FONT_NUM = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
const FONT_NUM_SM = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
const FONT_LABEL = '12px "Noto Sans SC", sans-serif';
const FONT_NOTE = '11px "Noto Sans SC", sans-serif';

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (t) => {
  const c = clamp01(t);
  return c * c * (3 - 2 * c);
};

/** 隐藏节点 h 的圆心坐标（2 列×16 行：h = col*16 + row）。 */
function hiddenXY(h) {
  const col = (h / HIDDEN.rows) | 0;
  const row = h % HIDDEN.rows;
  return [
    HIDDEN.x0 + col * HIDDEN.colGap,
    HIDDEN.y0 + (row * (HIDDEN.y1 - HIDDEN.y0)) / (HIDDEN.rows - 1),
  ];
}

/** 输出行 o 的中心 y。 */
function outY(o) {
  return OUT.y0 + o * OUT.rowH + OUT.rowH / 2;
}

/**
 * 输入像素 i（0..3599，60×60 行主序）在其所属分块内的近似位置。
 * 分块 cell=30px 对应 10×10 像素 → 每像素 3px。
 */
function pixelInBlockXY(i) {
  const px = i % 60;
  const py = (i / 60) | 0;
  const bx = (px / BLOCKS.px) | 0;
  const by = (py / BLOCKS.px) | 0;
  const wx = px % BLOCKS.px;
  const wy = py % BLOCKS.px;
  const unit = BLOCKS.cell / BLOCKS.px; // 3
  return [
    BLOCKS.x + bx * BLOCKS.cell + (wx + 0.5) * unit,
    BLOCKS.y + by * BLOCKS.cell + (wy + 0.5) * unit,
  ];
}

/** 在行主序数组 row（起点 off，长 n）中取 |值| 最大的 k 个下标。 */
function topKAbs(arr, off, n, k) {
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => Math.abs(arr[off + b]) - Math.abs(arr[off + a]));
  return idx.slice(0, Math.min(k, n));
}

/**
 * 按当前网络构建边的几何缓存。
 * e1: input→hidden（每隐藏节点 top-8，共 256 条，起点为像素在分块内的近似位置）；
 * e2: hidden→output（32×10 全画）。
 */
function buildEdges(net) {
  const { inputs: nIn, hidden: nH, outputs: nOut } = net.arch;
  const e1 = [];
  let max1 = 1e-9;
  for (let h = 0; h < nH; h++) {
    const top = topKAbs(net.w1, h * nIn, nIn, 8);
    const [hx, hy] = hiddenXY(h);
    for (const i of top) {
      const w = net.w1[h * nIn + i];
      const aw = Math.abs(w);
      if (aw > max1) max1 = aw;
      const [px, py] = pixelInBlockXY(i);
      e1.push({ h, i, x1: px, y1: py, x2: hx, y2: hy, w });
    }
  }
  const e2 = [];
  const e2ByOutput = Array.from({ length: nOut }, () => []);
  let max2 = 1e-9;
  for (let o = 0; o < nOut; o++) {
    for (let h = 0; h < nH; h++) {
      const w = net.w2[o * nH + h];
      const aw = Math.abs(w);
      if (aw > max2) max2 = aw;
      const [hx, hy] = hiddenXY(h);
      const e = { o, h, x1: hx, y1: hy, x2: OUT.barX, y2: outY(o), w };
      e2.push(e);
      e2ByOutput[o].push(e);
    }
  }
  return { e1, e2, e2ByOutput, max1, max2 };
}

/** DPR 自适应：backing store 按 dpr 放大，逻辑坐标恒为 680×440。 */
function fitCanvas(canvas, ctx) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

/**
 * 初始化舞台。
 * @param {HTMLCanvasElement} canvas 中栏 <canvas id="stage">
 * @param {object} net createNetwork 产物（引用；setNet 可替换）
 * @returns {{ setNet, render, playInference, playTrainingStep, highlightNode,
 *   clearHighlight, getLastTrace, lastTrace, _freeze }}
 *   _freeze 仅测试用：冻结当前动画帧（供截图），不对外文档化。
 */
export function initStage(canvas, net) {
  const ctx = canvas.getContext('2d');
  fitCanvas(canvas, ctx);

  const sys = createParticleSystem(MAX_PARTICLES);
  const nH = () => state.net.arch.hidden;
  const nIn = () => state.net.arch.inputs;

  const state = {
    net,
    edges: buildEdges(net),
    x: null, // 当前输入（3600 数组）或 null
    trace: null, // inferWithTrace 结果或 null
    lastTrace: null,
    heatmap: null, // 正在做权重热力图的隐藏节点下标，或 null
    heatColors: null, // { h, colors[3600] } 热力图颜色缓存（避免每帧重算）
    blockDetail: null, // 正在看详情的分块下标（0..35），或 null
    blockInfo: null, // { b, top3:[h,h,h] } 分块详情缓存
    compare: null, // { before: p[10], after: p[10] } 或 null
    pulses: [], // 权重脉冲 [{ x, y, dw, t0, dur }]
    glowT: 1, // 隐藏节点辉光进度（动画中 0→1）
    barT: 1, // 输出柱收敛进度（动画中 0→1）
    now: null, // 当前动画已流逝 ms（脉冲用），非动画时 null
    anim: null, // 进行中的动画 { promise, raf, cancelled, finish, exposed }
    reduced: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  };

  /* ————— 渲染 ————— */

  function drawEdges() {
    const { e1, e2, max1, max2 } = state.edges;
    for (const e of e1) {
      const k = Math.abs(e.w) / max1;
      ctx.strokeStyle = `rgba(${e.w >= 0 ? POS : NEG},${(0.12 + 0.3 * k).toFixed(3)})`;
      ctx.lineWidth = 0.6 + 2.2 * k;
      ctx.beginPath();
      ctx.moveTo(e.x1, e.y1);
      ctx.lineTo(e.x2, e.y2);
      ctx.stroke();
    }
    for (const e of e2) {
      const k = Math.abs(e.w) / max2;
      ctx.strokeStyle = `rgba(${e.w >= 0 ? POS : NEG},${(0.12 + 0.3 * k).toFixed(3)})`;
      ctx.lineWidth = 0.6 + 2.2 * k;
      ctx.beginPath();
      ctx.moveTo(e.x1, e.y1);
      ctx.lineTo(e.x2, e.y2);
      ctx.stroke();
    }
  }

  /** 分块 (bx,by) 的平均墨量（0~1），无输入时为 0。 */
  function blockAvg(bx, by) {
    if (!state.x) return 0;
    let s = 0;
    for (let wy = 0; wy < BLOCKS.px; wy++) {
      const row = (by * BLOCKS.px + wy) * 60 + bx * BLOCKS.px;
      for (let wx = 0; wx < BLOCKS.px; wx++) s += state.x[row + wx];
    }
    return s / (BLOCKS.px * BLOCKS.px);
  }

  /** 隐藏节点 h 的 3600 权重热力图颜色（蓝正/红负），带缓存。 */
  function heatColors(h) {
    const cached = state.heatColors;
    if (cached && cached.h === h && cached.net === state.net) return cached.colors;
    const n = nIn();
    let m = 1e-9;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(state.net.w1[h * n + i]);
      if (a > m) m = a;
    }
    const colors = new Array(n);
    for (let i = 0; i < n; i++) {
      const w = state.net.w1[h * n + i];
      const k = clamp01(Math.abs(w) / m);
      const c = Math.round(245 - 200 * k);
      colors[i] = w >= 0 ? `rgb(${c},${c},245)` : `rgb(245,${c},${c})`;
    }
    state.heatColors = { h, net: state.net, colors };
    return colors;
  }

  /** 左侧输入区：分块 / 权重热力图 / 分块详情 三种模式。 */
  function drawInputArea() {
    const { x, y, cell, n } = BLOCKS;
    const size = n * cell;
    ctx.fillStyle = GRID_BG;
    ctx.fillRect(x, y, size, size);

    if (state.heatmap != null) {
      // 60×60 权重热力图（3px/像素）
      const colors = heatColors(state.heatmap);
      const pc = size / 60;
      for (let i = 0; i < 3600; i++) {
        const px = i % 60;
        const py = (i / 60) | 0;
        ctx.fillStyle = colors[i];
        ctx.fillRect(x + px * pc, y + py * pc, pc + 0.5, pc + 0.5);
      }
    } else if (state.blockDetail != null) {
      // 该块 100 像素小图（18px/像素）+ top-3 隐藏节点
      const b = state.blockDetail;
      const bx = b % n;
      const by = (b / n) | 0;
      const pc = size / BLOCKS.px;
      for (let wy = 0; wy < BLOCKS.px; wy++) {
        for (let wx = 0; wx < BLOCKS.px; wx++) {
          const v = state.x ? clamp01(state.x[(by * BLOCKS.px + wy) * 60 + bx * BLOCKS.px + wx]) : 0;
          const g = Math.round(255 * (1 - v));
          ctx.fillStyle = `rgb(${g},${g},${g})`;
          ctx.fillRect(x + wx * pc, y + wy * pc, pc, pc);
        }
      }
      ctx.strokeStyle = '#e7e5e4';
      ctx.lineWidth = 1;
      for (let k = 0; k <= BLOCKS.px; k++) {
        ctx.beginPath();
        ctx.moveTo(x + k * pc + 0.5, y);
        ctx.lineTo(x + k * pc + 0.5, y + size);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y + k * pc + 0.5);
        ctx.lineTo(x + size, y + k * pc + 0.5);
        ctx.stroke();
      }
      if (state.blockInfo) {
        ctx.fillStyle = MUTED;
        ctx.font = FONT_NUM_SM;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        const t3 = state.blockInfo.top3.map((h) => `h${h}`).join(' ');
        ctx.fillText(`块(${bx},${by}) Σ|w| top-3: ${t3}`, x, y + size + 20);
      }
    } else {
      // 6×6 分块：灰度 = 平均墨量
      for (let by = 0; by < n; by++) {
        for (let bx = 0; bx < n; bx++) {
          const g = Math.round(255 * (1 - blockAvg(bx, by)));
          ctx.fillStyle = `rgb(${g},${g},${g})`;
          ctx.fillRect(x + bx * cell, y + by * cell, cell, cell);
        }
      }
      ctx.strokeStyle = '#e7e5e4';
      ctx.lineWidth = 1;
      for (let k = 0; k <= n; k++) {
        ctx.beginPath();
        ctx.moveTo(x + k * cell + 0.5, y);
        ctx.lineTo(x + k * cell + 0.5, y + size);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y + k * cell + 0.5);
        ctx.lineTo(x + size, y + k * cell + 0.5);
        ctx.stroke();
      }
    }
    ctx.strokeStyle = '#d6d3d1';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, size + 1, size + 1);
  }

  /** 隐藏节点：圆圈 + 辉光（亮度 ∝ a1），选中描边。 */
  function drawHidden() {
    const a1 = state.trace ? state.trace.a1 : null;
    const n = nH();
    let scale = 1e-9;
    if (a1) {
      for (let h = 0; h < n; h++) {
        const a = Math.abs(a1[h]);
        if (a > scale) scale = a;
      }
    }
    for (let h = 0; h < n; h++) {
      const [cx, cy] = hiddenXY(h);
      const b = a1 ? clamp01(a1[h] / scale) * state.glowT : 0;
      // 辉光（径向渐变圆，不用 shadowBlur）
      if (b > 0.01) {
        const g = ctx.createRadialGradient(cx, cy, 1, cx, cy, HIDDEN.r * 2.4);
        g.addColorStop(0, `rgba(251,146,60,${(0.85 * b).toFixed(3)})`);
        g.addColorStop(1, 'rgba(251,146,60,0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(cx, cy, HIDDEN.r * 2.4, 0, TAU);
        ctx.fill();
      }
      // 本体
      const shade = Math.round(28 + 200 * (1 - b));
      ctx.beginPath();
      ctx.arc(cx, cy, HIDDEN.r, 0, TAU);
      ctx.fillStyle = `rgb(${shade},${Math.round(shade * 0.75)},${Math.round(shade * 0.45)})`;
      ctx.fill();
      ctx.strokeStyle = '#44403c';
      ctx.lineWidth = 1;
      ctx.stroke();
      // 热力图选中态
      if (state.heatmap === h) {
        ctx.beginPath();
        ctx.arc(cx, cy, HIDDEN.r + 3.5, 0, TAU);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }
    }
  }

  /** 输出概率柱；compare 模式画 before（灰）/ after（蓝）双色对比。 */
  function drawOutput() {
    const tr = state.trace;
    ctx.textBaseline = 'middle';
    for (let o = 0; o < 10; o++) {
      const cy = outY(o);
      ctx.fillStyle = '#57534e';
      ctx.font = FONT_NUM;
      ctx.textAlign = 'right';
      ctx.fillText(String(o), OUT.labelX, cy);
      if (state.compare) {
        const pb = state.compare.before[o];
        const pa = state.compare.after[o];
        // 上半：更新前（灰）；下半：更新后（蓝）
        ctx.fillStyle = '#a8a29e';
        ctx.fillRect(OUT.barX, cy - 6, pb * OUT.barMax, 5.5);
        ctx.fillStyle = `rgb(${POS})`;
        ctx.fillRect(OUT.barX, cy + 0.5, pa * OUT.barMax, 5.5);
        ctx.fillStyle = '#57534e';
        ctx.font = FONT_NUM_SM;
        ctx.textAlign = 'left';
        ctx.fillText(pa.toFixed(3), OUT.barX + OUT.barMax + 6, cy);
      } else if (tr) {
        const pShow = 0.1 + (tr.p[o] - 0.1) * smooth(state.barT);
        const isTop = o === tr.pred;
        ctx.fillStyle = isTop ? '#f59e0b' : '#64748b';
        ctx.fillRect(OUT.barX, cy - 6, pShow * OUT.barMax, 12);
        ctx.fillStyle = isTop ? INK : '#57534e';
        ctx.font = FONT_NUM_SM;
        ctx.textAlign = 'left';
        ctx.fillText(pShow.toFixed(3), OUT.barX + OUT.barMax + 6, cy);
      } else {
        ctx.fillStyle = '#e7e5e4';
        ctx.fillRect(OUT.barX, cy - 6, 0.1 * OUT.barMax, 12);
      }
    }
  }

  /** 权重脉冲：扩张圆环（训练动画用，state.now 驱动）。 */
  function drawPulses() {
    if (state.now == null) return;
    for (const pl of state.pulses) {
      const pt = (state.now - pl.t0) / pl.dur;
      if (pt < 0 || pt > 1) continue;
      const r = 3 + 17 * pt;
      const a = 0.75 * (1 - pt);
      ctx.beginPath();
      ctx.arc(pl.x, pl.y, r, 0, TAU);
      ctx.strokeStyle = `rgba(${pl.dw >= 0 ? POS : NEG},${a.toFixed(3)})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  function drawCaptions() {
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = MUTED;
    ctx.font = FONT_LABEL;
    // 左：随模式变化的标题
    ctx.textAlign = 'left';
    let leftTitle = '输入 · 60×60（6×6 块）';
    if (state.heatmap != null) leftTitle = `输入墙：隐藏节点 h${state.heatmap} 权重热力图（点节点/空白恢复）`;
    else if (state.blockDetail != null) leftTitle = '分块详情（点该块/空白恢复）';
    ctx.fillText(leftTitle, BLOCKS.x, BLOCKS.y - 14);
    // 中
    ctx.textAlign = 'center';
    ctx.fillText('隐藏层 · 32', HIDDEN.x0 + HIDDEN.colGap / 2, HIDDEN.y0 - 16);
    // 右
    ctx.textAlign = 'left';
    ctx.fillText('输出 · 概率', OUT.barX, OUT.y0 - 14);

    // 底栏：左 = 诚实标注（恒显），右 = 状态
    ctx.font = FONT_NOTE;
    ctx.fillStyle = MUTED;
    ctx.textAlign = 'left';
    ctx.fillText(HONEST, 16, H - 14);
    ctx.textAlign = 'right';
    if (state.compare) {
      const swY = H - 22;
      ctx.fillStyle = '#a8a29e';
      ctx.fillRect(W - 186, swY, 10, 10);
      ctx.fillStyle = `rgb(${POS})`;
      ctx.fillRect(W - 106, swY, 10, 10);
      ctx.fillStyle = MUTED;
      ctx.font = FONT_NOTE;
      ctx.fillText('更新前', W - 170, H - 13);
      ctx.fillText('更新后', W - 90, H - 13);
    } else if (state.trace) {
      ctx.fillStyle = MUTED;
      ctx.font = FONT_NOTE;
      ctx.fillText(`预测 ${state.trace.pred}　p=${state.trace.p[state.trace.pred].toFixed(3)}`, W - 16, H - 14);
    } else {
      ctx.fillStyle = MUTED;
      ctx.font = FONT_NOTE;
      ctx.fillText('等待推理…', W - 16, H - 14);
    }
  }

  /** 静态渲染当前状态（无动画）。 */
  function render() {
    ctx.clearRect(0, 0, W, H);
    drawEdges();
    drawPulses();
    drawInputArea();
    drawHidden();
    drawOutput();
    sys.draw(ctx);
    drawCaptions();
  }

  /* ————— 动画引擎 ————— */

  /**
   * 运行一段动画。重入时返回进行中的 Promise。
   * reduced-motion 下直接调 onFrame(1) 落终态并 resolve。
   */
  function startAnimation(durationMs, onFrame) {
    if (state.anim) return state.anim.promise;
    if (state.reduced) {
      state.now = durationMs;
      onFrame(1, 0);
      sys.clear();
      state.now = null;
      state.pulses.length = 0;
      state.glowT = 1;
      state.barT = 1;
      render();
      return Promise.resolve();
    }
    let resolvePromise;
    const promise = new Promise((res) => {
      resolvePromise = res;
    });
    const t0 = performance.now();
    let last = t0;
    const anim = { promise, raf: 0, cancelled: false, finish: null, exposed: null };
    state.anim = anim;
    const finish = () => {
      if (state.anim !== anim) return;
      state.anim = null;
      state.now = null;
      state.pulses.length = 0;
      sys.clear();
      state.glowT = 1;
      state.barT = 1;
      render(); // 落到终态
      resolvePromise();
    };
    anim.finish = finish;
    const frame = (now) => {
      if (anim.cancelled) return;
      const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
      last = now;
      const t = Math.min(1, (now - t0) / durationMs);
      state.now = now - t0;
      onFrame(t, dt);
      sys.update(dt);
      render();
      if (t < 1) {
        anim.raf = requestAnimationFrame(frame);
      } else {
        finish();
      }
    };
    anim.raf = requestAnimationFrame(frame);
    return promise;
  }

  /* ————— 对外 API ————— */

  const api = {
    /** 切换网络引用（模型加载/重置时调用），重建边缓存并清状态。 */
    setNet(next) {
      state.net = next;
      state.edges = buildEdges(next);
      state.x = null;
      state.trace = null;
      state.lastTrace = null;
      state.heatmap = null;
      state.heatColors = null;
      state.blockDetail = null;
      state.blockInfo = null;
      state.compare = null;
      state.pulses.length = 0;
      sys.clear();
      state.glowT = 1;
      state.barT = 1;
      render();
    },

    render,

    /** 最近一次推理的 trace（过程面板用）。 */
    getLastTrace() {
      return state.lastTrace;
    },

    /** 最近一次推理/训练步的 trace（旧调用方兼容）。 */
    get lastTrace() {
      return state.lastTrace;
    },

    /**
     * 推理动画：粒子 input→hidden→output 分两段流动，隐藏节点辉光渐亮，
     * 输出柱从 10% 收敛到 p。结束后 state.lastTrace = trace。
     * @param {number[]|Float64Array} x 3600 维输入（0~1）；长度不对直接抛错
     * @returns {Promise<object>} resolve(trace)；动画进行中重复调用
     *   （含与 playTrainingStep 交叉）返回进行中的同一 Promise
     */
    playInference(x) {
      if (!Array.isArray(x) && !(x instanceof Float64Array)) {
        throw new Error('playInference 需要 3600 维输入数组。');
      }
      if (x.length !== 3600) throw new Error(`playInference 需要 3600 维输入，收到 ${x.length}。`);
      if (state.anim) return state.anim.exposed; // 禁止重入
      const trace = inferWithTrace(state.net, x);
      state.x = Array.from(x);
      state.trace = trace;
      state.lastTrace = trace;
      state.compare = null;
      state.heatmap = null;
      state.heatColors = null;
      state.blockDetail = null;
      state.blockInfo = null;

      const { e1, e2 } = state.edges;
      const n = nH();
      let maxA1 = 1e-9;
      for (let h = 0; h < n; h++) {
        const a = Math.abs(trace.a1[h]);
        if (a > maxA1) maxA1 = a;
      }
      const totalA = 200;
      const totalB = 120;
      let spawnedA = 0;
      let spawnedB = 0;

      /** 前向段1：在 input→hidden 边上按源像素亮度采样生成粒子。 */
      const spawnFwd1 = () => {
        for (let k = 0; k < 4; k++) {
          const e = e1[(Math.random() * e1.length) | 0];
          const b = clamp01(state.x[e.i]);
          if (Math.random() < 0.12 + 0.88 * b) {
            sys.spawn(e, {
              speed: 2.6 + Math.random(),
              size: 2.2,
              color: `255,${Math.round(255 - 150 * b)},${Math.round(255 - 200 * b)}`, // 白→橙
              alpha: 0.85,
            });
            return;
          }
        }
      };
      /** 前向段2：在 hidden→output 边上按隐藏激活生成粒子。 */
      const spawnFwd2 = () => {
        const e = e2[(Math.random() * e2.length) | 0];
        const b = clamp01(trace.a1[e.h] / maxA1);
        sys.spawn(e, {
          speed: 2.6 + Math.random(),
          size: 2.2,
          color: `255,${Math.round(190 - 90 * b)},60`,
          alpha: 0.4 + 0.5 * b,
        });
      };

      const exposed = startAnimation(700, (t) => {
        state.glowT = clamp01(t / 0.45);
        state.barT = smooth((t - 0.4) / 0.6);
        if (t < 0.5) {
          const wantA = Math.floor(totalA * (t / 0.5));
          for (; spawnedA < wantA; spawnedA++) spawnFwd1();
        } else {
          for (; spawnedA < totalA; spawnedA++) spawnFwd1();
          const wantB = Math.floor(totalB * ((t - 0.5) / 0.5));
          for (; spawnedB < wantB; spawnedB++) spawnFwd2();
        }
      }).then(() => trace);
      // reduced-motion 下 startAnimation 不建 anim，直接返回即可
      if (state.anim) state.anim.exposed = exposed;
      return exposed;
    },

    /**
     * 训练步动画：红色误差粒子 output→hidden→input 反向流动，
     * |Δw| top-24 的边发扩张脉冲，输出柱显示 before/after 双色对比。
     * @param {object} before { trace, w1: 快照, w2: 快照, y? }
     * @param {object} after 同 before（训练步之后的快照）
     *   调用方保证 w1/w2 是训练步前后的切片快照（不是网络活引用），直接用即可。
     * @returns {Promise<void>}；动画中重复调用返回同一 Promise
     */
    playTrainingStep(before, after) {
      if (state.anim) return state.anim.exposed; // 禁止重入：返回进行中的同一 Promise
      if (!before || !before.trace || !after || !after.trace) {
        throw new Error('playTrainingStep 需要 before/after（含 trace、w1、w2 快照）。');
      }
      state.compare = { before: before.trace.p.slice(), after: after.trace.p.slice() };
      state.trace = after.trace;
      state.lastTrace = after.trace;

      // 误差 |onehot(y) − p|；y 缺省用 before 的预测
      const y = before.y != null ? before.y : after.y != null ? after.y : before.trace.pred;
      const err = new Array(10);
      let errSum = 0;
      for (let o = 0; o < 10; o++) {
        err[o] = Math.abs((o === y ? 1 : 0) - before.trace.p[o]);
        errSum += err[o];
      }
      if (errSum < 1e-9) errSum = 1e-9;

      // |Δw| top-24 脉冲（取自实际绘制的边）
      const { e1, e2 } = state.edges;
      const nI = nIn();
      const nHd = nH();
      const cands = [];
      for (const e of e1) {
        const dw = after.w1[e.h * nI + e.i] - before.w1[e.h * nI + e.i];
        cands.push({ x: (e.x1 + e.x2) / 2, y: (e.y1 + e.y2) / 2, dw });
      }
      for (const e of e2) {
        const dw = after.w2[e.o * nHd + e.h] - before.w2[e.o * nHd + e.h];
        cands.push({ x: (e.x1 + e.x2) / 2, y: (e.y1 + e.y2) / 2, dw });
      }
      cands.sort((a, b) => Math.abs(b.dw) - Math.abs(a.dw));
      state.pulses = cands.slice(0, 24).map((c, k) => ({
        x: c.x,
        y: c.y,
        dw: c.dw,
        t0: 120 + (k / 24) * 480,
        dur: 320,
      }));

      const { e2ByOutput } = state.edges;
      const totalB2 = 130;
      let spawnedB1 = 0;
      let spawnedB2 = 0;
      const totalB1 = 240;

      /** 反向段1：按输出误差加权选输出节点，粒子沿边反向流动。 */
      const spawnBack1 = () => {
        let r = Math.random() * errSum;
        let o = 0;
        while (o < 9 && r > err[o]) {
          r -= err[o];
          o++;
        }
        const edges = e2ByOutput[o];
        const e = edges[(Math.random() * edges.length) | 0];
        const b = clamp01(err[o]);
        sys.spawn(
          { x1: e.x2, y1: e.y2, x2: e.x1, y2: e.y1 },
          {
            speed: 2.4 + Math.random(),
            size: 2.2,
            color: `220,${Math.round(70 - 40 * b)},${Math.round(70 - 40 * b)}`, // 红
            alpha: 0.5 + 0.4 * b,
          }
        );
      };
      /** 反向段2：hidden→input 继续反向。 */
      const spawnBack2 = () => {
        const e = e1[(Math.random() * e1.length) | 0];
        sys.spawn(
          { x1: e.x2, y1: e.y2, x2: e.x1, y2: e.y1 },
          { speed: 2.4 + Math.random(), size: 2, color: '220,50,50', alpha: 0.6 }
        );
      };

      const exposed = startAnimation(950, (t) => {
        state.glowT = 1;
        if (t < 0.5) {
          const want = Math.floor(totalB1 * (t / 0.5));
          for (; spawnedB1 < want; spawnedB1++) spawnBack1();
        } else {
          for (; spawnedB1 < totalB1; spawnedB1++) spawnBack1();
          const want2 = Math.floor(totalB2 * ((t - 0.5) / 0.5));
          for (; spawnedB2 < want2; spawnedB2++) spawnBack2();
        }
      });
      if (state.anim) state.anim.exposed = exposed;
      return exposed;
    },

    /**
     * 高亮节点：layer='hidden'。切换该隐藏节点 3600 权重的蓝红热力图
     * （再调一次取消），同时清除分块详情。
     */
    highlightNode(layer, idx) {
      if (layer !== 'hidden') throw new Error(`highlightNode 暂不支持 layer='${layer}'。`);
      if (!Number.isInteger(idx) || idx < 0 || idx >= nH()) {
        throw new Error(`hidden 节点下标越界：${idx}。`);
      }
      state.heatmap = state.heatmap === idx ? null : idx;
      state.heatColors = null;
      state.blockDetail = null;
      state.blockInfo = null;
      render();
    },

    /** 清除热力图与分块详情，回到分块显示。 */
    clearHighlight() {
      state.heatmap = null;
      state.heatColors = null;
      state.blockDetail = null;
      state.blockInfo = null;
      render();
    },

    /**
     * 仅测试用：冻结当前动画帧（取消 rAF，保留画面，Promise 保持 pending）。
     * 生产代码不要调用。
     */
    _freeze() {
      const anim = state.anim;
      if (!anim) return false;
      anim.cancelled = true;
      cancelAnimationFrame(anim.raf);
      state.anim = null;
      return true;
    },
  };

  /* ————— 点击交互 ————— */

  function hitHidden(lx, ly) {
    const n = nH();
    for (let h = 0; h < n; h++) {
      const [cx, cy] = hiddenXY(h);
      const dx = lx - cx;
      const dy = ly - cy;
      if (dx * dx + dy * dy <= (HIDDEN.r + 5) * (HIDDEN.r + 5)) return h;
    }
    return -1;
  }

  function hitBlock(lx, ly) {
    const { x, y, cell, n } = BLOCKS;
    if (lx < x || ly < y || lx >= x + n * cell || ly >= y + n * cell) return -1;
    const bx = ((lx - x) / cell) | 0;
    const by = ((ly - y) / cell) | 0;
    return by * n + bx;
  }

  /** 分块 b 的 100 像素上 Σ|w1| 最大的 top-3 隐藏节点。 */
  function blockTop3(b) {
    const n = nIn();
    const nHd = nH();
    const bx = b % BLOCKS.n;
    const by = (b / BLOCKS.n) | 0;
    const sums = new Array(nHd);
    for (let h = 0; h < nHd; h++) {
      let s = 0;
      const off = h * n;
      for (let wy = 0; wy < BLOCKS.px; wy++) {
        const row = off + (by * BLOCKS.px + wy) * 60 + bx * BLOCKS.px;
        for (let wx = 0; wx < BLOCKS.px; wx++) s += Math.abs(state.net.w1[row + wx]);
      }
      sums[h] = s;
    }
    return sums
      .map((s, h) => [s, h])
      .sort((a, b2) => b2[0] - a[0])
      .slice(0, 3)
      .map(([, h]) => h);
  }

  function toLogical(ev) {
    const r = canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) * (W / r.width), (ev.clientY - r.top) * (H / r.height)];
  }

  canvas.addEventListener('click', (ev) => {
    const [lx, ly] = toLogical(ev);
    const h = hitHidden(lx, ly);
    if (h >= 0) {
      api.highlightNode('hidden', h);
      return;
    }
    const b = hitBlock(lx, ly);
    if (b >= 0) {
      if (state.heatmap != null) {
        state.heatmap = null;
        state.heatColors = null;
      }
      if (state.blockDetail === b) {
        state.blockDetail = null;
        state.blockInfo = null;
      } else {
        state.blockDetail = b;
        state.blockInfo = { b, top3: blockTop3(b) };
      }
    } else {
      state.heatmap = null;
      state.heatColors = null;
      state.blockDetail = null;
      state.blockInfo = null;
    }
    render();
  });

  canvas.addEventListener('mousemove', (ev) => {
    const [lx, ly] = toLogical(ev);
    canvas.style.cursor = hitHidden(lx, ly) >= 0 || hitBlock(lx, ly) >= 0 ? 'pointer' : 'default';
  });

  window.addEventListener('resize', () => {
    fitCanvas(canvas, ctx);
    render();
  });

  render();
  return api;
}
