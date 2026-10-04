/**
 * 神经网络舞台：169→16→10 MLP 的实时可视化（Canvas 2D）。
 *
 * ES module，零 npm 依赖。布局：
 *   左：13×13 输入像素墙 ｜ 中：16 隐藏节点竖排 ｜ 右：10 条输出概率柱。
 * 连线：input→hidden 每节点只画 |w1| 最大的 top-12（192 条），
 *       hidden→output 16×10 全画（160 条）；线宽 ∝ |w|，蓝=正 / 红=负。
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
 *
 * # 状态语义（教学停留用）
 * - playTrainingStep 结束后，输出柱保持 before（灰）/ after（蓝）双色对比态，
 *   直到下一次 playInference / render；
 * - playInference 开始时会清除热力图（highlightNode）与对比柱，回到"当前输入"态。
 *
 * # 点击交互
 * 点击隐藏节点 → highlightNode('hidden', idx)：输入墙切换显示该节点
 * 169 权重的蓝红热力图；再点一次该节点、或点击空白处 → 恢复输入显示。
 */

import { inferWithTrace } from '../trainer.js';
import { createParticleSystem, MAX_PARTICLES } from './particles.js';

const TAU = Math.PI * 2;

/* —— 逻辑画布尺寸（与 <canvas width=640 height=420> 属性一致） —— */
const W = 640;
const H = 420;

/* —— 布局 —— */
const WALL = { x: 20, y: 139, cell: 11, n: 13 }; // 13×13=143，垂直居中
const HIDDEN = { x: 300, y0: 40, y1: 380, r: 12, n: 16 };
const OUT = { labelX: 396, barX: 425, barMax: 140, y0: 60, rowH: 30, n: 10 };

/* —— 配色 —— */
const POS = '37,99,235'; // 蓝（正权重）
const NEG = '220,38,38'; // 红（负权重）
const INK = '#1c1917';
const MUTED = '#78716c';
const GRID_BG = '#fafaf9';

const FONT_NUM = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
const FONT_NUM_SM = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
const FONT_LABEL = '12px "Noto Sans SC", sans-serif';

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (t) => {
  const c = clamp01(t);
  return c * c * (3 - 2 * c);
};

/** 隐藏节点 h 的圆心 y。 */
function hiddenY(h) {
  return HIDDEN.y0 + (h * (HIDDEN.y1 - HIDDEN.y0)) / (HIDDEN.n - 1);
}

/** 输出行 o 的中心 y。 */
function outY(o) {
  return OUT.y0 + o * OUT.rowH + OUT.rowH / 2;
}

/** 输入像素 i 的中心坐标。 */
function pixelXY(i) {
  const c = i % WALL.n;
  const r = (i / WALL.n) | 0;
  return [WALL.x + c * WALL.cell + WALL.cell / 2, WALL.y + r * WALL.cell + WALL.cell / 2];
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
 * e1: input→hidden（每隐藏节点 top-12）；e2: hidden→output（全）。
 */
function buildEdges(net) {
  const { inputs: nIn, hidden: nH, outputs: nOut } = net.arch;
  const e1 = [];
  let max1 = 1e-9;
  for (let h = 0; h < nH; h++) {
    const top = topKAbs(net.w1, h * nIn, nIn, 12);
    for (const i of top) {
      const w = net.w1[h * nIn + i];
      const aw = Math.abs(w);
      if (aw > max1) max1 = aw;
      const [px, py] = pixelXY(i);
      e1.push({ h, i, x1: px, y1: py, x2: HIDDEN.x, y2: hiddenY(h), w });
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
      const e = { o, h, x1: HIDDEN.x, y1: hiddenY(h), x2: OUT.barX, y2: outY(o), w };
      e2.push(e);
      e2ByOutput[o].push(e);
    }
  }
  return { e1, e2, e2ByOutput, max1, max2 };
}

/** DPR 自适应：backing store 按 dpr 放大，逻辑坐标恒为 640×420。 */
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
 * @returns {{ setNet, render, playInference, playTrainingStep, highlightNode, lastTrace, _freeze }}
 *   _freeze 仅测试用：冻结当前动画帧（供截图），不对外文档化。
 */
export function initStage(canvas, net) {
  const ctx = canvas.getContext('2d');
  fitCanvas(canvas, ctx);

  const sys = createParticleSystem(MAX_PARTICLES);

  const state = {
    net,
    edges: buildEdges(net),
    x: null, // 当前输入（169 数组）或 null
    trace: null, // inferWithTrace 结果或 null
    lastTrace: null,
    heatmap: null, // 正在做权重热力图的隐藏节点下标，或 null
    compare: null, // { before: p[10], after: p[10] } 或 null
    pulses: [], // 权重脉冲 [{ x, y, dw, t0, dur }]
    glowT: 1, // 隐藏节点辉光进度（动画中 0→1）
    barT: 1, // 输出柱收敛进度（动画中 0→1）
    now: null, // 当前动画已流逝 ms（脉冲用），非动画时 null
    anim: null, // 进行中的动画 { promise, raf, cancelled, finish }
    reduced: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  };

  /* ————— 渲染 ————— */

  function drawEdges() {
    const { e1, e2, max1, max2 } = state.edges;
    ctx.lineWidth = 1;
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

  /** 输入墙：热力图模式显示隐藏节点权重，否则显示当前输入 x（无则空白）。 */
  function drawInputWall() {
    ctx.fillStyle = GRID_BG;
    ctx.fillRect(WALL.x, WALL.y, WALL.n * WALL.cell, WALL.n * WALL.cell);
    if (state.heatmap != null) {
      const h = state.heatmap;
      const { inputs: nIn } = state.net.arch;
      let m = 1e-9;
      for (let i = 0; i < nIn; i++) {
        const a = Math.abs(state.net.w1[h * nIn + i]);
        if (a > m) m = a;
      }
      for (let i = 0; i < nIn; i++) {
        const w = state.net.w1[h * nIn + i];
        const k = clamp01(Math.abs(w) / m);
        const c = Math.round(245 - 200 * k);
        ctx.fillStyle = w >= 0 ? `rgb(${c},${c},245)` : `rgb(245,${c},${c})`;
        const [px, py] = pixelXY(i);
        ctx.fillRect(px - WALL.cell / 2, py - WALL.cell / 2, WALL.cell, WALL.cell);
      }
    } else if (state.x) {
      for (let i = 0; i < 169; i++) {
        const g = Math.round(255 * (1 - clamp01(state.x[i])));
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        const [px, py] = pixelXY(i);
        ctx.fillRect(px - WALL.cell / 2, py - WALL.cell / 2, WALL.cell, WALL.cell);
      }
    }
    ctx.strokeStyle = '#e7e5e4';
    ctx.lineWidth = 1;
    ctx.strokeRect(WALL.x, WALL.y, WALL.n * WALL.cell + 1, WALL.n * WALL.cell + 1);
  }

  /** 隐藏节点：圆圈 + 辉光（亮度 ∝ a1），选中描边。 */
  function drawHidden() {
    const a1 = state.trace ? state.trace.a1 : null;
    let scale = 1e-9;
    if (a1) {
      for (let h = 0; h < 16; h++) {
        const a = Math.abs(a1[h]);
        if (a > scale) scale = a;
      }
    }
    for (let h = 0; h < 16; h++) {
      const cx = HIDDEN.x;
      const cy = hiddenY(h);
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
    ctx.fillStyle = MUTED;
    ctx.font = FONT_LABEL;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(state.heatmap != null ? `输入墙：隐藏节点 h${state.heatmap} 权重热力图（点节点切换回输入）` : '输入 · 13×13', WALL.x, WALL.y - 14);
    ctx.textAlign = 'center';
    ctx.fillText('隐藏层 · 16', HIDDEN.x, HIDDEN.y0 - 16);
    ctx.textAlign = 'left';
    ctx.fillText('输出 · 概率', OUT.barX, OUT.y0 - 14);
    // 底栏状态
    ctx.textAlign = 'left';
    if (state.compare) {
      ctx.fillStyle = '#a8a29e';
      ctx.fillRect(W - 190, H - 22, 10, 10);
      ctx.fillStyle = `rgb(${POS})`;
      ctx.fillRect(W - 100, H - 22, 10, 10);
      ctx.fillStyle = MUTED;
      ctx.fillText('更新前', W - 176, H - 13);
      ctx.fillText('更新后', W - 86, H - 13);
    } else if (state.trace) {
      ctx.fillStyle = MUTED;
      ctx.fillText(
        `预测 ${state.trace.pred}　p=${state.trace.p[state.trace.pred].toFixed(3)}`,
        WALL.x,
        H - 14
      );
    } else {
      ctx.fillStyle = MUTED;
      ctx.fillText('等待推理…', WALL.x, H - 14);
    }
  }

  /** 静态渲染当前状态（无动画）。 */
  function render() {
    ctx.clearRect(0, 0, W, H);
    drawEdges();
    drawPulses();
    drawInputWall();
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
    const anim = { promise, raf: 0, cancelled: false, finish: null };
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
      state.compare = null;
      state.pulses.length = 0;
      sys.clear();
      state.glowT = 1;
      state.barT = 1;
      render();
    },

    render,

    /** 最近一次推理/训练步的 trace（测试与 Task 11/12 读取用）。 */
    get lastTrace() {
      return state.lastTrace;
    },

    /**
     * 推理动画：粒子 input→hidden→output 分两段流动，隐藏节点辉光渐亮，
     * 输出柱从 10% 收敛到 p。
     * @param {number[]} x 169 维输入（0~1）
     * @returns {Promise<object>} resolve(trace)；动画进行中重复调用
     *   （含与 playTrainingStep 交叉）返回进行中的同一 Promise
     */
    playInference(x) {
      if (state.anim) return state.anim.exposed; // 禁止重入
      if (!Array.isArray(x) && !(x instanceof Float64Array)) {
        throw new Error('playInference 需要 169 维输入数组。');
      }
      if (x.length !== 169) throw new Error(`playInference 需要 169 维输入，收到 ${x.length}。`);
      const trace = inferWithTrace(state.net, x);
      state.x = Array.from(x);
      state.trace = trace;
      state.lastTrace = trace;
      state.compare = null;
      state.heatmap = null;

      const { e1, e2 } = state.edges;
      let maxA1 = 1e-9;
      for (let h = 0; h < 16; h++) {
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
     * @param {object} before { trace, w1:[...], w2:[...], y? }
     *   **注意：w1/w2 必须是训练步之前的快照拷贝**（如 net.w1.slice()）；
     *   传网络的活引用会导致 Δw 全零、脉冲静默消失。
     * @param {object} after 同 before（训练步之后的快照）
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
      const cands = [];
      for (const e of e1) {
        const dw = after.w1[e.h * 169 + e.i] - before.w1[e.h * 169 + e.i];
        cands.push({ x: (e.x1 + e.x2) / 2, y: (e.y1 + e.y2) / 2, dw });
      }
      for (const e of e2) {
        const dw = after.w2[e.o * 16 + e.h] - before.w2[e.o * 16 + e.h];
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
     * 高亮节点：目前支持 layer='hidden'。切换该隐藏节点 169 权重的
     * 蓝红热力图（再调一次取消）。
     */
    highlightNode(layer, idx) {
      if (layer !== 'hidden') throw new Error(`highlightNode 暂不支持 layer='${layer}'。`);
      if (!Number.isInteger(idx) || idx < 0 || idx >= 16) {
        throw new Error(`hidden 节点下标越界：${idx}。`);
      }
      state.heatmap = state.heatmap === idx ? null : idx;
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
    for (let h = 0; h < 16; h++) {
      const dx = lx - HIDDEN.x;
      const dy = ly - hiddenY(h);
      if (dx * dx + dy * dy <= (HIDDEN.r + 5) * (HIDDEN.r + 5)) return h;
    }
    return -1;
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
    } else if (state.heatmap != null) {
      state.heatmap = null;
      render();
    }
  });

  canvas.addEventListener('mousemove', (ev) => {
    const [lx, ly] = toLogical(ev);
    canvas.style.cursor = hitHidden(lx, ly) >= 0 ? 'pointer' : 'default';
  });

  window.addEventListener('resize', () => {
    fitCanvas(canvas, ctx);
    render();
  });

  render();
  return api;
}
