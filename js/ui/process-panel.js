/**
 * 过程面板：用纯 DOM 展示推理 / 训练的真实计算过程。
 *
 * ES module，零 npm 依赖。只操作 initProcessPanel 传入的 root，
 * 不碰 root 之外的 DOM（样式全部用行内 style，不注入全局 <style>）。
 *
 * # 接口（见 docs/rebuild-plan.md「接口约定」）
 * initProcessPanel(root) → { showInference(trace, preInfo), showTraining(info), clear() }
 * - trace = { x(3600), z1(32), a1(32), z2(10), p(10) }（inferWithTrace 返回）
 * - preInfo = { cx, cy, dx, dy }（preprocessWithInfo 返回）
 * - info = { before, after, y, lr, optimizer, epoch }；
 *   before/after = { trace, w1: 切片, w2: 切片, y }（stage.playTrainingStep 用的 before/after）
 *   optimizer 为 Adam 状态对象 { m, v, t }（见 js/optimizer.js createAdam）
 *
 * # 额外（可选，不在接口约定内，仅为补齐权重引用）
 * 面板额外暴露 setNet(net)：绑定当前网络后，showInference 的「主要贡献」
 * 一节才能算出 a1[h]·w2[o*,h] 与 x[i]·w1[h*,i]。未绑定时该节显示提示。
 */

const IN = 3600;   // 60×60
const NH = 32;
const NOUT = 10;

/** 3 位小数；把 "-0.000" 归一成 "0.000"。 */
function f3(v) {
  const s = Number(v).toFixed(3);
  return s === '-0.000' ? '0.000' : s;
}

/** 6 位小数（Δw 量级小，3 位会全变成 0.000）。 */
function f6(v) {
  const s = (v >= 0 ? '+' : '') + Number(v).toFixed(6);
  return s === '+-0.000000' ? '+0.000000' : s;
}

function el(tag, css, parent, text) {
  const e = document.createElement(tag);
  if (css) e.style.cssText = css;
  if (text !== undefined && text !== null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

function mono(parent, text) {
  return el('div', 'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;' +
    'font-size:12px;background:#0f172a;color:#e2e8f0;border-radius:6px;' +
    'padding:6px 8px;margin:4px 0;white-space:pre-wrap;word-break:break-all;', parent, text);
}

function argmax(a) {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}

/** 取数组中按 key 排序 top-k 的下标（降序）。 */
function topK(a, k, key) {
  const idx = a.map((_, i) => i);
  idx.sort((p, q) => key(q) - key(p));
  return idx.slice(0, k);
}

const STEP_CSS = 'border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;margin:8px 0;background:#fff;';
const TITLE_CSS = 'font-size:13px;font-weight:700;color:#0f172a;margin-bottom:2px;';
const NUM_CSS = 'font-size:12px;color:#334155;line-height:1.7;';
const BADGE_CSS = 'display:inline-block;min-width:20px;height:20px;line-height:20px;text-align:center;' +
  'background:#0f172a;color:#fff;border-radius:10px;font-size:12px;font-weight:700;margin-right:6px;';

function stepBlock(parent, n, title) {
  const box = el('div', STEP_CSS, parent);
  const t = el('div', TITLE_CSS, box);
  el('span', BADGE_CSS, t, String(n));
  t.appendChild(document.createTextNode(title));
  return box;
}

export function initProcessPanel(root) {
  if (!root) throw new Error('initProcessPanel: 需要传入 root 容器元素');
  root.style.cssText = 'font-size:13px;color:#0f172a;';
  let net = null;

  function setNet(n) { net = n || null; }

  function clear() {
    root.innerHTML = '';
    el('div', 'color:#94a3b8;font-size:13px;padding:24px 12px;text-align:center;line-height:1.8;', root,
      '跑一次推理或训练，这里会展示真实计算过程');
  }

  function showInference(trace, preInfo) {
    root.innerHTML = '';
    if (!trace || !trace.x || trace.x.length !== IN) {
      el('div', 'color:#b91c1c;padding:12px;', root, 'trace 无效：需要 3600 维输入');
      return;
    }
    const { x, z1, a1, z2, p } = trace;
    const { cx = 0, cy = 0, dx = 0, dy = 0 } = preInfo || {};
    const pred = argmax(p);

    el('div', 'font-size:14px;font-weight:800;margin-bottom:4px;', root, '推理过程：260×260 手写 → 数字');

    // ①
    let b = stepBlock(root, 1, '输入：260×260 手写画布');
    el('div', NUM_CSS, b, '原始笔迹为 260×260 灰度图（共 67600 像素），墨量归一化到 [0, 1]。');

    // ②
    b = stepBlock(root, 2, '找重心');
    mono(b, `cx = ${f3(cx)},  cy = ${f3(cy)}`);
    el('div', NUM_CSS, b, '墨量加权平均位置：cx = Σx·m / Σm，cy = Σy·m / Σm。');

    // ③
    b = stepBlock(root, 3, '平移居中');
    mono(b, `dx = ${f3(dx)},  dy = ${f3(dy)}\nT(x, y) = 双线性采样(I, x−dx, y−dy)`);
    el('div', NUM_CSS, b, '把重心搬到画布中心，消除书写位置偏差。');

    // ④
    b = stepBlock(root, 4, '下采样到 60×60（块平均）');
    let nz = 0, mx = 0;
    for (let i = 0; i < IN; i++) { if (x[i] > 0) nz++; if (x[i] > mx) mx = x[i]; }
    mono(b, '260×260 → 60×60：每格约 4.33×4.33 像素取平均，共 3600 维输入向量 x');
    el('div', NUM_CSS, b, `非零像素 ${nz} 个，最大墨量 ${f3(mx)}。`);

    // ⑤
    b = stepBlock(root, 5, 'z₁ = W₁·x + b₁');
    mono(b, 'z₁[h] = Σᵢ w₁[h,i]·x[i] + b₁[h]，h = 0..31（共 32×3600 = 115200 次乘加）');
    const topH = topK(a1, 3, (h) => a1[h]);
    el('div', NUM_CSS, b, '激活最高的 3 个隐藏节点（按 a₁ 从大到小）：');
    mono(b, topH.map((h) => `h${h}：z₁ = ${f3(z1[h])}  →  a₁ = ${f3(a1[h])}`).join('\n'));

    // ⑥
    b = stepBlock(root, 6, 'a₁ = LeakyReLU(z₁)');
    mono(b, 'a₁[h] = z₁>0 ? z₁ : 0.01·z₁');
    const neg = z1.filter((v) => v <= 0).length;
    el('div', NUM_CSS, b, `32 个节点中有 ${neg} 个 z₁ ≤ 0，走了 0.01 的负半轴分支（避免神经元彻底"死亡"）。`);

    // ⑦
    b = stepBlock(root, 7, 'z₂ = W₂·a₁ + b₂');
    mono(b, 'z₂[o] = Σₕ w₂[o,h]·a₁[h] + b₂[o]，o = 0..9');
    mono(b, '10 个 logit：\n' + z2.map((v, o) =>
      `${o}: ${f3(v)}${o === argmax(z2) ? ' ← 最大' : ''}`).join('   '));

    // ⑧
    b = stepBlock(root, 8, 'p = softmax(z₂)');
    mono(b, 'p[o] = e^(z₂[o]−max) / Σₒ e^(z₂[o]−max)');
    mono(b, '10 个概率：\n' + p.map((v, o) =>
      `${o}: ${f3(v)}${o === pred ? ' ← 预测' : ''}`).join('   '));
    el('div', 'font-size:13px;font-weight:700;margin-top:4px;', b, `预测数字 = ${pred}（概率 ${f3(p[pred])}）`);

    // 主要贡献
    b = stepBlock(root, '★', `主要贡献（预测类别 o* = ${pred}）`);
    if (!net || !net.w1 || !net.w2) {
      el('div', NUM_CSS, b, '（绑定网络权重后展示：panel.setNet(net)）');
      return { pred };
    }
    // z2[o*] 的来源：a1[h]·w2[o*,h]
    const c2 = new Array(NH);
    for (let h = 0; h < NH; h++) c2[h] = a1[h] * net.w2[pred * NH + h];
    const topC2 = topK(c2, 3, (h) => c2[h]);
    el('div', TITLE_CSS, b, 'z₂[o*] 来自哪些隐藏节点：a₁[h]·w₂[o*,h]（top-3）');
    mono(b, topC2.map((h) =>
      `h${h}：a₁=${f3(a1[h])} × w₂=${f3(net.w2[pred * NH + h])} = ${f3(c2[h])}`).join('\n'));
    let sum2 = 0;
    for (let h = 0; h < NH; h++) sum2 += c2[h];
    const chk2 = sum2 + net.b2[pred];
    el('div', NUM_CSS, b,
      `校验：Σₕ a₁[h]·w₂[o*,h] + b₂[o*] = ${f3(chk2)}，z₂[o*] = ${f3(z2[pred])}` +
      (Math.abs(chk2 - z2[pred]) < 1e-9 ? ' ✓ 对得上' : '（差值超出浮点误差，请检查）'));

    // z1[h*] 的来源：x[i]·w1[h*,i]
    const hStar = topH[0];
    const c1 = new Array(IN);
    const w1row = hStar * IN;
    for (let i = 0; i < IN; i++) c1[i] = x[i] * net.w1[w1row + i];
    const topC1 = topK(c1, 3, (i) => c1[i]);
    el('div', TITLE_CSS, b, `z₁[h${hStar}] 来自哪些像素：x[i]·w₁[h${hStar},i]（top-3）`);
    mono(b, topC1.map((i) => {
      const r = Math.floor(i / 60), c = i % 60;
      return `像素 i=${i}（行${r},列${c}）：x=${f3(x[i])} × w₁=${f3(net.w1[w1row + i])} = ${f3(c1[i])}`;
    }).join('\n'));
    return { pred };
  }

  function showTraining(info) {
    root.innerHTML = '';
    const { before, after, y, lr, optimizer, epoch } = info || {};
    if (!before || !after || !before.trace || !after.trace) {
      el('div', 'color:#b91c1c;padding:12px;', root, '训练信息无效：需要 before/after（含 trace）');
      return;
    }
    const pb = before.trace.p, pa = after.trace.p;
    const nll = (pp) => (pp[y] > 0 ? -Math.log(pp[y]) : Infinity);
    const lossB = nll(pb), lossA = nll(pa);
    const fmtLoss = (v) => (Number.isFinite(v) ? f3(v) : '+∞');

    el('div', 'font-size:14px;font-weight:800;margin-bottom:4px;', root,
      `训练一步：样本标签 y = ${y}${epoch !== undefined ? `（第 ${epoch} 轮/步）` : ''}`);

    // 损失
    let b = stepBlock(root, 1, '损失：交叉熵 loss = −ln(p[y])');
    mono(b, `loss_before = −ln(${f3(pb[y])}) = ${fmtLoss(lossB)}\n` +
      `loss_after  = −ln(${f3(pa[y])}) = ${fmtLoss(lossA)}`);
    el('div', NUM_CSS, b, Number.isFinite(lossB) && Number.isFinite(lossA)
      ? `Δloss = ${f6(lossA - lossB)}（负值说明这一步学对了方向）`
      : '某一步 p[y] = 0，损失为 +∞（极端欠拟合时的真实情况）。');

    // 输出误差
    b = stepBlock(root, 2, '输出误差 δ₂ = p − onehot(y)');
    const d2 = pb.map((v, o) => v - (o === y ? 1 : 0));
    const topD = topK(d2, 3, (o) => Math.abs(d2[o]));
    mono(b, 'δ₂ 最大的 3 个分量（按 |δ|）：\n' + topD.map((o) =>
      `o=${o}：δ₂ = ${f3(d2[o])}（p=${f3(pb[o])}，目标 ${o === y ? 1 : 0}）`).join('\n'));
    el('div', NUM_CSS, b, '反向传播就从这里出发：误差沿 W₂ 传回隐藏层，再沿 W₁ 传回像素。');

    // |Δw| top-8
    b = stepBlock(root, 3, '|Δw| 最大的 8 条边（Δw = after − before）');
    if (before.w1 && after.w1 && before.w2 && after.w2) {
      const edges = [];
      const dw1 = after.w1, sw1 = before.w1;
      for (let i = 0; i < dw1.length; i++) {
        const d = dw1[i] - sw1[i];
        if (d !== 0) edges.push({ layer: 'W1', idx: i, d, ad: Math.abs(d) });
      }
      const dw2 = after.w2, sw2 = before.w2;
      for (let i = 0; i < dw2.length; i++) {
        const d = dw2[i] - sw2[i];
        if (d !== 0) edges.push({ layer: 'W2', idx: i, d, ad: Math.abs(d) });
      }
      edges.sort((p2, q2) => q2.ad - p2.ad);
      const label = (e2) => {
        if (e2.layer === 'W1') {
          const h = Math.floor(e2.idx / IN), i = e2.idx % IN;
          return `W1[h${h},i${i}(行${Math.floor(i / 60)},列${i % 60})]`;
        }
        const o = Math.floor(e2.idx / NH), h = e2.idx % NH;
        return `W2[o${o},h${h}]`;
      };
      mono(b, edges.slice(0, 8).map((e2) => `${label(e2)}  Δ = ${f6(e2.d)}`).join('\n') ||
        '（权重无变化：Δw 全为 0）');
      el('div', NUM_CSS, b, '变化最大的边正是"这次最该负责"的连接——梯度 × 学习率落到了它们身上。');
    } else {
      el('div', NUM_CSS, b, '（before/after 未带 w1/w2 切片，无法计算 Δw）');
    }

    // Adam 状态
    b = stepBlock(root, 4, '优化器状态');
    let optLine = `lr = ${lr !== undefined ? lr : '—'}`;
    if (optimizer && Array.isArray(optimizer.m) && Array.isArray(optimizer.v)) {
      const meanAbs = (arr) => arr.reduce((s, v) => s + Math.abs(v), 0) / arr.length;
      optLine += `　Adam：t = ${optimizer.t}，全局平均 |m| = ${meanAbs(optimizer.m).toExponential(3)}` +
        `，平均 |v| = ${meanAbs(optimizer.v).toExponential(3)}`;
      el('div', NUM_CSS, b, 'm 是一阶矩（梯度的滑动平均，管方向），v 是二阶矩（梯度平方的滑动平均，管步长缩放）。');
    } else if (typeof optimizer === 'string') {
      optLine += `　优化器：${optimizer}`;
    } else {
      optLine += '　（未传入 Adam 状态对象）';
    }
    mono(b, optLine);

    // before/after 概率对比
    b = stepBlock(root, 5, 'before → after：10 类概率对比');
    const predB = argmax(pb), predA = argmax(pa);
    el('div', NUM_CSS, b, `before 预测 = ${predB}（p=${f3(pb[predB])}）→ after 预测 = ${predA}（p=${f3(pa[predA])}）` +
      (predA === y ? ' ✓ 答对了' : predB === y ? '（这一步把它学错了）' : '（还没学会）'));
    for (let o = 0; o < NOUT; o++) {
      const row = el('div', 'display:flex;align-items:center;gap:6px;margin:3px 0;' +
        (o === y ? 'background:#ecfdf5;border-radius:4px;padding:2px 4px;' : ''), b);
      el('div', 'width:52px;font-size:12px;font-weight:700;flex:none;', row,
        `数字 ${o}${o === y ? '◀真值' : ''}`);
      const barBox = el('div', 'flex:1;display:flex;flex-direction:column;gap:2px;', row);
      const bar = (val, color) => {
        const w = el('div', `height:10px;border-radius:3px;background:${color};` +
          `width:${Math.max(0, Math.min(100, val * 100)).toFixed(1)}%;`, barBox);
        w.title = f3(val);
      };
      bar(pb[o], '#94a3b8');
      bar(pa[o], '#0d9488');
      el('div', 'width:120px;flex:none;font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#475569;', row,
        `${f3(pb[o])} → ${f3(pa[o])}`);
    }
    el('div', NUM_CSS, b, '灰条 = 训练前，青条 = 训练后。真值行有浅绿底。');
    return { lossBefore: lossB, lossAfter: lossA, predBefore: predB, predAfter: predA };
  }

  clear();
  return { showInference, showTraining, clear, setNet };
}
