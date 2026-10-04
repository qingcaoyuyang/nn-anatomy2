/**
 * MLP 网络核心：169→16→10，LeakyReLU(0.01) + Softmax + 对数似然梯度。
 * ES module，零 npm 依赖，无 DOM。
 *
 * # 符号约定（重要，训练循环必须遵守）
 * `backward` 返回的是**对数似然** ln(p[y]) 对参数的梯度：
 * `d ln(p[y])/dz2[o] = onehot(y)[o] − p[o]`，
 * 与交叉熵损失 L = −ln(p[y]) 的梯度符号**相反**。
 * 因此训练时必须做梯度**上升** `w += lr · g`（等价于交叉熵梯度下降）。
 *
 * # 初始化约定
 * He 初始化：权重 w = z0 · √(2/fan_in)，z0 为标准正态（见 js/rng.js）。
 * 抽取顺序：w1 行主序（w1[h·inputs+x]）→ b1 全零 → w2 行主序 → b2 全零。
 * 同 seed 的两次 createNetwork 产生完全相同的权重。
 *
 * # 前向约定
 * z1[h] = Σ w1·x + b1；a1[h] = z1>0 ? z1 : 0.01·z1（z1==0 取 0.01 分支）；
 * z2[o] = Σ w2·a1 + b2；p = softmax(z2)（减 max 做数值稳定）。
 * LeakyReLU 导数：z1>0 为 1，z1<=0 为 0.01（与 forward 分支一致）。
 */

import { createRng } from './rng.js';

/** LeakyReLU 负半轴斜率（全局约束）。 */
export const LEAKY_ALPHA = 0.01;

/**
 * 创建网络：arch = { inputs, hidden, outputs }，seed 截断到 u32。
 * 返回 { arch, w1, b1, w2, b2 }，w1/w2 为行优先扁平数组。
 */
export function createNetwork(arch, seed) {
  const { inputs, hidden, outputs } = arch;
  const rng = createRng(seed);
  const scale1 = Math.sqrt(2 / inputs);
  const scale2 = Math.sqrt(2 / hidden);

  const w1 = new Array(hidden * inputs);
  for (let i = 0; i < w1.length; i++) w1[i] = rng.nextNormal() * scale1;
  const b1 = new Array(hidden).fill(0);

  const w2 = new Array(outputs * hidden);
  for (let i = 0; i < w2.length; i++) w2[i] = rng.nextNormal() * scale2;
  const b2 = new Array(outputs).fill(0);

  return { arch: { inputs, hidden, outputs }, w1, b1, w2, b2 };
}

/** 前向：返回 { z1, a1, z2, p }。 */
export function forward(net, x) {
  const { inputs: nIn, hidden: nH, outputs: nOut } = net.arch;

  const z1 = new Array(nH);
  const a1 = new Array(nH);
  for (let h = 0; h < nH; h++) {
    let s = net.b1[h];
    const row = h * nIn;
    for (let xi = 0; xi < nIn; xi++) s += net.w1[row + xi] * x[xi];
    z1[h] = s;
    a1[h] = s > 0 ? s : LEAKY_ALPHA * s;
  }

  const z2 = new Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let s = net.b2[o];
    const row = o * nH;
    for (let h = 0; h < nH; h++) s += net.w2[row + h] * a1[h];
    z2[o] = s;
  }

  let max = -Infinity;
  for (let o = 0; o < nOut; o++) if (z2[o] > max) max = z2[o];
  const p = new Array(nOut);
  let sum = 0;
  for (let o = 0; o < nOut; o++) {
    const e = Math.exp(z2[o] - max);
    p[o] = e;
    sum += e;
  }
  for (let o = 0; o < nOut; o++) p[o] /= sum;

  return { z1, a1, z2, p };
}

/**
 * 反向：返回**对数似然** ln(p[y]) 对全部参数的解析梯度
 * { gW1, gb1, gW2, gb2 }。见文件头符号约定：训练时做梯度上升。
 */
export function backward(net, x, y, fwd) {
  const { inputs: nIn, hidden: nH, outputs: nOut } = net.arch;

  // dz2 = onehot(y) − p
  const dz2 = new Array(nOut);
  for (let o = 0; o < nOut; o++) dz2[o] = (o === y ? 1 : 0) - fwd.p[o];

  const gW2 = new Array(nOut * nH);
  const gb2 = new Array(nOut);
  for (let o = 0; o < nOut; o++) {
    gb2[o] = dz2[o];
    const row = o * nH;
    for (let h = 0; h < nH; h++) gW2[row + h] = dz2[o] * fwd.a1[h];
  }

  const dz1 = new Array(nH);
  const gW1 = new Array(nH * nIn);
  const gb1 = new Array(nH);
  for (let h = 0; h < nH; h++) {
    let s = 0;
    for (let o = 0; o < nOut; o++) s += net.w2[o * nH + h] * dz2[o];
    const dRelu = fwd.z1[h] > 0 ? 1 : LEAKY_ALPHA;
    dz1[h] = s * dRelu;
    gb1[h] = dz1[h];
    const row = h * nIn;
    for (let xi = 0; xi < nIn; xi++) gW1[row + xi] = dz1[h] * x[xi];
  }

  return { gW1, gb1, gW2, gb2 };
}
