import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createAdam, adamStep, sgdStep, paramCount } from '../js/optimizer.js';
import { createNetwork, forward, backward } from '../js/network.js';

const ARCH = { inputs: 169, hidden: 16, outputs: 10 };

/** 全常数 g 的梯度（形状与网络匹配）。 */
function constGrads(net, g) {
  return {
    gW1: new Array(net.w1.length).fill(g),
    gb1: new Array(net.b1.length).fill(g),
    gW2: new Array(net.w2.length).fill(g),
    gb2: new Array(net.b2.length).fill(g),
  };
}

/** 按 w1+b1+w2+b2 展平网络参数。 */
function flatParams(net) {
  return [...net.w1, ...net.b1, ...net.w2, ...net.b2];
}

test('createAdam 状态零初始化，t=0', () => {
  const s = createAdam(12);
  assert.strictEqual(s.t, 0);
  assert.ok(s.m.every((v) => v === 0));
  assert.ok(s.v.every((v) => v === 0));
  assert.strictEqual(s.m.length, 12);
});

test('Adam 单步：t 计数 1，平均 |Δw| ≈ lr（容差 10%）', () => {
  const net = createNetwork(ARCH, 42);
  const before = flatParams(net);
  const state = createAdam(paramCount(net));
  // 真实反向梯度
  const x = new Array(169).fill(0).map((_, i) => ((i * 37) % 13) / 13);
  const f = forward(net, x);
  const g = backward(net, x, 3, f);

  const lr = 0.01;
  adamStep(net, state, g, lr);
  assert.strictEqual(state.t, 1, 't 应从 0 增至 1');

  const after = flatParams(net);
  const deltas = after.map((w, i) => Math.abs(w - before[i]));
  const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  assert.ok(
    Math.abs(mean - lr) / lr < 0.1,
    `平均 |Δw| = ${mean}，期望 ≈ lr = ${lr}`
  );
});

test('Adam 第二步：t 增至 2，矩按偏差校正继续更新', () => {
  const net = createNetwork(ARCH, 42);
  const state = createAdam(paramCount(net));
  const grads = constGrads(net, 0.5);
  adamStep(net, state, grads, 0.01);
  assert.strictEqual(state.t, 1);
  // 第一步后：m = (1-β1)*g，v = (1-β2)*g²
  assert.ok(Math.abs(state.m[0] - 0.1 * 0.5) < 1e-15);
  assert.ok(Math.abs(state.v[0] - 0.001 * 0.25) < 1e-15);
  adamStep(net, state, grads, 0.01);
  assert.strictEqual(state.t, 2);
  // 恒定梯度下 m̂ ≡ g
  const t = 2;
  const mHat = state.m[0] / (1 - Math.pow(0.9, t));
  assert.ok(Math.abs(mHat - 0.5) < 1e-12, `m̂ = ${mHat}，期望 0.5`);
});

test('SGD：Δw == lr·g（精确，逐位）', () => {
  const net = createNetwork(ARCH, 42);
  const before = flatParams(net);
  const grads = constGrads(net, 0.5);
  const lr = 0.1;
  sgdStep(net, grads, lr);
  const after = flatParams(net);
  const gFlat = [...grads.gW1, ...grads.gb1, ...grads.gW2, ...grads.gb2];
  for (let i = 0; i < after.length; i++) {
    assert.strictEqual(after[i], before[i] + lr * gFlat[i], `参数 ${i}`);
  }
});

test('SGD 负梯度：参数减小（上升约定验证）', () => {
  const net = createNetwork(ARCH, 42);
  const before = flatParams(net);
  sgdStep(net, constGrads(net, -2.0), 0.1);
  const after = flatParams(net);
  for (let i = 0; i < after.length; i++) {
    assert.strictEqual(after[i], before[i] - 0.2);
  }
});

test('100 随机样本 × 3 epoch 训练：平均对数似然上升', () => {
  // 确定性伪随机数据集（简单 LCG，与被测 RNG 无关）
  let s = 20261004;
  const rand = () => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return s / 4294967296;
  };
  const samples = [];
  for (let k = 0; k < 100; k++) {
    const x = new Array(169);
    for (let i = 0; i < 169; i++) x[i] = rand();
    samples.push({ x, y: Math.floor(rand() * 10) });
  }

  const net = createNetwork(ARCH, 7);
  const meanLoglik = () =>
    samples.reduce((a, { x, y }) => a + Math.log(forward(net, x).p[y]), 0) / samples.length;
  const before = meanLoglik();

  const state = createAdam(paramCount(net));
  const lr = 0.01;
  for (let epoch = 0; epoch < 3; epoch++) {
    for (const { x, y } of samples) {
      const f = forward(net, x);
      const g = backward(net, x, y, f);
      adamStep(net, state, g, lr);
    }
  }
  const after = meanLoglik();
  assert.ok(after > before, `平均对数似然未上升：${before} -> ${after}`);
});
