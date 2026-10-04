import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork, forward, backward } from '../js/network.js';

const ARCH = { inputs: 169, hidden: 16, outputs: 10 };

// 确定性输入：与旧 Rust 测试同公式 ((i*37)%13)/13
function sampleInput() {
  const x = new Array(169);
  for (let i = 0; i < 169; i++) x[i] = ((i * 37) % 13) / 13;
  return x;
}

test('softmax 输出和为 1（误差 < 1e-12）', () => {
  const net = createNetwork(ARCH, 42);
  const f = forward(net, new Array(169).fill(0.5));
  assert.strictEqual(f.p.length, 10);
  const sum = f.p.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-12, `softmax 和 = ${sum}`);
  for (const v of f.p) assert.ok(v > 0 && v < 1, `概率越界: ${v}`);
});

test('同 seed 两次 createNetwork 权重完全相同', () => {
  const a = createNetwork(ARCH, 7);
  const b = createNetwork(ARCH, 7);
  assert.deepStrictEqual(a, b);
});

test('不同 seed 权重不同', () => {
  const a = createNetwork(ARCH, 7);
  const b = createNetwork(ARCH, 8);
  assert.notDeepStrictEqual(a.w1, b.w1);
});

test('He 初始化：w1 样本方差 ≈ 2/169（0.005~0.02）', () => {
  const net = createNetwork(ARCH, 1234);
  const n = net.w1.length;
  const mean = net.w1.reduce((a, b) => a + b, 0) / n;
  const variance = net.w1.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  assert.ok(Math.abs(mean) < 0.02, `均值偏离: ${mean}`);
  assert.ok(variance > 0.005 && variance < 0.02, `方差 = ${variance}，期望 ≈ ${(2 / 169).toFixed(5)}`);
});

test('He 初始化：w2 样本方差 ≈ 2/16', () => {
  const net = createNetwork(ARCH, 1234);
  const n = net.w2.length;
  const mean = net.w2.reduce((a, b) => a + b, 0) / n;
  const variance = net.w2.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  assert.ok(variance > 0.05 && variance < 0.25, `w2 方差 = ${variance}，期望 ≈ ${(2 / 16).toFixed(3)}`);
});

test('偏置全零初始化', () => {
  const net = createNetwork(ARCH, 99);
  assert.ok(net.b1.every((v) => v === 0));
  assert.ok(net.b2.every((v) => v === 0));
});

test('数值微分校验 backward：W1/W2/b1 各抽一点，误差 < 1e-6', () => {
  const net = createNetwork(ARCH, 7);
  const x = sampleInput();
  const y = 3;
  const f = forward(net, x);
  const g = backward(net, x, y, f);
  const eps = 1e-5;
  const loglik = (n) => Math.log(forward(n, x).p[y]);

  // 克隆网络并扰动单个参数
  const perturb = (setter, idx) => {
    const up = structuredClone(net);
    const dn = structuredClone(net);
    setter(up, idx, eps);
    setter(dn, idx, -eps);
    return (loglik(up) - loglik(dn)) / (2 * eps);
  };
  const cases = [
    ['w1', (n, i, d) => { n.w1[i] += d; }, 3 * 169 + 10, g.gW1[3 * 169 + 10]],
    ['w2', (n, i, d) => { n.w2[i] += d; }, 4, g.gW2[4]],
    ['b1', (n, i, d) => { n.b1[i] += d; }, 5, g.gb1[5]],
  ];
  for (const [name, setter, idx, analytic] of cases) {
    const numeric = perturb(setter, idx);
    assert.ok(
      Math.abs(numeric - analytic) < 1e-6,
      `${name}[${idx}] 数值=${numeric} 解析=${analytic}`
    );
  }
});

test('backward 符号约定：dz2 = onehot − p（对数似然梯度，上升方向）', () => {
  // y=3 时，若 p[3] 偏小，g_b2[3] 应为正（上升可增大 p[3]）
  const net = createNetwork(ARCH, 7);
  const x = sampleInput();
  const y = 3;
  const f = forward(net, x);
  const g = backward(net, x, y, f);
  assert.ok(Math.abs(g.gb2[y] - (1 - f.p[y])) < 1e-15);
  assert.ok(Math.abs(g.gb2[(y + 1) % 10] - (0 - f.p[(y + 1) % 10])) < 1e-15);
});

test('forward 返回 z1/a1/z2/p 且维度正确', () => {
  const net = createNetwork(ARCH, 1);
  const f = forward(net, sampleInput());
  assert.strictEqual(f.z1.length, 16);
  assert.strictEqual(f.a1.length, 16);
  assert.strictEqual(f.z2.length, 10);
  assert.strictEqual(f.p.length, 10);
  // LeakyReLU 分支一致性：a1[h] = z1>0 ? z1 : 0.01*z1
  for (let h = 0; h < 16; h++) {
    const expected = f.z1[h] > 0 ? f.z1[h] : 0.01 * f.z1[h];
    assert.strictEqual(f.a1[h], expected);
  }
});

test('isFiniteNet：正常网络 true；注入 NaN/Infinity 后 false', async () => {
  const { isFiniteNet } = await import('../js/network.js');
  const net = createNetwork(ARCH, 42);
  assert.strictEqual(isFiniteNet(net), true);
  net.w1[0] = NaN;
  assert.strictEqual(isFiniteNet(net), false);
  net.w1[0] = 0.5;
  net.b2[3] = Infinity;
  assert.strictEqual(isFiniteNet(net), false);
  assert.strictEqual(isFiniteNet(null), false);
});
