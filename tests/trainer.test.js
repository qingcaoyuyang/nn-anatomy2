import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork, forward } from '../js/network.js';
import { ADAM_DEFAULT_LR } from '../js/optimizer.js';
import { trainEpochs, evaluate, inferWithTrace, createOptState } from '../js/trainer.js';

const ARCH = { inputs: 169, hidden: 16, outputs: 10 };

// ---------- 合成数据（刻意不用 js/rng.js，避免自举；直接用 Math.random） ----------
// 每类一个 13×13 高斯斑点，中心排成 5×2 网格 + 均匀噪声，类别线性可分。
function makeSyntheticData(perClass) {
  const samples = [];
  for (let c = 0; c < 10; c++) {
    const cx = 2 + (c % 5) * 2.25;
    const cy = 3 + Math.floor(c / 5) * 7;
    for (let i = 0; i < perClass; i++) {
      const x = new Array(169);
      for (let r = 0; r < 13; r++) {
        for (let col = 0; col < 13; col++) {
          const d2 = (col - cx) ** 2 + (r - cy) ** 2;
          const v = Math.exp(-d2 / (2 * 1.2 * 1.2)) + Math.random() * 0.1;
          x[r * 13 + col] = Math.min(1, v);
        }
      }
      samples.push({ x, y: c });
    }
  }
  return samples;
}

// 当前网络在样本集上的平均对数似然（训练前后对比用）。
function meanLogLikelihood(net, samples) {
  let sum = 0;
  for (const s of samples) sum += Math.log(forward(net, s.x).p[s.y]);
  return sum / samples.length;
}

test('createOptState：adam 返回 {m,v,t}，sgd 返回 null', () => {
  const net = createNetwork(ARCH, 1);
  const adam = createOptState(net, 'adam');
  assert.strictEqual(adam.t, 0);
  assert.strictEqual(adam.m.length, 169 * 16 + 16 + 16 * 10 + 10);
  assert.strictEqual(adam.v.length, adam.m.length);
  assert.strictEqual(createOptState(net, 'sgd'), null);
  assert.throws(() => createOptState(net, 'rmsprop'), /未知 optimizer/);
});

test('可复现性：同 seed 两次 trainEpochs 权重 bit-identical', () => {
  const data = makeSyntheticData(5);
  const netA = createNetwork(ARCH, 7);
  const netB = createNetwork(ARCH, 7);
  const optA = createOptState(netA, 'adam');
  const optB = createOptState(netB, 'adam');
  const resA = trainEpochs(netA, optA, data, { epochs: 3, seed: 42 });
  const resB = trainEpochs(netB, optB, data, { epochs: 3, seed: 42 });
  assert.deepStrictEqual(resA, resB);
  assert.deepStrictEqual(netA.w1, netB.w1);
  assert.deepStrictEqual(netA.b1, netB.b1);
  assert.deepStrictEqual(netA.w2, netB.w2);
  assert.deepStrictEqual(netA.b2, netB.b2);
  assert.deepStrictEqual(optA, optB);
});

test('可复现性：不同 seed 训练结果不同（seed 真的生效）', () => {
  const data = makeSyntheticData(5);
  const netA = createNetwork(ARCH, 7);
  const netB = createNetwork(ARCH, 7);
  trainEpochs(netA, createOptState(netA, 'adam'), data, { epochs: 2, seed: 42 });
  trainEpochs(netB, createOptState(netB, 'adam'), data, { epochs: 2, seed: 43 });
  assert.notDeepStrictEqual(netA.w1, netB.w1);
});

test('200 合成样本 × 5 epoch：对数似然上升、准确率提升', () => {
  const data = makeSyntheticData(20);
  const net = createNetwork(ARCH, 11);
  const llBefore = meanLogLikelihood(net, data);
  const accBefore = evaluate(net, data).acc;
  const res = trainEpochs(net, createOptState(net, 'adam'), data, { epochs: 5, seed: 7 });
  const llAfter = meanLogLikelihood(net, data);
  const accAfter = evaluate(net, data).acc;
  assert.strictEqual(res.epochsDone, 5);
  assert.strictEqual(res.cancelled, false);
  assert.ok(Number.isFinite(res.logLikelihood), `logLikelihood 非有限: ${res.logLikelihood}`);
  assert.ok(llAfter > llBefore, `对数似然未上升: ${llBefore} → ${llAfter}`);
  assert.ok(accAfter > accBefore, `准确率未提升: ${accBefore} → ${accAfter}`);
  assert.ok(accAfter > 0.5, `5 epoch 后准确率过低: ${accAfter}`);
});

test('取消：onEpoch 在第 2 个 epoch 返回 false → epochsDone=1, cancelled', () => {
  const data = makeSyntheticData(5);
  const net = createNetwork(ARCH, 3);
  const w1Before = net.w1.slice();
  const seen = [];
  const res = trainEpochs(net, createOptState(net, 'adam'), data, {
    epochs: 5,
    seed: 9,
    onEpoch: (epoch, metrics) => {
      seen.push({ epoch, ll: metrics.logLikelihood, acc: metrics.acc });
      return epoch < 2; // 第 2 个 epoch 开始前返回 false
    },
  });
  assert.strictEqual(res.epochsDone, 1);
  assert.strictEqual(res.cancelled, true);
  assert.notDeepStrictEqual(net.w1, w1Before); // 第 1 个 epoch 已训练，权重变了
  // onEpoch 是"每 epoch 开始前"的门：epoch=1 时收到初始网络指标，epoch=2 时收到第 1 epoch 训练后指标
  assert.strictEqual(seen.length, 2);
  assert.ok(Number.isFinite(seen[0].ll) && Number.isFinite(seen[1].ll));
  assert.ok(seen[1].ll >= seen[0].ll, '训练 1 个 epoch 后对数似然不应下降');
  assert.ok(seen.every((m) => m.acc >= 0 && m.acc <= 1));
});

test('取消：第 1 个 epoch 就返回 false → epochsDone=0，权重不变', () => {
  const data = makeSyntheticData(5);
  const net = createNetwork(ARCH, 3);
  const snapshot = JSON.parse(JSON.stringify(net));
  const res = trainEpochs(net, createOptState(net, 'adam'), data, {
    epochs: 5,
    onEpoch: () => false,
  });
  assert.strictEqual(res.epochsDone, 0);
  assert.strictEqual(res.cancelled, true);
  assert.deepStrictEqual(net, snapshot);
});

test('evaluate：手工构造 10 样本，confusion 对角线全对', () => {
  // 构造"透视"网络：w1 取前 16 个像素直通，w2 对角放大 10 倍 → 样本 y 的预测必为 y
  const net = createNetwork(ARCH, 1);
  net.w1.fill(0); net.b1.fill(0); net.w2.fill(0); net.b2.fill(0);
  for (let h = 0; h < 16; h++) net.w1[h * 169 + h] = 1;
  for (let o = 0; o < 10; o++) net.w2[o * 16 + o] = 10;
  const samples = [];
  for (let y = 0; y < 10; y++) {
    const x = new Array(169).fill(0);
    x[y] = 1;
    samples.push({ x, y });
  }
  const { acc, perClass, confusion } = evaluate(net, samples);
  assert.strictEqual(acc, 1);
  for (let c = 0; c < 10; c++) {
    assert.strictEqual(perClass[c], 1, `perClass[${c}]`);
    for (let p = 0; p < 10; p++) {
      assert.strictEqual(confusion[c][p], c === p ? 1 : 0, `confusion[${c}][${p}]`);
    }
  }
});

test('evaluate：缺席类别的 perClass 为 null', () => {
  const net = createNetwork(ARCH, 1);
  net.w1.fill(0); net.b1.fill(0); net.w2.fill(0); net.b2.fill(0);
  for (let h = 0; h < 16; h++) net.w1[h * 169 + h] = 1;
  for (let o = 0; o < 10; o++) net.w2[o * 16 + o] = 10;
  const samples = [];
  for (let y = 0; y < 5; y++) {
    const x = new Array(169).fill(0);
    x[y] = 1;
    samples.push({ x, y });
  }
  const { acc, perClass } = evaluate(net, samples);
  assert.strictEqual(acc, 1);
  for (let c = 0; c < 5; c++) assert.strictEqual(perClass[c], 1);
  for (let c = 5; c < 10; c++) assert.strictEqual(perClass[c], null);
});

test('空样本：trainEpochs 返回 epochsDone=0，evaluate acc=null，都不抛错', () => {
  const net = createNetwork(ARCH, 1);
  const res = trainEpochs(net, createOptState(net, 'adam'), [], { epochs: 5 });
  assert.deepStrictEqual(res, { epochsDone: 0, logLikelihood: null, cancelled: false });
  const ev = evaluate(net, []);
  assert.strictEqual(ev.acc, null);
  assert.ok(ev.perClass.every((v) => v === null));
  assert.ok(ev.confusion.every((row) => row.every((v) => v === 0)));
});

test("optimizer='sgd'：2 epoch 跑通，对数似然不发散", () => {
  const data = makeSyntheticData(10);
  const net = createNetwork(ARCH, 5);
  const llBefore = meanLogLikelihood(net, data);
  const res = trainEpochs(net, createOptState(net, 'sgd'), data, {
    epochs: 2,
    optimizer: 'sgd',
    lr: 0.01,
    seed: 5,
  });
  assert.strictEqual(res.epochsDone, 2);
  assert.strictEqual(res.cancelled, false);
  const llAfter = meanLogLikelihood(net, data);
  assert.ok(Number.isFinite(llAfter), `sgd 对数似然发散: ${llAfter}`);
  assert.ok(llAfter > llBefore, `sgd 对数似然未上升: ${llBefore} → ${llAfter}`);
});

test('默认 lr 就是 ADAM_DEFAULT_LR', () => {
  const data = makeSyntheticData(3);
  const netA = createNetwork(ARCH, 2);
  const netB = createNetwork(ARCH, 2);
  trainEpochs(netA, createOptState(netA, 'adam'), data, { epochs: 2, seed: 1 });
  trainEpochs(netB, createOptState(netB, 'adam'), data, { epochs: 2, seed: 1, lr: ADAM_DEFAULT_LR });
  assert.deepStrictEqual(netA.w1, netB.w1);
});

test('inferWithTrace：返回全部中间值 + pred = argmax p', () => {
  const net = createNetwork(ARCH, 21);
  const x = new Array(169).fill(0.3);
  const t = inferWithTrace(net, x);
  for (const k of ['z1', 'a1', 'z2', 'p', 'pred']) {
    assert.ok(k in t, `缺少字段 ${k}`);
  }
  assert.strictEqual(t.z1.length, 16);
  assert.strictEqual(t.a1.length, 16);
  assert.strictEqual(t.z2.length, 10);
  assert.strictEqual(t.p.length, 10);
  let best = 0;
  for (let o = 1; o < 10; o++) if (t.p[o] > t.p[best]) best = o;
  assert.strictEqual(t.pred, best);
  assert.ok(Number.isInteger(t.pred) && t.pred >= 0 && t.pred <= 9);
});

test('分块训练等价性：3×{epochs:1,startEpoch:k} ≡ {epochs:3}（权重 bit-identical）', () => {
  const data = makeSyntheticData(5);
  const netA = createNetwork(ARCH, 7);
  const netB = createNetwork(ARCH, 7);
  const optA = createOptState(netA, 'adam');
  const optB = createOptState(netB, 'adam');
  trainEpochs(netA, optA, data, { epochs: 3, seed: 42 });
  const seenEpochs = [];
  for (let k = 0; k < 3; k++) {
    trainEpochs(netB, optB, data, {
      epochs: 1, seed: 42, startEpoch: k,
      onEpoch: (epoch) => { seenEpochs.push(epoch); },
    });
  }
  assert.deepStrictEqual(seenEpochs, [1, 2, 3]);
  assert.deepStrictEqual(netA.w1, netB.w1);
  assert.deepStrictEqual(netA.b1, netB.b1);
  assert.deepStrictEqual(netA.w2, netB.w2);
  assert.deepStrictEqual(netA.b2, netB.b2);
  assert.deepStrictEqual(optA.m, optB.m);
  assert.deepStrictEqual(optA.v, optB.v);
  assert.strictEqual(optA.t, optB.t);
});
