import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRng } from '../js/rng.js';

test('同 seed 产生完全相同的序列', () => {
  const a = createRng(12345);
  const b = createRng(12345);
  for (let i = 0; i < 100; i++) {
    assert.strictEqual(a.nextU32(), b.nextU32());
  }
  const c = createRng(12345);
  const d = createRng(12345);
  for (let i = 0; i < 50; i++) {
    assert.strictEqual(c.nextF64(), d.nextF64());
    assert.strictEqual(c.nextNormal(), d.nextNormal());
  }
});

test('不同 seed 产生不同序列', () => {
  const a = createRng(1);
  const b = createRng(2);
  let differ = false;
  for (let i = 0; i < 10; i++) {
    if (a.nextU32() !== b.nextU32()) { differ = true; break; }
  }
  assert.ok(differ, '不同 seed 的序列应当不同');
});

test('nextF64 范围为 [0, 1)', () => {
  const rng = createRng(7);
  for (let i = 0; i < 10000; i++) {
    const v = rng.nextF64();
    assert.ok(v >= 0 && v < 1, `nextF64 越界: ${v}`);
  }
});

test('seed 截断到 u32：2^32 ≡ 0，-1 ≡ 2^32-1', () => {
  const a = createRng(0);
  const b = createRng(2 ** 32);
  const c = createRng(0xDEADBEEF);
  const d = createRng(0xDEADBEEF + 2 ** 32 * 3);
  const e = createRng(-1);
  const f = createRng(2 ** 32 - 1);
  for (let i = 0; i < 20; i++) {
    assert.strictEqual(a.nextU32(), b.nextU32());
    assert.strictEqual(c.nextU32(), d.nextU32());
    assert.strictEqual(e.nextU32(), f.nextU32());
  }
});

test('nextNormal 大致服从标准正态（均值≈0，方差≈1）', () => {
  const rng = createRng(99);
  const n = 20000;
  let sum = 0, sum2 = 0;
  for (let i = 0; i < n; i++) {
    const v = rng.nextNormal();
    assert.ok(Number.isFinite(v), `nextNormal 非有限值: ${v}`);
    sum += v;
    sum2 += v * v;
  }
  const mean = sum / n;
  const variance = sum2 / n - mean * mean;
  assert.ok(Math.abs(mean) < 0.05, `均值偏离: ${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.05, `方差偏离: ${variance}`);
});

test('mulberry32 已知值：seed=42 的前三个 u32 输出', () => {
  // 独立 Python 实现（标准 mulberry32 算法）交叉验证过的参照值，防止实现漂移
  const rng = createRng(42);
  assert.strictEqual(rng.nextU32(), 0x99E1EF7C);
  assert.strictEqual(rng.nextU32(), 0x72C32B8A);
  assert.strictEqual(rng.nextU32(), 0xDA3B32C0);
});
