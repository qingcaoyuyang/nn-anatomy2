import { test } from 'node:test';
import assert from 'node:assert/strict';
import { centerOfMass, resampleTo13, preprocess } from '../js/preprocess.js';

const W = 260;
const H = 260;

function zeros() {
  return new Uint8Array(W * H);
}

function singlePixel() {
  const img = zeros();
  img[201 * W + 37] = 255;
  return img;
}

/** 对角线渐变：v(x,y) = floor((x+y)*255/(w+h−2))（与 Rust 测试同公式）。 */
function diagonalGradient() {
  const img = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      img[y * W + x] = Math.floor(((x + y) * 255) / (W + H - 2));
    }
  }
  return img;
}

test('zeros: center of mass = canvas center', () => {
  assert.deepStrictEqual(centerOfMass(zeros(), W, H), [129.5, 129.5]);
});

test('single bright pixel: center of mass = pixel coords', () => {
  assert.deepStrictEqual(centerOfMass(singlePixel(), W, H), [37, 201]);
});

test('centered + downsampled: center of mass ≈ (6,6) on 13×13 grid', () => {
  const out = resampleTo13(singlePixel(), W, H);
  assert.strictEqual(out.length, 169);
  // 回归 guard：单亮点经管线后必须恰好留下一个正值。
  // 若降采样把亮点混叠丢了，centerOfMass 的 m=0 fallback 会恰好返回 (6,6)，
  // 让下面的断言"空"通过——这个计数断言把那种静默退化锁死。
  const positive = out.filter((v) => v > 0).length;
  assert.strictEqual(positive, 1, `single pixel should survive resampling, got ${positive} positive elements`);
  const [cx, cy] = centerOfMass(out, 13, 13);
  assert.ok(Math.abs(cx - 6) < 1e-6, `cx should be ≈ 6, got ${cx}`);
  assert.ok(Math.abs(cy - 6) < 1e-6, `cy should be ≈ 6, got ${cy}`);
});

test('outputs all in [0, 1]', () => {
  const out = preprocess(diagonalGradient(), W, H, 13);
  assert.strictEqual(out.length, 169);
  for (const v of out) {
    assert.ok(v >= 0 && v <= 1, `output out of range: ${v}`);
  }
  assert.ok([...out].some((v) => v > 0), 'gradient image should have positive outputs');
});

test('default outSize=60: 3600 dims, values in [0, 1]', () => {
  const out = preprocess(diagonalGradient(), W, H);
  assert.strictEqual(out.length, 3600);
  for (const v of out) {
    assert.ok(v >= 0 && v <= 1, `output out of range: ${v}`);
  }
  assert.ok([...out].some((v) => v > 0), 'gradient image should have positive outputs');
});

test('preprocessWithInfo: x 与 preprocess(60) 一致，并返回重心/平移', async () => {
  const { preprocessWithInfo } = await import('../js/preprocess.js');
  const img = diagonalGradient();
  const info = preprocessWithInfo(img, W, H, 60);
  const plain = preprocess(img, W, H, 60);
  assert.strictEqual(info.x.length, 3600);
  for (let i = 0; i < 3600; i++) {
    assert.strictEqual(info.x[i], plain[i], `index ${i} mismatch`);
  }
  assert.ok(Number.isFinite(info.cx) && Number.isFinite(info.cy), 'cx/cy finite');
  assert.ok(Number.isFinite(info.dx) && Number.isFinite(info.dy), 'dx/dy finite');
});
