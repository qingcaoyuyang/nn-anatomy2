/**
 * decodeSamplePixels 的 node 测试：覆盖手写 260×260 全解码链路
 * （RGBA → 灰度 → 取反 → preprocess 居中降采样），以及 mnist-import 极性。
 *
 * 说明：浏览器里 createImageBitmap 解码 PNG 的一步无法在 node 覆盖
 * （且本环境 headless-shell 对 260×260 PNG 解码会挂起，见 AGENTS.md）；
 * 这里用程序化绘制的 RGBA 直接测解码纯逻辑，真机 Chrome 再补端到端。
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decodeSamplePixels } from '../js/dataset.js';

/** 在 260×260 白底上画一个粗"7"（黑），返回 RGBA。 */
function drawSeven260() {
  const W = 260, H = 260;
  const data = new Uint8ClampedArray(W * H * 4);
  data.fill(255);
  const px = (x, y, v) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const i = (y * W + x) * 4;
    data[i] = data[i + 1] = data[i + 2] = v;
  };
  const hline = (x0, x1, y, t) => {
    for (let x = x0; x <= x1; x++) for (let k = -t; k <= t; k++) px(x, y + k, 0);
  };
  const diag = (x0, y0, x1, y1, t) => {
    const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0));
    for (let s = 0; s <= steps; s++) {
      const x = Math.round(x0 + ((x1 - x0) * s) / steps);
      const y = Math.round(y0 + ((y1 - y0) * s) / steps);
      for (let a = -t; a <= t; a++) for (let b = -t; b <= t; b++) px(x + a, y + b, 0);
    }
  };
  hline(60, 200, 50, 9);      // 顶横
  diag(200, 50, 110, 220, 9); // 斜竖
  return { data, W, H };
}

test('handwritten：260×260 黑字白纸 → 3600 维墨=1，重心居中', () => {
  const { data, W, H } = drawSeven260();
  const x = decodeSamplePixels(data, W, H, 'handwritten');
  assert.strictEqual(x.length, 3600);
  for (const v of x) assert.ok(v >= 0 && v <= 1, `越界 ${v}`);
  const positive = x.filter((v) => v > 0.05).length;
  assert.ok(positive > 5, `笔画应存活，实际正值 ${positive} 个`);
  // 重心应在中心附近（7 的笔画大致居中）
  let m = 0, sx = 0, sy = 0;
  for (let i = 0; i < 3600; i++) {
    m += x[i]; sx += x[i] * (i % 60); sy += x[i] * Math.floor(i / 60);
  }
  const cx = sx / m, cy = sy / m;
  assert.ok(Math.abs(cx - 29.5) < 4 && Math.abs(cy - 29.5) < 4, `重心偏离中心：(${cx.toFixed(2)}, ${cy.toFixed(2)})`);
});

test('handwritten：全空白纸 → 全零输入', () => {
  const data = new Uint8ClampedArray(260 * 260 * 4).fill(255);
  const x = decodeSamplePixels(data, 260, 260, 'handwritten');
  assert.strictEqual(x.length, 3600);
  assert.ok(x.every((v) => v === 0), '空白纸应全零');
});

test('mnist-import：60×60 白字黑底 → x = gray/255（不取反）', () => {
  const data = new Uint8ClampedArray(60 * 60 * 4); // 全黑底
  // 中心画一个白点
  const i = (30 * 60 + 30) * 4;
  data[i] = data[i + 1] = data[i + 2] = 255;
  data[i + 3] = 255;
  for (let k = 0; k < data.length; k += 4) if (data[k + 3] !== 255) data[k + 3] = 255;
  const x = decodeSamplePixels(data, 60, 60, 'mnist-import');
  assert.strictEqual(x.length, 3600);
  assert.strictEqual(x[30 * 60 + 30], 1, '白字应为 1（不取反）');
  assert.strictEqual(x[0], 0, '黑底应为 0');
});

test('mnist-import：尺寸不是 60×60 抛错；未知来源抛错', () => {
  const data = new Uint8ClampedArray(14 * 14 * 4).fill(255);
  assert.throws(() => decodeSamplePixels(data, 14, 14, 'mnist-import'), /60×60/);
  assert.throws(() => decodeSamplePixels(data, 14, 14, 'nope'), /未知/);
});

test('augmented：走 handwritten 同路（取反+预处理）', () => {
  const { data, W, H } = drawSeven260();
  const x = decodeSamplePixels(data, W, H, 'augmented');
  assert.strictEqual(x.length, 3600);
  assert.ok(x.some((v) => v > 0.05), '笔画应存活');
});
