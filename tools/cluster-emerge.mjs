/** 隐藏空间聚类涌现实验：训练前后，32维a1空间中同类/异类距离对比 */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { inflateSync } from 'node:zlib';
import { createNetwork, forward } from '/home/hatch/workspace/nn-anatomy2/js/network.js';
import { createOptState, trainEpochs } from '/home/hatch/workspace/nn-anatomy2/js/trainer.js';

const ROOT = '/home/hatch/workspace/nn-anatomy2';

function decodeGrayPNG(path) {
  const d = readFileSync(path);
  let pos = 8, w = 0, h = 0, idat = [];
  while (pos < d.length) {
    const ln = d.readUInt32BE(pos);
    const typ = d.toString('ascii', pos + 4, pos + 8);
    if (typ === 'IHDR') { w = d.readUInt32BE(pos + 8); h = d.readUInt32BE(pos + 12); }
    else if (typ === 'IDAT') idat.push(d.subarray(pos + 8, pos + 8 + ln));
    pos += 12 + ln;
    if (typ === 'IEND') break;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const gray = new Float64Array(w * h);
  const stride = w + 1;
  let prev = new Array(w).fill(0);
  const paeth = (a, b, c) => {
    const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
  };
  for (let y = 0; y < h; y++) {
    const f = raw[y * stride];
    const row = new Array(w);
    for (let x = 0; x < w; x++) {
      const v = raw[y * stride + 1 + x];
      const a = x > 0 ? row[x - 1] : 0, b = prev[x], c = x > 0 ? prev[x - 1] : 0;
      row[x] = f === 0 ? v : f === 1 ? (v + a) & 255 : f === 2 ? (v + b) & 255
        : f === 3 ? (v + ((a + b) >> 1)) & 255 : (v + paeth(a, b, c)) & 255;
    }
    for (let x = 0; x < w; x++) gray[y * w + x] = row[x] / 255;
    prev = row;
  }
  return gray;
}

function loadTest() {
  const out = [];
  for (let c = 0; c <= 9; c++) {
    const dir = join(ROOT, 'data/sample-digits', 'test', String(c));
    for (const f of readdirSync(dir).filter(f => f.endsWith('.png')).sort())
      out.push({ x: decodeGrayPNG(join(dir, f)), y: c });
  }
  return out;
}

function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

/** 返回 {within, between}：同类平均距离 / 异类平均距离（抽样避免300选2全算太慢） */
function classDistances(vecs) {
  const byClass = Array.from({ length: 10 }, () => []);
  vecs.forEach((v, i) => byClass[v.y].push(v.a1));
  let wSum = 0, wN = 0, bSum = 0, bN = 0;
  for (let c = 0; c < 10; c++) {
    const arr = byClass[c];
    for (let i = 0; i < arr.length; i++)
      for (let j = i + 1; j < arr.length; j++) { wSum += dist(arr[i], arr[j]); wN++; }
    for (let c2 = c + 1; c2 < 10; c2++)
      for (const a of arr) for (const b of byClass[c2]) { bSum += dist(a, b); bN++; }
  }
  return { within: wSum / wN, between: bSum / bN };
}

console.log('loading...');
const test = loadTest();
const train = [];
for (let c = 0; c <= 9; c++) {
  const dir = join(ROOT, 'data/sample-digits', 'train', String(c));
  for (const f of readdirSync(dir).filter(f => f.endsWith('.png')).sort().slice(0, 300))
    train.push({ x: decodeGrayPNG(join(dir, f)), y: c });
}

// 未训练
const net0 = createNetwork({ inputs: 3600, hidden: 32, outputs: 10 }, 42);
const v0 = test.map(s => ({ y: s.y, a1: forward(net0, s.x).a1 }));
const d0 = classDistances(v0);
console.log(`未训练: 同类距离=${d0.within.toFixed(2)} 异类距离=${d0.between.toFixed(2)} 比值=${(d0.within / d0.between).toFixed(3)}`);

// 训练15轮
const net = createNetwork({ inputs: 3600, hidden: 32, outputs: 10 }, 42);
const st = createOptState(net, 'adam');
trainEpochs(net, st, train, { epochs: 15, lr: 0.001, optimizer: 'adam', seed: 7 });
const v1 = test.map(s => ({ y: s.y, a1: forward(net, s.x).a1 }));
const d1 = classDistances(v1);
console.log(`训练后: 同类距离=${d1.within.toFixed(2)} 异类距离=${d1.between.toFixed(2)} 比值=${(d1.within / d1.between).toFixed(3)}`);
