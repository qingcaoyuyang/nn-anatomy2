/** ReLU vs LeakyReLU 死亡神经元对比实验（课堂证据用，不碰产品代码）。
 * 训练循环与 js/trainer.js 逐样本 Adam 完全同构（同 seed=42、同 shuffle 流、同 lr），
 * 唯一变量是激活负半轴斜率 alpha：0 = 普通 ReLU，0.01 = LeakyReLU。
 * 死亡定义：某隐藏节点在全部 300 个测试样本上 z1 <= 0（ReLU 下输出恒 0、梯度恒 0，永不复活）。
 */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { inflateSync } from 'node:zlib';
import { createNetwork } from '/home/hatch/workspace/nn-anatomy2/js/network.js';
import { createRng } from '/home/hatch/workspace/nn-anatomy2/js/rng.js';
import { createAdam, paramCount, adamStep, ADAM_DEFAULT_LR } from '/home/hatch/workspace/nn-anatomy2/js/optimizer.js';

const ROOT = '/home/hatch/workspace/nn-anatomy2';

function decodeGrayPNG(path) {
  const d = readFileSync(path);
  let pos = 8, w = 0, h = 0; const idat = [];
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

function loadSplit(split, perClass) {
  const out = [];
  for (let c = 0; c <= 9; c++) {
    const dir = join(ROOT, 'data/sample-digits', split, String(c));
    const files = readdirSync(dir).filter(f => f.endsWith('.png')).sort().slice(0, perClass);
    for (const f of files) out.push({ x: decodeGrayPNG(join(dir, f)), y: c });
  }
  return out;
}

/** 与 js/network.js 同构的前向/反向，唯一区别：alpha 可配。 */
function makeAct(alpha) {
  function forward(net, x) {
    const { inputs: nIn, hidden: nH, outputs: nOut } = net.arch;
    const z1 = new Array(nH), a1 = new Array(nH);
    for (let h = 0; h < nH; h++) {
      let s = net.b1[h];
      const row = h * nIn;
      for (let xi = 0; xi < nIn; xi++) s += net.w1[row + xi] * x[xi];
      z1[h] = s;
      a1[h] = s > 0 ? s : alpha * s;
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
    for (let o = 0; o < nOut; o++) { const e = Math.exp(z2[o] - max); p[o] = e; sum += e; }
    for (let o = 0; o < nOut; o++) p[o] /= sum;
    return { z1, a1, z2, p };
  }
  function backward(net, x, y, fwd) {
    const { inputs: nIn, hidden: nH, outputs: nOut } = net.arch;
    const dz2 = new Array(nOut);
    for (let o = 0; o < nOut; o++) dz2[o] = (o === y ? 1 : 0) - fwd.p[o];
    const gW2 = new Array(nOut * nH), gb2 = new Array(nOut);
    for (let o = 0; o < nOut; o++) {
      gb2[o] = dz2[o];
      const row = o * nH;
      for (let h = 0; h < nH; h++) gW2[row + h] = dz2[o] * fwd.a1[h];
    }
    const dz1 = new Array(nH), gW1 = new Array(nH * nIn), gb1 = new Array(nH);
    for (let h = 0; h < nH; h++) {
      let s = 0;
      for (let o = 0; o < nOut; o++) s += net.w2[o * nH + h] * dz2[o];
      const dAct = fwd.z1[h] > 0 ? 1 : alpha;
      dz1[h] = s * dAct;
      gb1[h] = dz1[h];
      const row = h * nIn;
      for (let xi = 0; xi < nIn; xi++) gW1[row + xi] = dz1[h] * x[xi];
    }
    return { gW1, gb1, gW2, gb2 };
  }
  return { forward, backward };
}

function shuffledIndices(n, rng) {
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng.nextF64() * (i + 1));
    const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  return idx;
}

function train(net, samples, alpha, epochs, seed) {
  const { forward, backward } = makeAct(alpha);
  const st = createAdam(paramCount(net));
  const n = samples.length, lr = ADAM_DEFAULT_LR;
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const rng = createRng(seed * 100003 + epoch);
    const order = shuffledIndices(n, rng);
    for (const i of order) {
      const s = samples[i];
      const fwd = forward(net, s.x);
      const grads = backward(net, s.x, s.y, fwd);
      adamStep(net, st, grads, lr);
    }
  }
}

function evaluate(net, samples, alpha) {
  const { forward } = makeAct(alpha);
  let ok = 0;
  for (const s of samples) {
    const { p } = forward(net, s.x);
    let best = 0;
    for (let o = 1; o < 10; o++) if (p[o] > p[best]) best = o;
    if (best === s.y) ok++;
  }
  return ok / samples.length;
}

/** 死亡神经元：全部测试样本上 z1 <= 0 的隐藏节点下标。 */
function deadNeurons(net, samples) {
  const { forward } = makeAct(0.01); // 前向只取 z1，与 alpha 无关
  const { hidden: nH } = net.arch;
  const alive = new Array(nH).fill(false);
  for (const s of samples) {
    const { z1 } = forward(net, s.x);
    for (let h = 0; h < nH; h++) if (z1[h] > 0) alive[h] = true;
  }
  const dead = [];
  for (let h = 0; h < nH; h++) if (!alive[h]) dead.push(h);
  return dead;
}

console.log('loading data...');
const trainSet = loadSplit('train', 300);
const testSet = loadSplit('test', 30);
console.log(`train=${trainSet.length} test=${testSet.length}`);

for (const alpha of [0, 0.01]) {
  const t0 = Date.now();
  const net = createNetwork({ inputs: 3600, hidden: 32, outputs: 10 }, 42);
  train(net, trainSet, alpha, 15, 7);
  const acc = evaluate(net, testSet, alpha);
  const dead = deadNeurons(net, testSet);
  const name = alpha === 0 ? 'ReLU(普通)' : 'LeakyReLU(0.01)';
  console.log(`${name}: test acc=${(acc * 100).toFixed(2)}% 死亡神经元=${dead.length}/32` +
    (dead.length ? ` [${dead.join(',')}]` : '') + ` (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
}
