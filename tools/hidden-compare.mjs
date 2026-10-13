/** 隐藏层神经元数量对比实验：hidden = 1/2/4/8/16/32，15 epochs，同一 seed */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { inflateSync } from 'node:zlib';
import { createNetwork } from '/home/hatch/workspace/nn-anatomy2/js/network.js';
import { createOptState } from '/home/hatch/workspace/nn-anatomy2/js/trainer.js';
import { trainEpochs, evaluate } from '/home/hatch/workspace/nn-anatomy2/js/trainer.js';

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

function loadSplit(split, perClass) {
  const out = [];
  for (let c = 0; c <= 9; c++) {
    const dir = join(ROOT, 'data/sample-digits', split, String(c));
    const files = readdirSync(dir).filter(f => f.endsWith('.png')).sort().slice(0, perClass);
    for (const f of files) out.push({ x: decodeGrayPNG(join(dir, f)), y: c });
  }
  return out;
}

console.log('loading data...');
const train = loadSplit('train', 300);
const test = loadSplit('test', 30);
console.log(`train=${train.length} test=${test.length}`);

for (const hidden of [1, 2, 4, 8, 16, 32]) {
  const t0 = Date.now();
  const net = createNetwork({ inputs: 3600, hidden, outputs: 10 }, 42);
  const st = createOptState(net, 'adam');
  trainEpochs(net, st, train, { epochs: 15, lr: 0.001, optimizer: 'adam', seed: 7 });
  const { acc } = evaluate(net, test);
  console.log(`hidden=${hidden}: test acc=${(acc * 100).toFixed(2)}% (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
}
