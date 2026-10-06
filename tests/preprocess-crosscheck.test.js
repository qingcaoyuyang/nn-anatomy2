import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { preprocess } from '../js/preprocess.js';

// 与 Rust 测试 write_crosscheck_fixture 完全同公式的合成图生成器。
function makeImage(name, w, h) {
  const img = new Uint8Array(w * h);
  if (name === 'single_pixel') {
    img[201 * w + 37] = 255;
  } else if (name === 'diagonal_gradient') {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        img[y * w + x] = Math.floor(((x + y) * 255) / (w + h - 2));
      }
    }
  } else if (name !== 'zeros') {
    throw new Error(`unknown fixture case: ${name}`);
  }
  return img;
}

test('Rust↔JS cross-check: preprocess outputs match, per-element |Δ| < 1e-12', () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const fixture = JSON.parse(
    readFileSync(join(here, 'fixtures/preprocess-crosscheck.json'), 'utf8')
  );
  const { w, h, cases } = fixture;
  assert.strictEqual(w, 260);
  assert.strictEqual(h, 260);
  assert.strictEqual(cases.length, 3);

  let maxDelta = 0;
  for (const c of cases) {
    const js = preprocess(makeImage(c.name, w, h), w, h, 13);
    assert.strictEqual(js.length, 169, `JS output length for ${c.name}`);
    assert.strictEqual(c.output.length, 169, `fixture length for ${c.name}`);
    for (let i = 0; i < 169; i++) {
      const d = Math.abs(js[i] - c.output[i]);
      if (d > maxDelta) maxDelta = d;
      assert.ok(d < 1e-12, `case ${c.name} index ${i}: |Δ| = ${d}`);
    }
  }
  console.log(`max |Δ| across ${cases.length} cases: ${maxDelta}`);
});
