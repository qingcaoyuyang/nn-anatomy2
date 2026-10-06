import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MemoryBackend, Workspace } from '../js/workspace.js';
import { samplePaths, importSampleDataset } from '../js/dataset.js';

test('samplePaths()：共 3300 条，每类计数正确', () => {
  const paths = samplePaths();
  assert.strictEqual(paths.length, 3300, `期望 3300 条，实际 ${paths.length}`);

  const count = { train: {}, test: {} };
  for (const { split, label } of paths) {
    assert.ok(split === 'train' || split === 'test', `非法 split：${split}`);
    assert.ok(/^[0-9]$/.test(label), `非法 label：${label}`);
    count[split][label] = (count[split][label] || 0) + 1;
  }
  for (let d = 0; d <= 9; d++) {
    const label = String(d);
    assert.strictEqual(count.train[label], 300, `train/${label} 期望 300，实际 ${count.train[label]}`);
    assert.strictEqual(count.test[label], 30, `test/${label} 期望 30，实际 ${count.test[label]}`);
  }
  // 顺序：split → label → i（train 在前）
  assert.strictEqual(paths[0].split, 'train');
  assert.strictEqual(paths[999].split, 'train');
  assert.strictEqual(paths[3000].split, 'test');
  assert.strictEqual(paths[3299].split, 'test');
});

test('samplePaths()：路径格式 data/sample-digits/{split}/{label}/{iii}.png', () => {
  const paths = samplePaths();
  const re = /^data\/sample-digits\/(train|test)\/([0-9])\/([0-9]{3})\.png$/;
  const seen = { train: {}, test: {} };
  for (const { split, label, path } of paths) {
    const m = re.exec(path);
    assert.ok(m, `路径格式不符：${path}`);
    assert.strictEqual(m[1], split);
    assert.strictEqual(m[2], label);
    const i = Number(m[3]);
    const bucket = seen[split][label] || (seen[split][label] = []);
    bucket.push(i);
  }
  // 类内序号连续 000..099（train）/ 000..029（test）
  for (let d = 0; d <= 9; d++) {
    const label = String(d);
    assert.deepStrictEqual(seen.train[label].slice().sort((a, b) => a - b),
      Array.from({ length: 300 }, (_, k) => k), `train/${label} 序号应为 000..299`);
    assert.deepStrictEqual(seen.test[label].slice().sort((a, b) => a - b),
      Array.from({ length: 30 }, (_, k) => k), `test/${label} 序号应为 000..029`);
  }
  // 顺序：split（train 在前）→ label（'0'..'9'）→ i（000 起）
  const expected = [];
  for (const split of ['train', 'test']) {
    const n = split === 'train' ? 300 : 30;
    for (let d = 0; d <= 9; d++) {
      for (let i = 0; i < n; i++) {
        expected.push(`data/sample-digits/${split}/${d}/${String(i).padStart(3, '0')}.png`);
      }
    }
  }
  assert.deepStrictEqual(paths.map((p) => p.path), expected,
    'samplePaths() 应按 split→label→i 排序（train 在前）');
});

test('importSampleDataset：3300 张全部写入工作区，来源标记正确', async () => {
  const ws = new Workspace(new MemoryBackend());
  const fetched = [];
  const fakeFetch = async (path) => {
    fetched.push(path);
    return { ok: true, blob: async () => new Blob(['fake-png-bytes']) };
  };
  const progress = [];
  await importSampleDataset(ws, { fetchImpl: fakeFetch, onProgress: (d, t) => progress.push([d, t]) });

  assert.strictEqual(fetched.length, 3300, `应 fetch 3300 次，实际 ${fetched.length}`);
  // fetch 的路径与 samplePaths() 一致
  assert.deepStrictEqual(fetched, samplePaths().map((p) => p.path));

  const train = ws.listSamples('train');
  const test = ws.listSamples('test');
  assert.strictEqual(train.length, 3000, `train 期望 3000，实际 ${train.length}`);
  assert.strictEqual(test.length, 300, `test 期望 300，实际 ${test.length}`);

  const dist = (list) => {
    const m = {};
    for (const e of list) m[e.label] = (m[e.label] || 0) + 1;
    return m;
  };
  for (let d = 0; d <= 9; d++) {
    const label = String(d);
    assert.strictEqual(dist(train)[label], 300, `train label ${label} 期望 300`);
    assert.strictEqual(dist(test)[label], 30, `test label ${label} 期望 30`);
  }
  for (const e of [...train, ...test]) {
    assert.strictEqual(e.source, 'mnist-import', '来源应为 mnist-import');
    assert.strictEqual(e.width, 60);
    assert.strictEqual(e.height, 60);
  }
  // onProgress 被调用 3300 次，最后一次 (3300, 3300)
  assert.strictEqual(progress.length, 3300);
  assert.deepStrictEqual(progress[3299], [3300, 3300]);
});

test('importSampleDataset：无 onProgress 也能跑', async () => {
  const ws = new Workspace(new MemoryBackend());
  const fakeFetch = async () => ({ ok: true, blob: async () => new Blob(['x']) });
  await importSampleDataset(ws, { fetchImpl: fakeFetch });
  assert.strictEqual(ws.listSamples('train').length, 3000);
});

test('importSampleDataset：某路径 fetch 失败 → 抛错且错误信息含该 path', async () => {
  const ws = new Workspace(new MemoryBackend());
  const badPath = 'data/sample-digits/train/7/042.png';
  const fakeFetch = async (path) => {
    if (path === badPath) return { ok: false, status: 404, blob: async () => { throw new Error('不应被调用'); } };
    return { ok: true, blob: async () => new Blob(['x']) };
  };
  await assert.rejects(
    () => importSampleDataset(ws, { fetchImpl: fakeFetch }),
    (err) => {
      assert.ok(err instanceof Error, '应抛出 Error');
      assert.ok(String(err.message).includes(badPath), `错误信息应包含路径 ${badPath}，实际：${err.message}`);
      return true;
    },
  );
});
