/**
 * 工作区文件系统共享测试套件：node（MemoryBackend）与浏览器（OPFS DirHandleBackend）共用。
 *
 * 用法：`const result = await runWorkspaceSuite(makeBackend, assert);`
 * - makeBackend: 每次调用返回一个全新的空 backend（async 工厂函数）。
 * - assert: 断言对象，需提供 ok / strictEqual / deepStrictEqual / rejects。
 * 返回 { pass, fail, errors: [描述字符串] }，调用方自行决定如何报告。
 */
import { Workspace } from '../js/workspace.js';

function fakePngBytes(seed) {
  // 伪造的 PNG 字节（仅用于"字节一致"校验，不要求是合法 PNG）。
  const bytes = new Uint8Array(64);
  bytes[0] = 0x89; bytes[1] = 0x50; bytes[2] = 0x4e; bytes[3] = 0x47; // PNG 魔数
  for (let i = 4; i < bytes.length; i++) bytes[i] = (seed * 31 + i * 7) & 0xff;
  return bytes;
}

function fakePngBlob(seed) {
  return new Blob([fakePngBytes(seed)], { type: 'image/png' });
}

async function blobBytes(blob) {
  return new Uint8Array(await blob.arrayBuffer());
}

function demoNet() {
  // 小规模纯数据网络（形状与真实网络一致，数值随意）。
  const arch = { inputs: 169, hidden: 16, outputs: 10 };
  const seq = (n, s) => Array.from({ length: n }, (_, i) => (i + 1) * s);
  return {
    arch,
    w1: seq(arch.hidden * arch.inputs, 0.001),
    b1: seq(arch.hidden, 0.01),
    w2: seq(arch.outputs * arch.hidden, 0.002),
    b2: seq(arch.outputs, 0.02),
  };
}

function demoOptimizerState() {
  return { m: [0.1, 0.2, 0.3], v: [0.01, 0.02, 0.03], t: 7 };
}

function demoMetrics() {
  return { epochsTrained: 3, samplesSeen: 1500, history: [0.51, 0.32, 0.21] };
}

const CASES = [
  {
    name: 'create 初始化目录结构与 workspace.json',
    run: async (makeBackend, assert) => {
      const ws = new Workspace(await makeBackend());
      await ws.initialize();

      const blob = await ws.backend.readFile('workspace.json');
      const meta = JSON.parse(await blob.text());
      assert.strictEqual(meta.version, 1);
      assert.strictEqual(meta.app, 'nn-anatomy2');
      assert.ok(typeof meta.createdAt === 'string');

      for (const split of ['train', 'test']) {
        for (let d = 0; d <= 9; d++) {
          assert.ok(await ws.backend.exists(`datasets/${split}/${d}`), `缺少目录 datasets/${split}/${d}`);
        }
        const manifestBlob = await ws.backend.readFile(`datasets/${split}/manifest.json`);
        assert.deepStrictEqual(JSON.parse(await manifestBlob.text()), []);
      }
      assert.ok(await ws.backend.exists('models'));
      assert.ok(await ws.backend.exists('exports'));
    },
  },
  {
    name: 'open 缺少 workspace.json 时抛错',
    run: async (makeBackend, assert) => {
      const ws = new Workspace(await makeBackend()); // 空 backend，未 initialize
      await assert.rejects(() => ws.loadFromDisk(), '空目录 open 应抛错');
    },
  },
  {
    name: 'addSample 写入 PNG 并更新 manifest',
    run: async (makeBackend, assert) => {
      const ws = new Workspace(await makeBackend());
      await ws.initialize();

      const bytes = fakePngBytes(3);
      const id = await ws.addSample('train', '5', fakePngBlob(3),
        { width: 260, height: 260, source: 'handwritten' });
      assert.ok(/^\d+-\d+$/.test(id), `id 格式异常: ${id}`);

      const samples = ws.listSamples('train');
      assert.strictEqual(samples.length, 1);
      const entry = samples[0];
      assert.strictEqual(entry.id, id);
      assert.strictEqual(entry.label, '5');
      assert.strictEqual(entry.file, `${id}.png`);
      assert.strictEqual(entry.source, 'handwritten');
      assert.ok(typeof entry.createdAt === 'string');

      // 磁盘字节与写入一致
      const back = await blobBytes(await ws.backend.readFile(`datasets/train/5/${id}.png`));
      assert.deepStrictEqual(Array.from(back), Array.from(bytes));

      // manifest 落盘
      const manifestBlob = await ws.backend.readFile('datasets/train/manifest.json');
      assert.deepStrictEqual(JSON.parse(await manifestBlob.text()), [entry]);
    },
  },
  {
    name: 'removeSample 删除文件与 manifest 条目',
    run: async (makeBackend, assert) => {
      const ws = new Workspace(await makeBackend());
      await ws.initialize();

      const idA = await ws.addSample('test', '0', fakePngBlob(1), { width: 13, height: 13, source: 'mnist-import' });
      const idB = await ws.addSample('test', '0', fakePngBlob(2), { width: 13, height: 13, source: 'mnist-import' });

      await ws.removeSample('test', idA);
      assert.strictEqual(await ws.backend.exists(`datasets/test/0/${idA}.png`), false);
      const samples = ws.listSamples('test');
      assert.strictEqual(samples.length, 1);
      assert.strictEqual(samples[0].id, idB);

      await assert.rejects(() => ws.removeSample('test', '不存在的-id'), '删除不存在的样本应抛错');
    },
  },
  {
    name: 'saveModel/loadModel 往返一致',
    run: async (makeBackend, assert) => {
      const ws = new Workspace(await makeBackend());
      await ws.initialize();

      const net = demoNet();
      const optimizerState = demoOptimizerState();
      const metrics = demoMetrics();
      await ws.saveModel('演示模型 v1', net, optimizerState, metrics);

      const models = await ws.listModels();
      assert.strictEqual(models.length, 1);
      assert.strictEqual(models[0].name, '演示模型 v1');

      const loaded = await ws.loadModel('演示模型 v1');
      assert.deepStrictEqual(loaded.net, net);
      assert.deepStrictEqual(loaded.architecture, net.arch);
      assert.deepStrictEqual(loaded.optimizerState, optimizerState);
      assert.deepStrictEqual(loaded.metrics, metrics);
      assert.ok(typeof loaded.trainedAt === 'string');

      await assert.rejects(() => ws.loadModel('不存在的模型'), '加载不存在的模型应抛错');
    },
  },
  {
    name: 'deleteModel 后 listModels 为空',
    run: async (makeBackend, assert) => {
      const ws = new Workspace(await makeBackend());
      await ws.initialize();

      await ws.saveModel('模型A', demoNet(), demoOptimizerState(), demoMetrics());
      await ws.saveModel('模型B', demoNet(), demoOptimizerState(), demoMetrics());
      assert.strictEqual((await ws.listModels()).length, 2);

      await ws.deleteModel('模型A');
      assert.strictEqual((await ws.listModels()).length, 1);
      assert.strictEqual(await ws.backend.exists('models/模型A.nnmodel.json'), false);

      await ws.deleteModel('模型B');
      assert.deepStrictEqual(await ws.listModels(), []);

      await assert.rejects(() => ws.deleteModel('模型A'), '删除不存在的模型应抛错');
    },
  },
  {
    name: '同名 saveModel 覆盖旧文件',
    run: async (makeBackend, assert) => {
      const ws = new Workspace(await makeBackend());
      await ws.initialize();

      const net1 = demoNet();
      const net2 = demoNet();
      net2.w1[0] = 123.456; // 确保不同
      await ws.saveModel('覆盖测试', net1, demoOptimizerState(), demoMetrics());
      await ws.saveModel('覆盖测试', net2, demoOptimizerState(), demoMetrics());

      assert.strictEqual((await ws.listModels()).length, 1);
      const loaded = await ws.loadModel('覆盖测试');
      assert.strictEqual(loaded.net.w1[0], 123.456);
    },
  },
  {
    name: '模型名做文件名安全过滤',
    run: async (makeBackend, assert) => {
      const ws = new Workspace(await makeBackend());
      await ws.initialize();

      const evil = 'a/b\\c:d*e?f"g<h>i|j';
      await ws.saveModel(evil, demoNet(), demoOptimizerState(), demoMetrics());

      const files = await ws.backend.list('models');
      assert.strictEqual(files.length, 1);
      assert.ok(!/[/\\:*?"<>|]/.test(files[0].name), `文件名含非法字符: ${files[0].name}`);
      assert.ok(files[0].name.endsWith('.nnmodel.json'));

      // 显示名保持原样
      const models = await ws.listModels();
      assert.strictEqual(models[0].name, evil);
      const loaded = await ws.loadModel(evil);
      assert.deepStrictEqual(loaded.net.w1, demoNet().w1);

      await assert.rejects(() => ws.saveModel('   ', demoNet(), demoOptimizerState(), demoMetrics()), '空名称应抛错');
    },
  },
  {
    name: 'addSample 参数校验',
    run: async (makeBackend, assert) => {
      const ws = new Workspace(await makeBackend());
      await ws.initialize();
      const opts = { width: 260, height: 260, source: 'handwritten' };

      await assert.rejects(() => ws.addSample('valid', '5', fakePngBlob(1), opts), '非法 split 应抛错');
      await assert.rejects(() => ws.addSample('train', 'x', fakePngBlob(1), opts), '非法 label 应抛错');
      await assert.rejects(() => ws.addSample('train', '10', fakePngBlob(1), opts), '两位 label 应抛错');
      await assert.rejects(() => ws.addSample('train', '5', fakePngBlob(1), { ...opts, source: 'nope' }), '非法 source 应抛错');
      await assert.rejects(() => ws.addSample('train', '5', 'not-a-blob', opts), '非 Blob 应抛错');
      await assert.rejects(async () => ws.listSamples('oops'), '非法 split 的 listSamples 应抛错');
    },
  },
];

/**
 * 运行全部用例。返回 { pass, fail, errors }。
 */
export async function runWorkspaceSuite(makeBackend, assert) {
  let pass = 0;
  let fail = 0;
  const errors = [];
  for (const c of CASES) {
    try {
      await c.run(makeBackend, assert);
      pass++;
    } catch (e) {
      fail++;
      errors.push(`${c.name}: ${e && e.message ? e.message : String(e)}`);
    }
  }
  return { pass, fail, errors };
}
