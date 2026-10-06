/**
 * 示例数据集：一键导入打包好的 MNIST 样本（60×60 PNG）。
 *
 * 纯浏览器架构：`data/sample-digits/{train|test}/{label}/{i:03d}.png`
 * 共 3300 张（train 每类 300，test 每类 30），由 tools/build-dataset.py 一次性生成并随仓库打包。
 *
 * ES module，零依赖，不直接碰 DOM；fetch 可注入以便测试。
 */

import { preprocess } from './preprocess.js';

/** 打包数据集在应用内的根目录（相对路径）。 */
export const SAMPLE_DATASET_ROOT = 'data/sample-digits';
/** 每个划分每类包含的样本数。 */
export const SAMPLE_COUNTS = { train: 300, test: 30 };
/** 打包样本的宽高（60×60 灰度 PNG）。 */
export const SAMPLE_SIZE = 60;

/**
 * 纯函数：按固定命名约定生成全部 3300 个相对路径。
 * @returns {Array<{ split: 'train'|'test', label: string, path: string }>}
 *          顺序为 split（train 在前）→ label（'0'..'9'）→ i（000 起）。
 */
export function samplePaths() {
  const out = [];
  for (const split of ['train', 'test']) {
    const perClass = SAMPLE_COUNTS[split];
    for (let d = 0; d <= 9; d++) {
      const label = String(d);
      for (let i = 0; i < perClass; i++) {
        const name = `${String(i).padStart(3, '0')}.png`;
        out.push({ split, label, path: `${SAMPLE_DATASET_ROOT}/${split}/${label}/${name}` });
      }
    }
  }
  return out;
}

/**
 * 主流程：fetch 打包好的 PNG 并逐张写入工作区。
 *
 * @param {import('./workspace.js').Workspace} workspace 目标工作区。
 * @param {{ fetchImpl?: Function, onProgress?: (done: number, total: number) => void }} opts
 *        fetchImpl 默认为全局 fetch；onProgress 每完成一张调用一次，可省略。
 * @throws 中途某张 fetch 失败（resp.ok 为 false）即抛错中断，错误信息包含该 path。
 */
export async function importSampleDataset(workspace, { fetchImpl = fetch, onProgress } = {}) {
  const paths = samplePaths();
  const total = paths.length;
  let done = 0;
  for (const { split, label, path } of paths) {
    const resp = await fetchImpl(path);
    if (!resp || !resp.ok) {
      const status = resp && resp.status !== undefined ? `（HTTP ${resp.status}）` : '';
      throw new Error(`示例数据集下载失败：${path}${status}。请检查网络后重试。`);
    }
    const blob = await resp.blob();
    await workspace.addSample(split, label, blob, {
      width: SAMPLE_SIZE,
      height: SAMPLE_SIZE,
      source: 'mnist-import',
    });
    done += 1;
    if (typeof onProgress === 'function') onProgress(done, total);
  }
}

/* —— 样本解码（Task 11） —— */

/** 网络输入维度（60×60）。 */
export const INPUT_DIM = 60 * 60;

/**
 * Load-bearing 极性约定（网络输入统一"墨=1"）：
 * - handwritten（260×260 黑字白纸 PNG）：灰度 g → 墨 255−g → preprocess(ink, w, h)
 * - mnist-import（60×60 白字黑底 PNG）：x = g/255（不取反、不居中，已是标准输入）
 * - augmented：来自手写 PNG 的增强，极性同 handwritten
 */

/** RGBA → 灰度 Uint8Array（0~255）。 */
function toGray(data) {
  const n = data.length / 4;
  const gray = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    gray[i] = Math.round((data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3);
  }
  return gray;
}

/**
 * 按来源极性把像素解码为 3600 维"墨=1"输入。
 * @param {Uint8ClampedArray} data RGBA 像素
 * @param {number} width 宽度
 * @param {number} height 高度
 * @param {string} source manifest 中的来源字段
 * @returns {Array|Float64Array} 3600 个 0~1
 * @throws 未知来源或 mnist-import 尺寸不是 60×60 时抛错（调用方按损坏样本跳过）
 * （导出以便 node 测试覆盖手写 260×260 解码链路；loadSamples 在浏览器里调它。）
 */
export function decodeSamplePixels(data, width, height, source) {
  const gray = toGray(data);
  if (source === 'mnist-import') {
    if (width !== 60 || height !== 60 || gray.length !== INPUT_DIM) {
      throw new Error(`mnist-import 样本尺寸异常：${width}×${height}（应为 60×60）。`);
    }
    const x = new Array(INPUT_DIM);
    for (let i = 0; i < INPUT_DIM; i++) x[i] = gray[i] / 255;
    return x;
  }
  if (source === 'handwritten' || source === 'augmented') {
    const w = width || 0;
    const h = height || 0;
    if (gray.length !== w * h) {
      throw new Error(`样本像素数与尺寸不符：${gray.length} vs ${w}×${h}。`);
    }
    // 黑字白纸 → 墨=1：取反后再走标准预处理管线（居中 + 60×60 + 归一化）
    const ink = new Uint8Array(gray.length);
    for (let i = 0; i < gray.length; i++) ink[i] = 255 - gray[i];
    return preprocess(ink, w, h);
  }
  throw new Error(`未知的样本来源：${source}。`);
}

/**
 * 从工作区加载某划分的全部样本（仅浏览器：依赖 createImageBitmap + document）。
 * 损坏条目（读像素失败/解码失败）会被 console.warn 并跳过，不中断整体。
 * @param {import('./workspace.js').Workspace} workspace
 * @param {'train'|'test'} split
 * @returns {Promise<Array<{ x: Array|Float64Array, y: number, id: string }>>}
 *          顺序与 manifest 一致；y 为数字 0~9
 */
export async function loadSamples(workspace, split) {
  if (typeof createImageBitmap === 'undefined' || typeof document === 'undefined') {
    throw new Error('loadSamples 仅在浏览器环境中可用。');
  }
  const entries = workspace.listSamples(split);
  const out = [];
  for (const entry of entries) {
    try {
      const { data, width, height } = await workspace.readSamplePixels(split, entry.id);
      const x = decodeSamplePixels(data, width, height, entry.source);
      const y = Number(entry.label);
      if (!Number.isInteger(y) || y < 0 || y > 9) {
        throw new Error(`标签非法：${entry.label}（应为 0~9 整数）。`);
      }
      out.push({ x, y, id: entry.id });
    } catch (e) {
      console.warn(`跳过损坏样本 ${entry.id}：`, e && e.message ? e.message : e);
    }
  }
  return out;
}
