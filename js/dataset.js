/**
 * 示例数据集：一键导入打包好的 MNIST 样本（13×13 PNG）。
 *
 * 纯浏览器架构：`data/sample-digits/{train|test}/{label}/{i:03d}.png`
 * 共 1300 张（train 每类 100，test 每类 30），由 tools/build-dataset.py 一次性生成并随仓库打包。
 *
 * ES module，零依赖，不直接碰 DOM；fetch 可注入以便测试。
 */

/** 打包数据集在应用内的根目录（相对路径）。 */
export const SAMPLE_DATASET_ROOT = 'data/sample-digits';
/** 每个划分每类包含的样本数。 */
export const SAMPLE_COUNTS = { train: 100, test: 30 };
/** 打包样本的宽高（13×13 灰度 PNG）。 */
export const SAMPLE_SIZE = 13;

/**
 * 纯函数：按固定命名约定生成全部 1300 个相对路径。
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
