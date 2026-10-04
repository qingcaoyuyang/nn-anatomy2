/**
 * 数据集管理 modal：train/test 页签 + 一键导入示例数据集（进度条）+ 缩略图网格 + 删除。
 *
 * ES module，零 npm 依赖。缩略图直接画原始 PNG 像素（不做极性变换，所见即文件）。
 * 导入/删除后调用 api.notifySamplesChanged()，由应用壳刷新计数并触发面板重评。
 */
import { importSampleDataset } from '../dataset.js';

/** 缩略图边长（CSS 像素）。 */
const THUMB = 26;

/**
 * 初始化数据集管理 modal（挂到 document.body，单例）。
 * @param {object} api { getWorkspace(), notifySamplesChanged() }
 * @returns {{ open():Promise<void>, close():void, refresh():Promise<void> }}
 */
export function initDatasetViewer(api) {
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay hidden';
  overlay.id = 'dataset-modal';
  overlay.innerHTML = `
    <div class="modal-card" role="dialog" aria-label="数据集管理">
      <div class="modal-head">
        <h3 class="card-title">数据集管理</h3>
        <button class="btn btn-ghost btn-sm" type="button" data-close aria-label="关闭">×</button>
      </div>
      <div class="tab-row" role="tablist">
        <button class="tab selected" type="button" data-split="train" role="tab">训练集</button>
        <button class="tab" type="button" data-split="test" role="tab">测试集</button>
      </div>
      <div class="import-row">
        <button id="ds-import" class="btn btn-secondary" type="button">一键导入示例数据集</button>
        <div class="progress hidden" id="ds-progress"><div class="progress-fill" id="ds-progress-fill"></div></div>
        <span class="mono-sm" id="ds-progress-text"></span>
      </div>
      <p class="status" id="ds-status"></p>
      <div class="ds-grid-wrap"><div id="ds-grid"></div></div>
    </div>`;
  document.body.appendChild(overlay);

  const grid = overlay.querySelector('#ds-grid');
  const statusEl = overlay.querySelector('#ds-status');
  const importBtn = overlay.querySelector('#ds-import');
  const progressEl = overlay.querySelector('#ds-progress');
  const progressFill = overlay.querySelector('#ds-progress-fill');
  const progressText = overlay.querySelector('#ds-progress-text');
  const tabs = [...overlay.querySelectorAll('.tab')];
  let split = 'train';
  let importing = false;

  const setStatus = (t) => { statusEl.textContent = t; };

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      split = tab.dataset.split;
      tabs.forEach((t) => t.classList.toggle('selected', t === tab));
      refresh();
    });
  });

  overlay.querySelector('[data-close]').addEventListener('click', close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });

  /** 画一张缩略图（原始像素等比缩放到 26×26）。 */
  async function paintThumb(canvas, entry) {
    const ws = api.getWorkspace();
    if (!ws) return;
    try {
      const { data, width, height } = await ws.readSamplePixels(split, entry.id);
      const tmp = document.createElement('canvas');
      tmp.width = width;
      tmp.height = height;
      tmp.getContext('2d').putImageData(new ImageData(data, width, height), 0, 0);
      canvas.getContext('2d').drawImage(tmp, 0, 0, THUMB, THUMB);
    } catch (e) {
      // 损坏样本：缩略图留空，列表照常显示 id
    }
  }

  async function refresh() {
    const ws = api.getWorkspace();
    grid.innerHTML = '';
    if (!ws) {
      setStatus('请先选择工作区。');
      return;
    }
    const entries = ws.listSamples(split);
    setStatus(`${split === 'train' ? '训练集' : '测试集'}共 ${entries.length} 个样本（点击缩略图删除）。`);
    for (let d = 0; d <= 9; d++) {
      const label = String(d);
      const group = entries.filter((e) => e.label === label);
      const row = document.createElement('div');
      row.className = 'ds-class-row';
      row.innerHTML = `<div class="ds-class-label mono">${d}<span class="ds-class-count">${group.length} 张</span></div>`;
      const thumbs = document.createElement('div');
      thumbs.className = 'ds-thumbs';
      for (const entry of group) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'ds-thumb';
        btn.title = `id …${entry.id.slice(-6)} · 来源 ${entry.source}`;
        const cv = document.createElement('canvas');
        cv.width = THUMB;
        cv.height = THUMB;
        btn.appendChild(cv);
        btn.addEventListener('click', async () => {
          if (!window.confirm(`删除${split === 'train' ? '训练' : '测试'}集样本 …${entry.id.slice(-6)}（标签 ${label}）？`)) return;
          btn.disabled = true;
          try {
            await ws.removeSample(split, entry.id);
            await api.notifySamplesChanged();
            await refresh();
          } catch (e) {
            btn.disabled = false;
            window.alert(`删除失败：${e && e.message ? e.message : e}`);
          }
        });
        thumbs.appendChild(btn);
        paintThumb(cv, entry);
      }
      row.appendChild(thumbs);
      grid.appendChild(row);
    }
  }

  importBtn.addEventListener('click', async () => {
    const ws = api.getWorkspace();
    if (!ws) { setStatus('请先选择工作区。'); return; }
    if (importing) return;
    importing = true;
    importBtn.disabled = true;
    progressEl.classList.remove('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = '';
    setStatus('正在导入示例数据集…');
    try {
      await importSampleDataset(ws, {
        onProgress: (done, total) => {
          progressFill.style.width = `${((done / total) * 100).toFixed(1)}%`;
          progressText.textContent = `${done}/${total}`;
        },
      });
      setStatus('示例数据集导入完成。');
      await api.notifySamplesChanged();
      await refresh();
    } catch (e) {
      setStatus(`导入失败：${e && e.message ? e.message : e}`);
    } finally {
      importing = false;
      importBtn.disabled = false;
    }
  });

  function open() {
    overlay.classList.remove('hidden');
    return refresh();
  }
  function close() {
    overlay.classList.add('hidden');
  }

  return { open, close, refresh };
}
