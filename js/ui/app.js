/**
 * 应用壳：工作区引导流程 + 三栏布局装配。
 *
 * ES module，零 npm 依赖。后续 Task 10（舞台）、Task 11（面板）、Task 12（教学引导）
 * 在此骨架上填充：中栏 <canvas id="stage"> 与右栏面板均预留挂载点。
 */
import {
  Workspace,
  MemoryBackend,
  requestDirectory,
  ensurePermission,
  saveRecentHandle,
  loadRecentHandle,
  canvasToPngBlob,
} from '../workspace.js';
import { initDrawing, BOARD_SIZE } from './drawing.js';

/** 调试句柄（生产保留，无害）。 */
const debugHandle = { workspace: null, drawing: null, boot, label: null };

/** 工作区显示名：真实目录用 handle.name，内存后端用固定名。 */
function workspaceDisplayName(workspace) {
  const backend = workspace && workspace.backend;
  if (backend && backend.handle && typeof backend.handle.name === 'string') {
    return backend.handle.name;
  }
  return '内存工作区（测试）';
}

/**
 * 启动应用。
 * @param {{ workspaceBackend?: object }} [opts] 传入 backend（测试用）时跳过工作区引导，
 *   直接用 Workspace.create(backend) 初始化。
 */
export async function boot({ workspaceBackend } = {}) {
  const app = document.getElementById('app');
  if (!app) throw new Error('页面缺少 #app 容器。');
  app.innerHTML = '';

  // —— 顶栏 ——
  const topbar = document.createElement('header');
  topbar.className = 'topbar';
  topbar.innerHTML = `
    <h1 class="topbar-title">神经网络解剖教室</h1>
    <div class="topbar-right">
      <span class="ws-name" id="ws-name">未选择工作区</span>
      <button class="btn btn-ghost" id="ws-switch" type="button">切换工作区</button>
    </div>`;
  app.appendChild(topbar);

  // —— 三栏主体 ——
  const main = document.createElement('main');
  main.className = 'layout';
  main.innerHTML = `
    <section class="card panel-left" aria-label="数据采集">
      <h2 class="card-title">数据采集</h2>
      <canvas id="board" class="board" aria-label="手写画板"></canvas>
      <div class="label-row">
        <span class="label-hint">标签：</span>
        <div class="label-grid" id="label-grid" role="group" aria-label="数字标签"></div>
      </div>
      <div class="btn-row">
        <button class="btn btn-primary" id="btn-save" type="button">保存并训练</button>
        <button class="btn btn-secondary" id="btn-clear" type="button">清空</button>
      </div>
      <div class="preview-block">
        <canvas id="preview13" class="preview13" width="13" height="13" aria-label="13×13 输入预览"></canvas>
        <span class="preview-caption">13×13 输入预览</span>
      </div>
      <p class="status" id="status">在画板上写一个数字，选择标签后保存。</p>
    </section>
    <section class="card panel-center" aria-label="神经网络舞台">
      <h2 class="card-title">神经网络舞台</h2>
      <canvas id="stage" class="stage" width="640" height="420"></canvas>
    </section>
    <aside class="card panel-right" aria-label="训练">
      <h2 class="card-title">训练</h2>
      <dl class="kv">
        <div class="kv-row"><dt>模型</dt><dd class="mono" id="kv-model">—</dd></div>
        <div class="kv-row"><dt>优化器</dt><dd class="mono">Adam</dd></div>
        <div class="kv-row"><dt>轮次</dt><dd class="mono" id="kv-epoch">0</dd></div>
        <div class="kv-row"><dt>训练样本</dt><dd class="mono" id="kv-train-count">0</dd></div>
      </dl>
      <p class="placeholder-hint">训练面板（Task 11 实现）</p>
    </aside>`;
  app.appendChild(main);

  // —— 工作区引导遮罩 ——
  const guide = document.createElement('div');
  guide.className = 'overlay';
  guide.id = 'ws-guide';
  guide.innerHTML = `
    <div class="card guide-card">
      <h2 class="card-title">选择工作区</h2>
      <p class="guide-text">工作区是一个真实文件夹：手写样本 PNG、数据集 manifest、模型文件都会保存在里面，整个文件夹可以直接拷贝传播。</p>
      <button class="btn btn-primary btn-block" id="ws-pick" type="button">选择工作区文件夹</button>
      <button class="btn btn-secondary btn-block hidden" id="ws-resume" type="button"></button>
      <p class="guide-error hidden" id="ws-error"></p>
      <p class="guide-note">需要桌面版 Chrome / Edge（File System Access API）。</p>
    </div>`;
  app.appendChild(guide);

  const $ = (id) => document.getElementById(id);
  const wsNameEl = $('ws-name');
  const statusEl = $('status');
  const errorEl = $('ws-error');

  /** 状态行（id 等数值用等宽字体由 CSS 控制）。 */
  const setStatus = (text) => {
    statusEl.textContent = text;
  };

  const showGuideError = (text) => {
    errorEl.textContent = text;
    errorEl.classList.remove('hidden');
  };
  const clearGuideError = () => {
    errorEl.textContent = '';
    errorEl.classList.add('hidden');
  };

  // —— 左栏：画板 + 标签 + 预览 ——
  const previewCanvas = $('preview13');
  const previewCtx = previewCanvas.getContext('2d');
  let currentLabel = '0';

  const refreshPreview = () => {
    if (!debugHandle.drawing) return;
    const px = debugHandle.drawing.getPixels13();
    const img = previewCtx.createImageData(13, 13);
    for (let i = 0; i < 169; i++) {
      const g = Math.round(255 * (1 - px[i])); // 0=纸面白，1=墨迹黑
      img.data[i * 4] = g;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = g;
      img.data[i * 4 + 3] = 255;
    }
    previewCtx.putImageData(img, 0, 0);
  };

  const drawing = initDrawing($('board'), refreshPreview);
  debugHandle.drawing = drawing;
  refreshPreview();

  const labelGrid = $('label-grid');
  const labelButtons = [];
  for (let d = 0; d <= 9; d++) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'label-btn mono';
    btn.dataset.label = String(d);
    btn.textContent = String(d);
    btn.setAttribute('aria-pressed', d === 0 ? 'true' : 'false');
    if (d === 0) btn.classList.add('selected');
    btn.addEventListener('click', () => {
      currentLabel = String(d);
      debugHandle.label = currentLabel;
      labelButtons.forEach((b) => {
        const on = b === btn;
        b.classList.toggle('selected', on);
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
    });
    labelGrid.appendChild(btn);
    labelButtons.push(btn);
  }
  debugHandle.label = currentLabel;

  $('btn-clear').addEventListener('click', () => {
    drawing.clear();
    refreshPreview();
    setStatus('画板已清空。');
  });

  /** 「保存并训练」：本任务只落盘样本，真正的单步训练是 Task 11 的事。 */
  const saveSample = async () => {
    const ws = debugHandle.workspace;
    if (!ws) {
      setStatus('请先选择工作区。');
      return;
    }
    const saveBtn = $('btn-save');
    saveBtn.disabled = true;
    try {
      const blob = await canvasToPngBlob(drawing.canvas);
      const id = await ws.addSample('train', currentLabel, blob, {
        width: BOARD_SIZE,
        height: BOARD_SIZE,
        source: 'handwritten',
      });
      const n = ws.listSamples('train').length;
      $('kv-train-count').textContent = String(n);
      setStatus(`已保存样本 #…${id.slice(-6)}（标签 ${currentLabel}，训练集共 ${n} 个）`);
      drawing.clear();
      refreshPreview();
    } catch (e) {
      setStatus(`保存失败：${e && e.message ? e.message : e}`);
    } finally {
      saveBtn.disabled = false;
    }
  };
  $('btn-save').addEventListener('click', saveSample);

  // —— 中栏：舞台占位（Task 10 实现） ——
  const drawStagePlaceholder = () => {
    const stage = $('stage');
    const sctx = stage.getContext('2d');
    sctx.fillStyle = '#f5f5f4';
    sctx.fillRect(0, 0, stage.width, stage.height);
    sctx.fillStyle = '#a8a29e';
    sctx.font = '28px "Noto Sans SC", sans-serif';
    sctx.textAlign = 'center';
    sctx.textBaseline = 'middle';
    sctx.fillText('舞台加载中（Task 10 实现）', stage.width / 2, stage.height / 2);
  };
  drawStagePlaceholder();

  // —— 工作区装配 ——
  const setWorkspace = async (ws) => {
    debugHandle.workspace = ws;
    wsNameEl.textContent = workspaceDisplayName(ws);
    const n = ws.listSamples('train').length;
    $('kv-train-count').textContent = String(n);
    guide.classList.add('hidden');
    setStatus(`工作区就绪：${workspaceDisplayName(ws)}，训练集 ${n} 个样本。在画板上写字吧。`);
  };

  /** 目录 handle → 打开或初始化工作区。 */
  const openWorkspaceFromHandle = async (handle) => {
    const ok = await ensurePermission(handle);
    if (!ok) throw new Error('未获得文件夹读写权限，请重试并点击"允许"。');
    let ws;
    try {
      ws = await Workspace.open(handle);
    } catch (e) {
      // 不是已有工作区 → 初始化为新工作区
      ws = await Workspace.create(handle);
    }
    try {
      await saveRecentHandle(handle);
    } catch (e) {
      // 记住最近工作区失败不影响使用（无痕模式等）
      console.warn('保存最近工作区失败：', e);
    }
    return ws;
  };

  $('ws-pick').addEventListener('click', async () => {
    clearGuideError();
    const pickBtn = $('ws-pick');
    pickBtn.disabled = true;
    try {
      const handle = await requestDirectory(); // 需用户手势
      const ws = await openWorkspaceFromHandle(handle);
      await setWorkspace(ws);
    } catch (e) {
      if (e && e.name === 'AbortError') {
        showGuideError('已取消选择。');
      } else {
        showGuideError(`打开工作区失败：${e && e.message ? e.message : e}`);
      }
    } finally {
      pickBtn.disabled = false;
    }
  });

  /** 恢复最近工作区（IndexedDB 存的 handle）。失败则静默隐藏恢复按钮。 */
  const tryOfferResume = async () => {
    const resumeBtn = $('ws-resume');
    resumeBtn.disabled = false; // 每次重新提供恢复入口时先解禁（避免上次成功后永久灰色）
    try {
      const handle = await loadRecentHandle();
      if (!handle || typeof handle.name !== 'string') return;
      resumeBtn.textContent = `继续使用上次的「${handle.name}」`;
      resumeBtn.classList.remove('hidden');
      // 用 onclick 赋值而非 addEventListener：tryOfferResume 可能被多次调用，
      // 赋值语义保证同一时刻只有一个 handler，不会重复挂载。
      resumeBtn.onclick = async () => {
        clearGuideError();
        resumeBtn.disabled = true;
        try {
          const ok = await ensurePermission(handle);
          if (!ok) throw new Error('需要重新授权文件夹访问权限。');
          const ws = await Workspace.open(handle);
          await setWorkspace(ws);
        } catch (e) {
          showGuideError(`恢复失败：${e && e.message ? e.message : e}，请重新选择文件夹。`);
          resumeBtn.disabled = false;
        }
      };
    } catch (e) {
      // IndexedDB 不可用（如无头测试环境）：不提供恢复入口
    }
  };

  $('ws-switch').addEventListener('click', () => {
    clearGuideError();
    guide.classList.remove('hidden');
    tryOfferResume();
  });

  // —— 启动分支 ——
  window.__nnanatomy = debugHandle;
  if (workspaceBackend) {
    // 测试入口：跳过引导
    const ws = await Workspace.create(workspaceBackend);
    await setWorkspace(ws);
  } else {
    tryOfferResume();
  }
}
