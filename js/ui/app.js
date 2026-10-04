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
import { initStage } from './stage.js';
import { createNetwork } from '../network.js';
import { createOptState } from '../trainer.js';
import { ADAM_DEFAULT_LR, paramCount } from '../optimizer.js';
import { initPanel, uniqueModelName } from './panel.js';
import { initDatasetViewer } from './dataset-viewer.js';

/** 调试句柄（生产保留，无害）。 */
const debugHandle = { workspace: null, drawing: null, boot, label: null, panel: null, viewer: null };

/** 当前模型状态（Task 11）：训练面板与左栏单步训练共享的唯一可变状态。 */
const ARCH = { inputs: 169, hidden: 16, outputs: 10 };
const model = {
  name: null,
  net: null,
  optState: null,
  optimizer: 'adam',
  lr: ADAM_DEFAULT_LR,
  epochsTrained: 0,
  samplesSeen: 0,
  runSeed: 0,
  history: [],
};
/** 当前标签（模块级：api.getLabel 需要在 boot 外访问）。 */
let currentLabel = '0';
let panelApi = null;
let viewerApi = null;

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
    <aside class="card panel-right" id="panel-right" aria-label="训练"></aside>`;
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

  /** 「保存并训练」：保存样本 → 单样本训练一步（舞台动画）→ 清空画板。 */
  const saveSample = async () => {
    const ws = debugHandle.workspace;
    if (!ws) {
      setStatus('请先选择工作区。');
      return;
    }
    const saveBtn = $('btn-save');
    saveBtn.disabled = true;
    try {
      const x = drawing.getPixels13(); // 清空前捕获当前画布（墨=1 的 13×13）
      const y = Number(currentLabel);
      const blob = await canvasToPngBlob(drawing.canvas);
      const id = await ws.addSample('train', currentLabel, blob, {
        width: BOARD_SIZE,
        height: BOARD_SIZE,
        source: 'handwritten',
      });
      drawing.clear();
      refreshPreview();
      updateKv();
      setStatus(`已保存样本 #…${id.slice(-6)}（标签 ${currentLabel}）`);
      // 单步训练：舞台播放这次权重更新的 before/after 对比。
      // 批量训练进行中则跳过单步（避免两次训练交错改权重），样本照常保存。
      if (panelApi.isTraining()) {
        setStatus(`已保存样本 #…${id.slice(-6)}（标签 ${currentLabel}；批量训练进行中，跳过单步训练）。`);
      } else {
        try {
          await panelApi.trainSingleStep(x, y);
          setStatus(`已保存样本 #…${id.slice(-6)}（标签 ${currentLabel}），并完成一次单步训练。`);
        } catch (e) {
          setStatus(`样本已保存，但单步训练失败：${e && e.message ? e.message : e}`);
        }
      }
    } catch (e) {
      setStatus(`保存失败：${e && e.message ? e.message : e}`);
    } finally {
      saveBtn.disabled = false;
    }
  };
  $('btn-save').addEventListener('click', saveSample);

  // —— 中栏：神经网络舞台（Task 10） ——
  // 网络与舞台实例挂在 debugHandle 上，供训练面板（Task 11）与教学引导（Task 12）复用。
  // 注意：这里是启动时的临时网络；工作区就绪后 initDefaultModel 会加载/创建真实模型并 setNet。
  const net = createNetwork({ inputs: 169, hidden: 16, outputs: 10 }, 7);
  const stage = initStage($('stage'), net);
  debugHandle.net = net;
  debugHandle.stage = stage;

  // —— 模型状态管理（Task 11） ——

  /** kv 行刷新（左栏计数 + 右栏模型信息；面板创建前调用则静默跳过缺失节点）。 */
  function updateKv() {
    const set = (id, text) => {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    };
    const ws = debugHandle.workspace;
    set('kv-model', model.name || '—');
    set('kv-optimizer', model.optimizer === 'adam' ? 'Adam' : 'SGD');
    set('kv-lr', String(Number(model.lr.toPrecision(3))));
    set('kv-epoch', String(model.epochsTrained));
    set('kv-train-count', ws ? String(ws.listSamples('train').length) : '0');
    set('kv-test-count', ws ? String(ws.listSamples('test').length) : '0');
  }

  /** 训练后自动保存：权重 + 优化器状态 + 训练元信息。 */
  async function saveModelNow() {
    const ws = debugHandle.workspace;
    if (!ws || !model.net || !model.name) return;
    await ws.saveModel(model.name, model.net, model.optState, {
      epochsTrained: model.epochsTrained,
      samplesSeen: model.samplesSeen,
      history: model.history,
      optimizer: model.optimizer,
      lr: model.lr,
      runSeed: model.runSeed,
    });
  }

  async function listModelNames() {
    const ws = debugHandle.workspace;
    if (!ws) return [];
    return (await ws.listModels()).map((e) => e.name);
  }

  async function newModel(name) {
    model.name = uniqueModelName(name, await listModelNames());
    model.net = createNetwork(ARCH, Date.now() % 2 ** 32);
    model.optimizer = 'adam';
    model.lr = ADAM_DEFAULT_LR;
    model.optState = createOptState(model.net, 'adam');
    model.epochsTrained = 0;
    model.samplesSeen = 0;
    model.runSeed = 0;
    model.history = [];
    if (debugHandle.stage) debugHandle.stage.setNet(model.net);
    debugHandle.net = model.net;
    await saveModelNow();
    updateKv();
  }

  async function loadModel(name) {
    const ws = debugHandle.workspace;
    if (!ws) throw new Error('请先选择工作区。');
    const doc = await ws.loadModel(name);
    if (!doc.net || !doc.net.arch) throw new Error(`模型文件损坏：${name}`);
    model.name = doc.name;
    model.net = doc.net;
    const m = doc.metrics || {};
    model.epochsTrained = Number.isFinite(m.epochsTrained) ? m.epochsTrained : 0;
    model.samplesSeen = Number.isFinite(m.samplesSeen) ? m.samplesSeen : 0;
    model.history = Array.isArray(m.history) ? m.history : [];
    model.optimizer = m.optimizer === 'sgd' ? 'sgd' : 'adam';
    model.lr = Number.isFinite(m.lr) && m.lr > 0 ? m.lr : ADAM_DEFAULT_LR;
    model.runSeed = Number.isFinite(m.runSeed) ? m.runSeed : 0;
    // 恢复 Adam 动量：形状对得上才复用，否则重建（架构变更时不炸）
    const st = doc.optimizerState;
    if (model.optimizer === 'adam' && st && st.m && st.m.length === paramCount(model.net)) {
      model.optState = st;
    } else {
      model.optState = createOptState(model.net, model.optimizer);
    }
    debugHandle.stage.setNet(model.net);
    debugHandle.net = model.net;
    updateKv();
  }

  async function deleteModel(name) {
    const ws = debugHandle.workspace;
    if (!ws) throw new Error('请先选择工作区。');
    await ws.deleteModel(name);
    if (model.name === name) {
      await newModel('未命名模型'); // 当前模型被删 → 自动新建
    }
  }

  /** 切换优化器：重建优化器状态（旧动量不清零会污染新优化器）。 */
  async function setOptimizer(name) {
    if (name !== 'adam' && name !== 'sgd') throw new Error(`未知优化器：${name}`);
    if (!model.net) return;
    model.optimizer = name;
    model.optState = createOptState(model.net, name);
    await saveModelNow();
  }

  function setLr(lr) {
    if (Number.isFinite(lr) && lr > 0) model.lr = lr;
  }

  /** 样本变更（导入/删除）后：刷新计数 + 面板重评。 */
  async function notifySamplesChanged() {
    updateKv();
    if (panelApi) await panelApi.onSamplesChanged();
  }

  /** 工作区就绪后：加载首个模型，不存在则创建「未命名模型」。 */
  async function initDefaultModel() {
    const names = await listModelNames();
    if (names.length > 0) await loadModel(names[0]);
    else await newModel('未命名模型');
    if (panelApi) await panelApi.refresh();
  }

  /** 训练面板与数据集管理器的宿主 API。 */
  const api = {
    getWorkspace: () => debugHandle.workspace,
    getStage: () => debugHandle.stage,
    getDrawing: () => debugHandle.drawing,
    getLabel: () => currentLabel,
    getModel: () => model,
    newModel,
    loadModel,
    deleteModel,
    listModelNames,
    setOptimizer,
    setLr,
    nextRunSeed: () => { model.runSeed += 1; return model.runSeed; },
    saveModelNow,
    notifySamplesChanged,
    updateKv,
    openDatasetViewer: () => { if (viewerApi) viewerApi.open(); },
  };

  // —— 右栏：训练面板 + 数据集管理（Task 11） ——
  panelApi = initPanel($('panel-right'), api);
  debugHandle.panel = panelApi;
  viewerApi = initDatasetViewer(api);
  debugHandle.viewer = viewerApi;

  // —— 工作区装配 ——
  const setWorkspace = async (ws) => {
    debugHandle.workspace = ws;
    wsNameEl.textContent = workspaceDisplayName(ws);
    guide.classList.add('hidden');
    await initDefaultModel(); // 加载/创建模型（含 updateKv）
    const n = ws.listSamples('train').length;
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
