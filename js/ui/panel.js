/**
 * 训练控制面板：右栏。模型管理 + 训练控制 + 评估指标（混淆矩阵）+ 数据集入口。
 *
 * ES module，零 npm 依赖。训练计算走 js/trainer.js（trainEpochs/evaluate），
 * 样本加载走 js/dataset.js（loadSamples），动画走 stage.playTrainingStep。
 *
 * # 分块训练与 shuffle 一致性（load-bearing）
 * 首 epoch 拆成"第 1 个样本做舞台动画 + 其余静默"时，必须复刻 trainer.js
 * 的 shuffle（seed*100003 + globalEpoch 的 Fisher–Yates），否则动画步与
 * trainEpochs 一次调用的样本顺序不一致。此处复制一份 shuffledIndices —
 * trainer.js 未导出它（按约束不得修改 trainer.js），实现逐行对应，
 * 若 trainer.js 的 shuffle 变更，此处必须同步。
 */
import {
  trainEpochs,
  evaluate,
  inferWithTrace,
} from '../trainer.js';
import { forward, backward, isFiniteNet } from '../network.js';
import { adamStep, sgdStep } from '../optimizer.js';
import { createRng } from '../rng.js';
import { loadSamples } from '../dataset.js';

/** 学习率滑条范围（对数刻度）：0.0001 ~ 0.01。 */
const LR_MIN = 0.0001;
const LR_MAX = 0.01;

/** Fisher–Yates shuffle：与 js/trainer.js 的 shuffledIndices 逐行对应。 */
function shuffledIndices(n, rng) {
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng.nextF64() * (i + 1));
    const t = idx[i];
    idx[i] = idx[j];
    idx[j] = t;
  }
  return idx;
}

/** argmax：与 js/trainer.js 内部实现一致（平局取最小下标）。 */
function argmaxP(p) {
  let best = 0;
  for (let o = 1; o < p.length; o++) {
    if (p[o] > p[best]) best = o;
  }
  return best;
}

/** 样本集上的平均对数似然（训练 history 用）。 */
function meanLogLikelihood(net, samples) {
  let s = 0;
  for (const sm of samples) s += Math.log(forward(net, sm.x).p[sm.y]);
  return s / samples.length;
}

/** 单次优化器步（梯度上升：backward 返回对数似然梯度）。 */
function stepOnce(model, grads) {
  if (model.optimizer === 'adam') adamStep(model.net, model.optState, grads, model.lr);
  else sgdStep(model.net, grads, model.lr);
}

/** 学习率 ↔ 滑条位置（对数刻度）。 */
function lrToSlider(lr) {
  const t = Math.log(lr / LR_MIN) / Math.log(LR_MAX / LR_MIN);
  return Math.round(Math.max(0, Math.min(1, t)) * 1000);
}
function sliderToLr(v) {
  const t = v / 1000;
  return LR_MIN * Math.pow(LR_MAX / LR_MIN, t);
}
function fmtLr(lr) {
  return String(Number(lr.toPrecision(3)));
}

/**
 * 初始化训练面板。
 * @param {HTMLElement} root 右栏容器（aside.panel-right），内容由本函数构建
 * @param {object} api 应用壳提供：
 *   getWorkspace(), getStage(), getDrawing(), getLabel(),
 *   getModel(), newModel(name), loadModel(name), deleteModel(name), listModelNames(),
 *   setOptimizer(name), setLr(lr), nextRunSeed(),
 *   saveModelNow(), notifySamplesChanged(), updateKv()
 * @returns {{ batchTrain(epochs?:number):Promise<void>, trainSingleStep(x,y):Promise<void>,
 *            refresh():Promise<void>, onSamplesChanged():Promise<void>, getEval():object|null,
 *            newModel(name:string, seed?:number):Promise<void>, saveModelNow():Promise<void>,
 *            loadModel(name:string):Promise<void> }}
 */
export function initPanel(root, api) {
  if (!root) throw new Error('initPanel 需要右栏容器。');

  root.innerHTML = `
    <h2 class="card-title">训练</h2>
    <section class="sub-card" aria-label="模型管理">
      <h3 class="sub-title">模型</h3>
      <div class="model-row">
        <select id="model-select" class="select" aria-label="当前模型"></select>
        <button id="model-new" class="btn btn-secondary btn-sm" type="button">新建</button>
        <button id="model-del" class="btn btn-secondary btn-sm" type="button">删除</button>
      </div>
      <div class="btn-row">
        <button id="model-weights" class="btn btn-secondary btn-sm" type="button">查看权重</button>
      </div>
      <dl class="kv">
        <div class="kv-row"><dt>模型</dt><dd class="mono" id="kv-model">—</dd></div>
        <div class="kv-row"><dt>优化器</dt><dd class="mono" id="kv-optimizer">Adam</dd></div>
        <div class="kv-row"><dt>学习率</dt><dd class="mono" id="kv-lr">0.001</dd></div>
        <div class="kv-row"><dt>轮次</dt><dd class="mono" id="kv-epoch">0</dd></div>
        <div class="kv-row"><dt>训练样本</dt><dd class="mono" id="kv-train-count">0</dd></div>
        <div class="kv-row"><dt>测试样本</dt><dd class="mono" id="kv-test-count">0</dd></div>
      </dl>
    </section>
    <section class="sub-card" aria-label="训练控制">
      <h3 class="sub-title">训练控制</h3>
      <div class="ctl-row">
        <label class="ctl-label" for="opt-select">优化器</label>
        <select id="opt-select" class="select">
          <option value="adam">Adam</option>
          <option value="sgd">SGD</option>
        </select>
      </div>
      <div class="ctl-row">
        <label class="ctl-label" for="lr-range">学习率 <span class="mono" id="lr-val">0.001</span></label>
        <input id="lr-range" type="range" min="0" max="1000" value="500" />
      </div>
      <div class="ctl-row">
        <label class="ctl-label" for="epochs-input">轮数</label>
        <input id="epochs-input" class="input-num mono" type="number" min="1" max="200" value="15" />
      </div>
      <div class="btn-row">
        <button id="btn-train" class="btn btn-primary" type="button">开始训练</button>
        <button id="btn-pause" class="btn btn-secondary" type="button" disabled>暂停</button>
        <button id="btn-resume" class="btn btn-secondary" type="button" disabled>继续</button>
      </div>
      <p class="status" id="train-status">就绪。</p>
      <button id="btn-reset-weights" class="btn btn-secondary btn-sm hidden" type="button">权重异常，重置模型</button>
      <canvas id="loss-curve" class="loss-curve" width="300" height="120" aria-label="对数似然曲线"></canvas>
      <p class="caption">对数似然 ↑ 越高越好</p>
    </section>
    <section class="sub-card" aria-label="评估指标">
      <h3 class="sub-title">评估指标</h3>
      <div class="metric-big">
        <span class="mono" id="metric-acc">—</span>
        <span class="metric-label">测试集准确率</span>
      </div>
      <div id="per-class" class="per-class" aria-label="每类准确率"></div>
      <h4 class="sub-title-sm">混淆矩阵（点击格子查看样本）</h4>
      <table id="confusion" class="confusion" aria-label="混淆矩阵"></table>
      <p class="status" id="eval-status">尚未评估。</p>
      <div class="btn-row">
        <button id="btn-eval" class="btn btn-secondary" type="button">重新评估</button>
      </div>
    </section>
    <section class="sub-card" aria-label="数据集">
      <h3 class="sub-title">数据集</h3>
      <p class="mono-sm" id="dataset-summary">训练 0 · 测试 0</p>
      <div class="btn-row">
        <button id="btn-dataset" class="btn btn-secondary" type="button">数据集管理</button>
      </div>
    </section>`;

  const $ = (id) => root.querySelector('#' + id);
  const modelSelect = $('model-select');
  const optSelect = $('opt-select');
  const lrRange = $('lr-range');
  const lrVal = $('lr-val');
  const epochsInput = $('epochs-input');
  const btnTrain = $('btn-train');
  const btnPause = $('btn-pause');
  const btnResume = $('btn-resume');
  const trainStatus = $('train-status');
  const resetWeightsBtn = $('btn-reset-weights');
  const lossCanvas = $('loss-curve');
  const metricAcc = $('metric-acc');
  const perClassEl = $('per-class');
  const confusionEl = $('confusion');
  const evalStatus = $('eval-status');
  const datasetSummary = $('dataset-summary');

  /** 权重有限性检查（Task 12：NaN 恢复）。false 时亮出重置按钮。 */
  function checkWeightsFinite(m) {
    if (isFiniteNet(m.net)) {
      resetWeightsBtn.classList.add('hidden');
      return true;
    }
    setTrainStatus('权重出现 NaN/无穷大（可能学习率过大或数值发散），训练已停止。点「权重异常，重置模型」恢复。');
    resetWeightsBtn.classList.remove('hidden');
    return false;
  }

  resetWeightsBtn.addEventListener('click', async () => {
    resetWeightsBtn.disabled = true;
    try {
      await api.resetModelWeights();
      resetWeightsBtn.classList.add('hidden');
      setTrainStatus('模型已重置：权重重新初始化，训练计数清零。');
    } catch (e) {
      setTrainStatus(`重置失败：${e && e.message ? e.message : e}`);
    } finally {
      resetWeightsBtn.disabled = false;
    }
  });
  /** 当前评估结果：{ samples, preds, acc, perClass, confusion }，样本变更后重评。 */
  let lastEval = null;
  /** 训练运行态：{ paused, total, done, seed }；null 表示空闲。 */
  let trainRun = null;
  /** 暂停后待继续：{ total, done, seed }；null 表示无。 */
  let pausedState = null;
  /** 单步训练重入锁。 */
  let singleStepBusy = false;
  /** 单步计数（节流保存用）。 */
  let singleStepCount = 0;
  /** 混淆矩阵格子弹层（单例）。 */
  let cellPopup = null;

  const setTrainStatus = (t) => { trainStatus.textContent = t; };
  const setButtons = (train, pause, resume) => {
    btnTrain.disabled = !train;
    btnPause.disabled = !pause;
    btnResume.disabled = !resume;
  };
  /** 训练进行时锁定模型/优化器/学习率/轮数控件（防并发改状态），暂停/空闲时放开。 */
  const setControlsEnabled = (on) => {
    modelSelect.disabled = !on;
    $('model-new').disabled = !on;
    $('model-del').disabled = !on;
    optSelect.disabled = !on;
    lrRange.disabled = !on;
    epochsInput.disabled = !on;
  };
  const yieldUi = () => new Promise((r) => setTimeout(r, 0));

  // —— 模型下拉框 ——

  async function refreshModelSelect() {
    const names = await api.listModelNames();
    const cur = api.getModel() && api.getModel().name;
    modelSelect.innerHTML = '';
    for (const n of names) {
      const opt = document.createElement('option');
      opt.value = n;
      opt.textContent = n;
      modelSelect.appendChild(opt);
    }
    if (cur && names.includes(cur)) modelSelect.value = cur;
  }

  modelSelect.addEventListener('change', async () => {
    try {
      await api.loadModel(modelSelect.value);
      await refresh();
      pausedState = null; // 暂停态属于旧模型，切模型后丢弃
      setTrainStatus(`已切换到模型「${modelSelect.value}」。`);
    } catch (e) {
      setTrainStatus(`切换模型失败：${e && e.message ? e.message : e}`);
      await refreshModelSelect();
    }
  });

  $('model-new').addEventListener('click', async () => {
    const name = window.prompt('新模型名称：', '我的模型');
    if (name === null) return;
    const trimmed = name.trim();
    if (!trimmed) { setTrainStatus('模型名称不能为空。'); return; }
    try {
      await api.newModel(trimmed);
      await refresh();
      setTrainStatus(`已创建模型「${trimmed}」。`);
    } catch (e) {
      setTrainStatus(`创建模型失败：${e && e.message ? e.message : e}`);
    }
  });

  $('model-weights').addEventListener('click', () => {
    if (api.openWeightsView) api.openWeightsView();
  });

  $('model-del').addEventListener('click', async () => {
    const m = api.getModel();
    if (!m || !m.name) return;
    if (!window.confirm(`删除模型「${m.name}」？此操作不可撤销。`)) return;
    try {
      await api.deleteModel(m.name);
      await refresh();
      pausedState = null; // 当前模型已删，暂停态一并丢弃
      setTrainStatus(`已删除模型「${m.name}」。`);
    } catch (e) {
      setTrainStatus(`删除模型失败：${e && e.message ? e.message : e}`);
    }
  });

  // —— 优化器 / 学习率 ——

  optSelect.addEventListener('change', async () => {
    try {
      await api.setOptimizer(optSelect.value);
      api.updateKv();
      setTrainStatus(`优化器已切换为 ${optSelect.value === 'adam' ? 'Adam' : 'SGD'}（状态已重置并保存）。`);
    } catch (e) {
      setTrainStatus(`切换优化器失败：${e && e.message ? e.message : e}`);
      await refresh();
    }
  });

  lrRange.addEventListener('input', () => {
    const lr = sliderToLr(Number(lrRange.value));
    lrVal.textContent = fmtLr(lr);
    api.setLr(lr);
    api.updateKv();
  });

  /** 把面板控件同步为当前模型状态（不碰 epochs 输入框的用户输入）。 */
  function syncControlsFromModel() {
    const m = api.getModel();
    if (!m) return;
    optSelect.value = m.optimizer;
    lrRange.value = String(lrToSlider(m.lr));
    lrVal.textContent = fmtLr(m.lr);
  }

  // —— loss 曲线 ——

  function drawLossCurve() {
    const m = api.getModel();
    const ctx = lossCanvas.getContext('2d');
    const W = lossCanvas.width;
    const H = lossCanvas.height;
    ctx.clearRect(0, 0, W, H);
    const hist = (m && m.history) || [];
    if (hist.length === 0) {
      ctx.fillStyle = '#a8a29e';
      ctx.font = '12px sans-serif';
      ctx.fillText('暂无训练记录', 12, H / 2);
      return;
    }
    const lls = hist.map((h) => h.logLikelihood);
    let lo = Math.min(...lls);
    let hi = Math.max(...lls);
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    const pad = 8;
    const X = (i) => pad + (i / Math.max(1, hist.length - 1)) * (W - pad * 2);
    const Y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - pad * 2);
    // 网格线
    ctx.strokeStyle = '#f0e6d8';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 2; g++) {
      const y = pad + (g / 2) * (H - pad * 2);
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(W - pad, y);
      ctx.stroke();
    }
    // 折线
    ctx.strokeStyle = '#ea580c';
    ctx.lineWidth = 2;
    ctx.beginPath();
    hist.forEach((h, i) => {
      if (i === 0) ctx.moveTo(X(i), Y(h.logLikelihood));
      else ctx.lineTo(X(i), Y(h.logLikelihood));
    });
    ctx.stroke();
    // 端点
    const last = hist[hist.length - 1];
    ctx.fillStyle = '#ea580c';
    ctx.beginPath();
    ctx.arc(X(hist.length - 1), Y(last.logLikelihood), 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#78716c';
    ctx.font = '10px sans-serif';
    ctx.fillText(last.logLikelihood.toFixed(3), pad + 2, pad + 10);
  }

  function pushHistory(m, samples) {
    m.history.push({ epoch: m.epochsTrained, logLikelihood: meanLogLikelihood(m.net, samples) });
    drawLossCurve();
  }

  // —— 评估与指标渲染 ——

  async function runEvaluation() {
    const ws = api.getWorkspace();
    const m = api.getModel();
    if (!ws || !m || !m.net) return;
    let testSamples = [];
    try {
      testSamples = await loadSamples(ws, 'test');
    } catch (e) {
      evalStatus.textContent = `评估失败：${e && e.message ? e.message : e}`;
      return;
    }
    const r = evaluate(m.net, testSamples);
    const preds = testSamples.map((s) => argmaxP(forward(m.net, s.x).p));
    lastEval = { samples: testSamples, preds, ...r };
    renderMetrics();
    if (testSamples.length > 0 && api.notifyLesson) api.notifyLesson('evaluated');
    evalStatus.textContent = testSamples.length === 0
      ? '测试集为空：导入示例数据集后可评估。'
      : `测试集 ${testSamples.length} 个样本。`;
  }

  function renderMetrics() {
    if (!lastEval) {
      metricAcc.textContent = '—';
      perClassEl.innerHTML = '';
      confusionEl.innerHTML = '';
      return;
    }
    const { acc, perClass, confusion } = lastEval;
    metricAcc.textContent = acc === null ? '—' : `${(acc * 100).toFixed(1)}%`;
    // 每类准确率条
    perClassEl.innerHTML = '';
    for (let c = 0; c < 10; c++) {
      const row = document.createElement('div');
      row.className = 'pc-row';
      const v = perClass[c];
      row.innerHTML = `<span class="pc-label mono">${c}</span>
        <span class="pc-bar"><span class="pc-fill" style="width:${v === null ? 0 : (v * 100).toFixed(1)}%"></span></span>
        <span class="pc-val mono">${v === null ? '—' : (v * 100).toFixed(0) + '%'}</span>`;
      perClassEl.appendChild(row);
    }
    // 混淆矩阵 10×10
    let maxCount = 0;
    for (let t = 0; t < 10; t++) {
      for (let p = 0; p < 10; p++) maxCount = Math.max(maxCount, confusion[t][p]);
    }
    confusionEl.innerHTML = '';
    const head = document.createElement('tr');
    head.innerHTML = '<th></th>' + Array.from({ length: 10 }, (_, p) => `<th class="mono">${p}</th>`).join('');
    confusionEl.appendChild(head);
    for (let t = 0; t < 10; t++) {
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.className = 'mono';
      th.textContent = String(t);
      tr.appendChild(th);
      for (let p = 0; p < 10; p++) {
        const td = document.createElement('td');
        const n = confusion[t][p];
        td.className = 'mono';
        td.textContent = String(n);
        if (n > 0 && maxCount > 0) {
          const a = 0.12 + 0.78 * (n / maxCount);
          td.style.backgroundColor = `rgba(234, 88, 12, ${a.toFixed(2)})`;
          if (n / maxCount > 0.55) td.style.color = '#fff';
        }
        td.title = `真值 ${t} → 预测 ${p}：${n} 个`;
        td.dataset.t = String(t);
        td.dataset.p = String(p);
        td.addEventListener('click', (ev) => openCellPopup(t, p, ev));
        tr.appendChild(td);
      }
      confusionEl.appendChild(tr);
    }
  }

  function closeCellPopup() {
    if (cellPopup) {
      cellPopup.remove();
      cellPopup = null;
    }
  }

  /** 混淆矩阵格子弹层：该格样本缩略图 + 逐张「删除错标」。 */
  function openCellPopup(t, p, ev) {
    closeCellPopup();
    if (!lastEval) return;
    const hits = [];
    for (let i = 0; i < lastEval.samples.length; i++) {
      const s = lastEval.samples[i];
      if (s.y === t && lastEval.preds[i] === p) hits.push(s);
    }
    const popup = document.createElement('div');
    popup.className = 'cell-popup';
    popup.innerHTML = `
      <div class="cell-popup-head">
        <span>真值 ${t} → 预测 ${p}（${hits.length} 个）</span>
        <button class="btn btn-ghost btn-sm" type="button" data-close>×</button>
      </div>
      <div class="cell-popup-grid"></div>`;
    const grid = popup.querySelector('.cell-popup-grid');
    if (hits.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'status';
      empty.textContent = '该格暂无样本。';
      grid.appendChild(empty);
    }
    const ws = api.getWorkspace();
    for (const s of hits) {
      const item = document.createElement('div');
      item.className = 'cell-thumb';
      const cv = document.createElement('canvas');
      cv.width = 26;
      cv.height = 26;
      cv.className = 'thumb26';
      item.appendChild(cv);
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'btn btn-secondary btn-xs';
      del.textContent = '删除错标';
      del.addEventListener('click', async () => {
        if (!window.confirm(`删除测试集样本 ${s.id.slice(-6)}（真值 ${t}）？`)) return;
        del.disabled = true;
        try {
          await ws.removeSample('test', s.id);
          await api.notifySamplesChanged(); // 含重评
          closeCellPopup();
        } catch (e) {
          del.disabled = false;
          window.alert(`删除失败：${e && e.message ? e.message : e}`);
        }
      });
      item.appendChild(del);
      grid.appendChild(item);
      // 异步画缩略图（失败则留空，不阻塞弹层）
      ws.readSamplePixels('test', s.id).then(({ data, width, height }) => {
        const tmp = document.createElement('canvas');
        tmp.width = width;
        tmp.height = height;
        tmp.getContext('2d').putImageData(new ImageData(data, width, height), 0, 0);
        cv.getContext('2d').drawImage(tmp, 0, 0, 26, 26);
      }).catch(() => {});
    }
    popup.querySelector('[data-close]').addEventListener('click', closeCellPopup);
    document.body.appendChild(popup);
    // 定位：尽量靠近点击处，不超出视口
    const x = ev && ev.clientX ? ev.clientX : window.innerWidth / 2;
    const y = ev && ev.clientY ? ev.clientY : window.innerHeight / 2;
    popup.style.left = Math.min(x, window.innerWidth - 300) + 'px';
    popup.style.top = Math.min(y, window.innerHeight - 320) + 'px';
    cellPopup = popup;
  }

  // —— 批量训练 ——

  /** 首 epoch：第 1 个样本做舞台动画（before/after 快照），其余静默。 */
  async function animatedFirstEpoch(m, samples, seed) {
    const E0 = m.epochsTrained;
    const n = samples.length;
    // 复刻 trainer.js 的 epoch E0+1 shuffle，保证与一次 trainEpochs 调用顺序一致
    const order = shuffledIndices(n, createRng(seed * 100003 + (E0 + 1)));
    const stage = api.getStage();
    const s0 = samples[order[0]];
    const traceBefore = inferWithTrace(m.net, s0.x);
    // load-bearing：playTrainingStep 前后必须传 w1/w2 的切片快照，活引用会导致 Δw 全零
    const before = { trace: traceBefore, w1: m.net.w1.slice(), w2: m.net.w2.slice(), y: s0.y };
    stepOnce(m, backward(m.net, s0.x, s0.y, traceBefore));
    const traceAfter = inferWithTrace(m.net, s0.x);
    const after = { trace: traceAfter, w1: m.net.w1.slice(), w2: m.net.w2.slice(), y: s0.y };
    await stage.playTrainingStep(before, after); // 教学停留
    for (let k = 1; k < order.length; k++) {
      const s = samples[order[k]];
      stepOnce(m, backward(m.net, s.x, s.y, forward(m.net, s.x)));
    }
    m.epochsTrained = E0 + 1;
    pushHistory(m, samples);
    api.updateKv();
    return { before, after, y: s0.y };
  }

  /**
   * 跑 epochs[firstJ..total]。firstJ===1 时首轮带舞台动画；其余用 trainEpochs
   * 分块（startEpoch=已训练轮数，保证 shuffle 与一次调用完全一致）。
   * @returns {Promise<{done:number, firstStep:object|null}>} 实际完成的轮数与首轮动画步的 before/after
   */
  async function runEpochs(m, samples, seed, total, firstJ) {
    let done = firstJ - 1;
    let firstStep = null;
    if (firstJ === 1) {
      firstStep = await animatedFirstEpoch(m, samples, seed);
      done = 1;
      setTrainStatus(`第 1/${total} 轮完成（含舞台动画）。`);
      await yieldUi();
    }
    for (let j = done + 1; j <= total; j++) {
      if (trainRun && trainRun.paused) break;
      const res = trainEpochs(m.net, m.optState, samples, {
        epochs: 1,
        lr: m.lr,
        optimizer: m.optimizer,
        seed,
        startEpoch: m.epochsTrained, // globalEpoch = 已训练轮数 + 1
        onEpoch: () => (trainRun && trainRun.paused ? false : undefined),
      });
      if (res.cancelled) break;
      m.epochsTrained++;
      done = j;
      pushHistory(m, samples);
      api.updateKv();
      setTrainStatus(`第 ${j}/${total} 轮完成。`);
      await yieldUi();
    }
    return { done, firstStep };
  }

  /** 开始训练（测试入口也调这个；epochs 缺省读输入框）。 */
  async function batchTrain(requestedEpochs) {
    if (trainRun) {
      setTrainStatus('训练进行中，请先暂停或等待完成。');
      return;
    }
    const ws = api.getWorkspace();
    if (!ws) { setTrainStatus('请先选择工作区。'); return; }
    const m = api.getModel();
    if (!m || !m.net) { setTrainStatus('没有可用的模型。'); return; }
    let samples = [];
    try {
      samples = await loadSamples(ws, 'train');
    } catch (e) {
      setTrainStatus(`加载训练样本失败：${e && e.message ? e.message : e}`);
      return;
    }
    if (samples.length === 0) {
      setTrainStatus('训练集为空：请先在左栏手写采集，或在数据集管理中导入示例数据集。');
      return;
    }
    const total = Math.max(1, Math.floor(
      requestedEpochs !== undefined ? requestedEpochs : Number(epochsInput.value),
    ) || 15);
    pausedState = null; // 新一轮训练清除旧的暂停态
    const seed = api.nextRunSeed(); // 每轮训练自增，保证同 seed 可复现
    trainRun = { paused: false, total, done: 0, seed };
    setButtons(false, true, false);
    setControlsEnabled(false);
    setTrainStatus(`训练开始：${samples.length} 个样本，${total} 轮…`);
    let weightsOk = true;
    let firstStep = null;
    try {
      const r = await runEpochs(m, samples, seed, total, 1);
      trainRun.done = r.done;
      firstStep = r.firstStep;
      if (trainRun.paused) {
        pausedState = { total, done: trainRun.done, seed, modelName: m.name };
        setTrainStatus(`已暂停：完成 ${trainRun.done}/${total} 轮。点击「继续」接着训练。`);
        setButtons(false, false, true);
        setControlsEnabled(true); // 暂停时允许调学习率/优化器再继续
      } else if (checkWeightsFinite(m)) {
        setTrainStatus(`训练完成：${total} 轮，轮次累计 ${m.epochsTrained}。`);
        if (api.notifyLesson) api.notifyLesson('trained');
        if (firstStep) {
          try {
            const pp = api.getProcessPanel && api.getProcessPanel();
            if (pp) pp.showTraining({ ...firstStep, lr: m.lr, optimizer: m.optimizer, epoch: m.epochsTrained });
          } catch (e) { console.error('过程面板更新失败：', e); }
        }
      } else {
        weightsOk = false; // NaN：已报错并亮出重置按钮；跳过评估与保存
      }
    } catch (e) {
      setTrainStatus(`训练出错：${e && e.message ? e.message : e}`);
    } finally {
      trainRun = null;
      if (!pausedState) {
        setButtons(true, false, false);
        setControlsEnabled(true);
      }
    }
    if (weightsOk) {
      await runEvaluation();
      await api.saveModelNow();
    }
    await refreshModelSelect();
  }

  btnTrain.addEventListener('click', () => batchTrain());

  btnPause.addEventListener('click', () => {
    if (trainRun) {
      trainRun.paused = true;
      setTrainStatus('正在暂停（当前轮完成后停下）…');
      btnPause.disabled = true;
    }
  });

  btnResume.addEventListener('click', async () => {
    if (trainRun || !pausedState) return;
    const ws = api.getWorkspace();
    const m = api.getModel();
    if (!ws || !m || !m.net) return;
    const { total, done, seed, modelName } = pausedState;
    pausedState = null;
    if (modelName !== m.name) {
      // 暂停期间切换了模型：旧 run 的 seed/轮次不能套用到新模型
      setTrainStatus(`暂停的训练属于模型「${modelName}」，当前是「${m.name}」，已丢弃暂停态。请重新开始训练。`);
      setButtons(true, false, false);
      return;
    }
    let samples = [];
    try {
      samples = await loadSamples(ws, 'train');
    } catch (e) {
      setTrainStatus(`加载训练样本失败：${e && e.message ? e.message : e}`);
      setButtons(true, false, false);
      return;
    }
    if (samples.length === 0) {
      setTrainStatus('训练集为空，无法继续。');
      setButtons(true, false, false);
      return;
    }
    trainRun = { paused: false, total, done, seed };
    setButtons(false, true, false);
    setControlsEnabled(false);
    setTrainStatus(`继续训练：第 ${done + 1}~${total} 轮…`);
    let weightsOk = true;
    try {
      // 同一 runSeed：shuffle 序列与不中断完全一致
      const r2 = await runEpochs(m, samples, seed, total, done + 1);
      const newDone = r2.done;
      trainRun.done = newDone;
      if (trainRun.paused) {
        pausedState = { total, done: newDone, seed, modelName: m.name };
        setTrainStatus(`已暂停：完成 ${newDone}/${total} 轮。点击「继续」接着训练。`);
        setButtons(false, false, true);
        setControlsEnabled(true);
      } else if (checkWeightsFinite(m)) {
        setTrainStatus(`训练完成：${total} 轮，轮次累计 ${m.epochsTrained}。`);
        if (api.notifyLesson) api.notifyLesson('trained');
      } else {
        weightsOk = false; // NaN：已报错并亮出重置按钮；跳过评估与保存
      }
    } catch (e) {
      setTrainStatus(`训练出错：${e && e.message ? e.message : e}`);
    } finally {
      trainRun = null;
      if (!pausedState) {
        setButtons(true, false, false);
        setControlsEnabled(true);
      }
    }
    if (weightsOk) {
      await runEvaluation();
      await api.saveModelNow();
    }
    await refreshModelSelect();
  });

  $('btn-eval').addEventListener('click', async () => {
    evalStatus.textContent = '评估中…';
    await runEvaluation();
  });

  $('btn-dataset').addEventListener('click', () => {
    if (api.openDatasetViewer) api.openDatasetViewer();
  });

  // —— 左栏「保存并训练」的单步训练 ——

  /**
   * 单样本训练一步：推理 trace → 快照 → 优化器步 → 舞台动画 → 计数与节流保存。
   * @param {Float64Array|Array} x 13×13 墨=1 输入（调用方在清空画板前捕获）
   * @param {number} y 标签 0~9
   */
  async function trainSingleStep(x, y) {
    const m = api.getModel();
    const stage = api.getStage();
    if (!m || !m.net || !stage) return;
    if (trainRun) return; // 与批量训练互斥：训练中不接受单步
    if (singleStepBusy) return; // 防重入：上一步动画未完成时跳过
    singleStepBusy = true;
    try {
      const traceBefore = inferWithTrace(m.net, x);
      const before = { trace: traceBefore, w1: m.net.w1.slice(), w2: m.net.w2.slice(), y };
      stepOnce(m, backward(m.net, x, y, traceBefore));
      if (!checkWeightsFinite(m)) return; // 权重已坏：停掉本次单步，不播动画不计数
      const traceAfter = inferWithTrace(m.net, x);
      const after = { trace: traceAfter, w1: m.net.w1.slice(), w2: m.net.w2.slice(), y };
      await stage.playTrainingStep(before, after);
      m.samplesSeen++;
      singleStepCount++;
      api.updateKv();
      try {
        const pp = api.getProcessPanel && api.getProcessPanel();
        if (pp) pp.showTraining({ before, after, y, lr: m.lr, optimizer: m.optimizer, epoch: m.epochsTrained });
      } catch (e) { console.error('过程面板更新失败：', e); }
      // 节流：每 5 次单步保存一次（samplesSeen 丢失至多 4 次，文档写明）
      if (singleStepCount % 5 === 0) await api.saveModelNow();
    } finally {
      singleStepBusy = false;
    }
  }

  // —— 对外 API ——

  /** 样本变更（导入/删除）后：刷新计数 + 重评（矩阵保持新鲜，不展示过期结果）。 */
  async function onSamplesChanged() {
    const ws = api.getWorkspace();
    if (ws) {
      datasetSummary.textContent =
        `训练 ${ws.listSamples('train').length} · 测试 ${ws.listSamples('test').length}`;
    }
    await runEvaluation();
  }

  async function refresh() {
    await refreshModelSelect();
    syncControlsFromModel();
    api.updateKv();
    drawLossCurve();
    if (!lastEval) renderMetrics();
  }

  const panelApi = {
    batchTrain,
    trainSingleStep,
    refresh,
    onSamplesChanged,
    getEval: () => lastEval,
    getModel: () => api.getModel(),
    isTraining: () => trainRun !== null,
    runEvaluation,
    closeCellPopup,
    /** 建模（seed 可选；e2e 用固定 seed 保证确定性）。 */
    newModel: (name, seed) => api.newModel(name, seed),
    /** 保存 / 加载当前模型（e2e round-trip 用）。 */
    saveModelNow: () => api.saveModelNow(),
    loadModel: (name) => api.loadModel(name),
  };
  return panelApi;
}

/** 供 app.js 在删除当前模型后生成不重名的默认名。 */
export function uniqueModelName(base, existing) {
  if (!existing.includes(base)) return base;
  let i = 2;
  while (existing.includes(`${base} ${i}`)) i++;
  return `${base} ${i}`;
}
