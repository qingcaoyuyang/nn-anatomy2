/**
 * 训练引擎：纯浏览器架构的唯一计算引擎上层。
 * 训练循环（逐样本 forward → backward → 优化器步）+ 评估 + 推理 trace。
 * ES module，零 npm 依赖，无 DOM，不碰 workspace（样本加载是 UI/dataset 的事）。
 *
 * # 符号约定（沿用全局约定）
 * js/network.js 的 `backward` 返回**对数似然** ln(p[y]) 的梯度，
 * 因此这里的优化器步是梯度**上升**（js/optimizer.js 的 adamStep/sgdStep 已实现）。
 *
 * # onEpoch 语义（重要）
 * `onEpoch(epoch /*1-based*\/, { logLikelihood, acc })` 在**每个 epoch 训练开始前**
 * 调用一次：收到的是当前网络（即上一个 epoch 训练后）的指标；返回 `false`
 * 则取消——该 epoch 不执行，已完成的 epoch 保留。返回其他值（含 undefined）
 * 继续训练。训练循环里不做每样本回调。
 *
 * # 确定性契约
 * 每个 epoch 用独立 seed 流做 Fisher–Yates shuffle：
 * `createRng(seed * 100003 + startEpoch + epoch)`（seed 先经 createRng 截断到 u32）。
 * 同 seed + 同初始网络 + 同样本顺序 ⇒ 两次 trainEpochs 权重 bit-identical。
 */

import { forward, backward } from './network.js';
import { createRng } from './rng.js';
import {
  ADAM_DEFAULT_LR,
  createAdam,
  paramCount,
  adamStep,
  sgdStep,
} from './optimizer.js';

/**
 * 为网络创建优化器状态：adam → createAdam(paramCount(net))；sgd → null。
 * 未知 optimizer 名直接抛错（静默用错优化器比报错更危险）。
 */
export function createOptState(net, optimizer = 'adam') {
  if (optimizer === 'adam') return createAdam(paramCount(net));
  if (optimizer === 'sgd') return null;
  throw new Error(`未知 optimizer: ${optimizer}`);
}

/** argmax：取第一个最大值的下标（平局取最小下标，保证确定性）。 */
function argmax(p) {
  let best = 0;
  for (let o = 1; o < p.length; o++) {
    if (p[o] > p[best]) best = o;
  }
  return best;
}

/** Fisher–Yates shuffle：返回 0..n-1 的一个随机排列（原地打乱副本）。 */
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

/**
 * 当前网络在样本集上的指标：平均对数似然 + 准确率。
 * logLikelihood = 全部样本 ln(p[y]) 的均值；acc = argmax 命中率。
 */
function currentMetrics(net, samples) {
  let llSum = 0;
  let correct = 0;
  for (const s of samples) {
    const { p } = forward(net, s.x);
    llSum += Math.log(p[s.y]);
    if (argmax(p) === s.y) correct++;
  }
  return { logLikelihood: llSum / samples.length, acc: correct / samples.length };
}

/**
 * 训练若干 epoch。
 * @param {object} net 网络（createNetwork 产物，就地更新）
 * @param {object|null} optState 优化器状态（createOptState 产物；sgd 传 null）
 * @param {Array<{x:number[169], y:number}>} samples 已预处理的样本
 * @param {object} opts { epochs=1, lr=ADAM_DEFAULT_LR, optimizer='adam'|'sgd',
 *                        seed=1, startEpoch=0, onEpoch }
 * @returns {{ epochsDone:number, logLikelihood:number|null, cancelled:boolean }}
 *   logLikelihood 为最终网络的平均对数似然；空样本/epochs<=0 时直接返回
 *   { epochsDone: 0, logLikelihood: null, cancelled: false }。
 *
 * # startEpoch（供 UI 分块训练）
 * UI 若需每 epoch 让出主线程保持可交互，可循环调用
 * `trainEpochs(net, st, samples, { epochs: 1, seed, startEpoch: k })`（k=0,1,2…）。
 * startEpoch 会叠加进 seed 流与 onEpoch 的 epoch 编号，保证分块调用与一次
 * `trainEpochs({ epochs: N })` 的 shuffle 序列、指标编号完全一致。
 */
export function trainEpochs(net, optState, samples, opts = {}) {
  const {
    epochs = 1,
    lr = ADAM_DEFAULT_LR,
    optimizer = 'adam',
    seed = 1,
    startEpoch = 0,
    onEpoch,
  } = opts;

  const n = samples.length;
  const epochsInt = Math.floor(epochs);
  if (n === 0 || epochsInt <= 0) {
    return { epochsDone: 0, logLikelihood: null, cancelled: false };
  }

  const useAdam = optimizer === 'adam';
  if (!useAdam && optimizer !== 'sgd') {
    throw new Error(`未知 optimizer: ${optimizer}`);
  }
  if (useAdam && !optState) {
    throw new Error('adam 需要 optState（用 createOptState(net, "adam") 创建）');
  }

  for (let epoch = 1; epoch <= epochsInt; epoch++) {
    const globalEpoch = startEpoch + epoch;
    // 门控：本 epoch 开始前先给 UI 一次"看指标、决定是否继续"的机会
    const metrics = currentMetrics(net, samples);
    if (onEpoch && onEpoch(globalEpoch, metrics) === false) {
      return { epochsDone: epoch - 1, logLikelihood: metrics.logLikelihood, cancelled: true };
    }

    // 独立 seed 流的 shuffle，保证同 (seed, startEpoch) 完全可复现
    const rng = createRng(seed * 100003 + globalEpoch);
    const order = shuffledIndices(n, rng);
    for (const i of order) {
      const s = samples[i];
      const fwd = forward(net, s.x);
      const grads = backward(net, s.x, s.y, fwd);
      if (useAdam) adamStep(net, optState, grads, lr);
      else sgdStep(net, grads, lr);
    }
  }

  const finalMetrics = currentMetrics(net, samples);
  return { epochsDone: epochsInt, logLikelihood: finalMetrics.logLikelihood, cancelled: false };
}

/**
 * 在样本集上评估。
 * @returns {{ acc:number|null, perClass:Array<number|null>, confusion:number[10][10] }}
 *   confusion[true][pred] 计数；perClass[c] 为真值 c 的准确率（该类无样本时 null）；
 *   空样本时 acc=null 且不抛错（UI 层显示"无数据"）。
 */
export function evaluate(net, samples) {
  const confusion = Array.from({ length: 10 }, () => new Array(10).fill(0));
  const perClass = new Array(10).fill(null);
  if (samples.length === 0) return { acc: null, perClass, confusion };

  const correct = new Array(10).fill(0);
  const total = new Array(10).fill(0);
  let allCorrect = 0;
  for (const s of samples) {
    const { p } = forward(net, s.x);
    const pred = argmax(p);
    confusion[s.y][pred]++;
    total[s.y]++;
    if (pred === s.y) {
      correct[s.y]++;
      allCorrect++;
    }
  }
  for (let c = 0; c < 10; c++) {
    perClass[c] = total[c] > 0 ? correct[c] / total[c] : null;
  }
  return { acc: allCorrect / samples.length, perClass, confusion };
}

/**
 * 带全部中间值的单样本推理（供 Task 10 舞台动画用）。
 * @returns {{ z1, a1, z2, p, pred }} pred = argmax p（平局取最小下标）
 */
export function inferWithTrace(net, x) {
  const fwd = forward(net, x);
  return { ...fwd, pred: argmax(fwd.p) };
}
