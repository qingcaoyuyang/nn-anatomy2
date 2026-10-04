/**
 * Adam 与 SGD 优化器。
 * ES module，零 npm 依赖，无 DOM。
 *
 * # 符号约定（重要）
 * js/network.js 的 `backward` 返回的是**对数似然** ln(p[y])
 * 对参数的梯度（dz2 = onehot(y) − p），与交叉熵损失梯度符号相反。
 * 因此本模块的更新步是梯度**上升**：
 * - SGD：param += lr · grad
 * - Adam：param += lr · m̂ / (√v̂ + eps)（带偏差校正）
 * 梯度上升对数似然 ≡ 梯度下降交叉熵损失。
 *
 * # 参数扁平布局
 * 展平顺序为 w1 ++ b1 ++ w2 ++ b2 拼接；梯度对象采用相同顺序。
 */

export const ADAM_BETA1 = 0.9;
export const ADAM_BETA2 = 0.999;
export const ADAM_EPS = 1e-8;
/** Adam 默认学习率（spec Global Constraints）。 */
export const ADAM_DEFAULT_LR = 0.001;

/** 为 n 个参数创建零初始化的 Adam 状态 { m, v, t }，t 从 0 起计。 */
export function createAdam(n) {
  return { m: new Array(n).fill(0), v: new Array(n).fill(0), t: 0 };
}

/** 网络总参数个数（扁平布局长度）。 */
export function paramCount(net) {
  return net.w1.length + net.b1.length + net.w2.length + net.b2.length;
}

/**
 * Adam 单步（梯度上升）。
 * t 先自增到当前步数再做偏差校正：
 * m = β1·m + (1−β1)·g，v = β2·v + (1−β2)·g²，
 * m̂ = m/(1−β1^t)，v̂ = v/(1−β2^t)，param += lr·m̂/(√v̂ + eps)。
 */
export function adamStep(net, state, grads, lr) {
  const pArrs = [net.w1, net.b1, net.w2, net.b2];
  const gArrs = [grads.gW1, grads.gb1, grads.gW2, grads.gb2];
  const names = ['w1', 'b1', 'w2', 'b2'];
  for (let k = 0; k < 4; k++) {
    if (pArrs[k].length !== gArrs[k].length) {
      throw new Error(`${names[k]} 参数/梯度长度不匹配: ${pArrs[k].length} vs ${gArrs[k].length}`);
    }
  }
  const n = paramCount(net);
  if (state.m.length !== n || state.v.length !== n) {
    throw new Error(`Adam 状态参数个数 ${state.m.length} 与网络 ${n} 不匹配`);
  }

  state.t += 1;
  const t = state.t;
  const bias1 = 1 - Math.pow(ADAM_BETA1, t);
  const bias2 = 1 - Math.pow(ADAM_BETA2, t);

  let i = 0;
  for (let k = 0; k < 4; k++) {
    const p = pArrs[k];
    const g = gArrs[k];
    for (let j = 0; j < p.length; j++, i++) {
      state.m[i] = ADAM_BETA1 * state.m[i] + (1 - ADAM_BETA1) * g[j];
      state.v[i] = ADAM_BETA2 * state.v[i] + (1 - ADAM_BETA2) * g[j] * g[j];
      const mHat = state.m[i] / bias1;
      const vHat = state.v[i] / bias2;
      p[j] += lr * mHat / (Math.sqrt(vHat) + ADAM_EPS);
    }
  }
}

/** SGD 单步（梯度上升）：param += lr · grad。 */
export function sgdStep(net, grads, lr) {
  const pairs = [
    [net.w1, grads.gW1, 'w1'],
    [net.b1, grads.gb1, 'b1'],
    [net.w2, grads.gW2, 'w2'],
    [net.b2, grads.gb2, 'b2'],
  ];
  for (const [p, g, name] of pairs) {
    if (p.length !== g.length) {
      throw new Error(`${name} 参数/梯度长度不匹配: ${p.length} vs ${g.length}`);
    }
    for (let j = 0; j < p.length; j++) p[j] += lr * g[j];
  }
}
