//! MLP 网络核心：169→16→10，LeakyReLU(0.01) + Softmax + 交叉熵，He 初始化。
//!
//! # RNG 确定性契约（Task 5 的 JS 实现必须逐位复现以下算法）
//!
//! ## mulberry32
//! - 状态：`u32`；`Network::new(arch, seed: u64)` 取 seed 的低 32 位
//!   （`seed as u32`，即 `seed & 0xFFFFFFFF`）作为初始状态。
//! - 单步（标准 mulberry32，Rust wrapping 语义）：
//!   ```text
//!   t = state + 0x6D2B79F5        (u32 wrapping 加法；state 先更新为 t)
//!   t = (t ^ (t >> 15)) * (t | 1)  (u32 wrapping 乘法)
//!   t = t ^ (t + ((t ^ (t >> 7)) * (t | 61)))   (u32 wrapping 加法/乘法)
//!   out = t ^ (t >> 14)
//!   ```
//! - 均匀数：`next_f64() = next_u32() as f64 / 4294967296.0`，范围 [0, 1)。
//!   `u32 as f64` 精确（u32 < 2^53），除以 2^32 是精确的 2 的幂次除法，
//!   与 JS 的 `(t >>> 0) / 4294967296` 逐位一致。
//!
//! ## Box-Muller（标准形式，非极坐标法）
//! - 每个权重消耗**一对**连续均匀数：`u1 = next_f64()`（先），`u2 = next_f64()`（后）。
//! - 只使用 z0，**丢弃 z1**（不缓存）：
//!   ```text
//!   z0 = sqrt(-2 * ln(u1)) * cos(2π * u2)
//!   ```
//! - 权重 `w = z0 * sqrt(2 / fan_in)`（He 缩放）。
//! - 偏置全部初始化为 0.0（不消耗 RNG）。
//!
//! ## 抽取顺序
//! 1. `w1`：`inputs*hidden` 个，按行优先 `w1[h*inputs + x]`（h=0..hidden，x=0..inputs），
//!    `fan_in = inputs`，`scale = sqrt(2/inputs)`。
//! 2. `b1`：全 0（不消耗 RNG）。
//! 3. `w2`：`hidden*outputs` 个，按行优先 `w2[o*hidden + h]`，
//!    `fan_in = hidden`，`scale = sqrt(2/hidden)`。
//! 4. `b2`：全 0（不消耗 RNG）。
//!
//! ## 前向 / 反向约定
//! - `z1[h] = Σ_x w1[h*inputs+x]*x[x] + b1[h]`；
//!   `a1[h] = z1[h] > 0 ? z1[h] : 0.01*z1[h]`（z1 == 0 时取 0.01 分支，
//!   导数在 z1 <= 0 时为 0.01，z1 > 0 时为 1.0）。
//! - `z2[o] = Σ_h w2[o*hidden+h]*a1[h] + b2[o]`；
//!   `p = softmax(z2)`（实现时先减 max 做数值稳定，不影响数学值）。
//! - **反向符号约定（重要）**：`backward` 返回的是对数似然 `ln(p[y])`
//!   对参数的梯度（由验收测试的数值微分定义：
//!   `num = (ln p_up[y] - ln p_dn[y]) / 2eps`），
//!   即 `d ln(p[y])/dz2[o] = onehot(y)[o] - p[o]`，
//!   与交叉熵损失 `L = -ln(p[y])` 的梯度符号**相反**。
//!   训练时必须做梯度**上升** `w += lr * g`（等价于交叉熵梯度下降），
//!   Task 3 的训练循环须遵守此约定。

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Arch {
    pub inputs: usize,
    pub hidden: usize,
    pub outputs: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
    pub arch: Arch,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

#[derive(Clone)]
pub struct ForwardOut {
    pub z1: Vec<f64>,
    pub a1: Vec<f64>,
    pub z2: Vec<f64>,
    pub p: Vec<f64>,
}

#[derive(Clone)]
pub struct Gradients {
    pub g_w1: Vec<f64>,
    pub g_b1: Vec<f64>,
    pub g_w2: Vec<f64>,
    pub g_b2: Vec<f64>,
}

/// LeakyReLU 负半轴斜率（全局约束）。
const LEAKY_ALPHA: f64 = 0.01;

/// 手写 mulberry32（u32 状态）。
///
/// 逐位复现要点（供 Task 5 JS 实现）：
/// - 初始状态 = `seed` 的低 32 位（`seed as u32`）。
/// - 单步：`t = state.wrapping_add(0x6D2B79F5)`（state 先更新为 t）；
///   `t = (t ^ (t >> 15)).wrapping_mul(t | 1)`；
///   `t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61))`；
///   输出 `t ^ (t >> 14)`。与标准 JS 版 `Math.imul` 语义逐位一致。
/// - `next_f64() = next_u32() as f64 / 4294967296.0`，范围 [0, 1)，
///   与 JS `(t >>> 0) / 4294967296` 逐位一致。
struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    fn new(seed: u64) -> Self {
        Self { state: seed as u32 }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        let mut t = self.state.wrapping_add(0x6D2B79F5);
        self.state = t;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t ^ (t >> 14)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / 4294967296.0
    }

    /// 标准 Box-Muller：消耗一对**连续**均匀数（u1 先、u2 后），
    /// 只返回 z0 = sqrt(-2·ln(u1))·cos(2π·u2)，**丢弃 z1，不缓存**。
    #[inline]
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }
}

impl Network {
    /// He 初始化：w1 行优先（`w1[h*inputs+x]`，fan_in=inputs），
    /// 然后 b1 全 0（不消耗 RNG），接着 w2 行优先（fan_in=hidden），
    /// 最后 b2 全 0。每权重 `w = z0 * sqrt(2/fan_in)`。
    pub fn new(arch: Arch, seed: u64) -> Network {
        let mut rng = Mulberry32::new(seed);
        let scale1 = (2.0 / arch.inputs as f64).sqrt();
        let scale2 = (2.0 / arch.hidden as f64).sqrt();

        let mut w1 = Vec::with_capacity(arch.hidden * arch.inputs);
        for _ in 0..arch.hidden * arch.inputs {
            w1.push(rng.next_normal() * scale1);
        }
        let b1 = vec![0.0; arch.hidden];

        let mut w2 = Vec::with_capacity(arch.outputs * arch.hidden);
        for _ in 0..arch.outputs * arch.hidden {
            w2.push(rng.next_normal() * scale2);
        }
        let b2 = vec![0.0; arch.outputs];

        Network {
            arch,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// 前向：z1 → LeakyReLU → z2 → Softmax（减 max 做数值稳定）。
    pub fn forward(&self, x: &[f64]) -> ForwardOut {
        let (n_in, n_h, n_out) = (self.arch.inputs, self.arch.hidden, self.arch.outputs);
        debug_assert_eq!(x.len(), n_in);

        let mut z1 = vec![0.0; n_h];
        let mut a1 = vec![0.0; n_h];
        for h in 0..n_h {
            let mut s = self.b1[h];
            let row = h * n_in;
            for xi in 0..n_in {
                s += self.w1[row + xi] * x[xi];
            }
            z1[h] = s;
            a1[h] = if s > 0.0 { s } else { LEAKY_ALPHA * s };
        }

        let mut z2 = vec![0.0; n_out];
        for o in 0..n_out {
            let mut s = self.b2[o];
            let row = o * n_h;
            for h in 0..n_h {
                s += self.w2[row + h] * a1[h];
            }
            z2[o] = s;
        }

        let max = z2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut p = vec![0.0; n_out];
        let mut sum = 0.0;
        for o in 0..n_out {
            let e = (z2[o] - max).exp();
            p[o] = e;
            sum += e;
        }
        for p_o in p.iter_mut() {
            *p_o /= sum;
        }

        ForwardOut { z1, a1, z2, p }
    }

    /// 反向：返回对数似然 `ln(p[y])` 对全部参数的解析梯度。
    /// 注意符号：`d ln(p[y])/dz2[o] = onehot(y)[o] - p[o]`，
    /// 与交叉熵损失梯度相反（见模块顶部"反向符号约定"）。
    /// LeakyReLU 导数在 z1>0 为 1，z1<=0 为 0.01（与 forward 分支一致）。
    pub fn backward(&self, x: &[f64], y: usize, f: &ForwardOut) -> Gradients {
        let (n_in, n_h, n_out) = (self.arch.inputs, self.arch.hidden, self.arch.outputs);

        let mut dz2 = vec![0.0; n_out];
        for o in 0..n_out {
            dz2[o] = if o == y { 1.0 } else { 0.0 } - f.p[o];
        }

        let mut g_w2 = vec![0.0; n_out * n_h];
        let mut g_b2 = vec![0.0; n_out];
        for o in 0..n_out {
            g_b2[o] = dz2[o];
            let row = o * n_h;
            for h in 0..n_h {
                g_w2[row + h] = dz2[o] * f.a1[h];
            }
        }

        let mut dz1 = vec![0.0; n_h];
        for h in 0..n_h {
            let mut s = 0.0;
            for o in 0..n_out {
                s += self.w2[o * n_h + h] * dz2[o];
            }
            let d_relu = if f.z1[h] > 0.0 { 1.0 } else { LEAKY_ALPHA };
            dz1[h] = s * d_relu;
        }

        let mut g_w1 = vec![0.0; n_h * n_in];
        let mut g_b1 = vec![0.0; n_h];
        for h in 0..n_h {
            g_b1[h] = dz1[h];
            let row = h * n_in;
            for xi in 0..n_in {
                g_w1[row + xi] = dz1[h] * x[xi];
            }
        }

        Gradients {
            g_w1,
            g_b1,
            g_w2,
            g_b2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_outputs_valid_softmax() {
        let net = Network::new(
            Arch {
                inputs: 169,
                hidden: 16,
                outputs: 10,
            },
            42,
        );
        let x = vec![0.5; 169];
        let f = net.forward(&x);
        assert_eq!(f.p.len(), 10);
        let sum: f64 = f.p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn backward_matches_numerical_gradient() {
        let net = Network::new(
            Arch {
                inputs: 169,
                hidden: 16,
                outputs: 10,
            },
            7,
        );
        let x: Vec<f64> = (0..169).map(|i| ((i * 37) % 13) as f64 / 13.0).collect();
        let y = 3;
        let f = net.forward(&x);
        let g = net.backward(&x, y, &f);
        let eps = 1e-5;
        // 校验 W1[3*169+10]、W2[4]、B1[5]
        let cases = [
            (0, 3 * 169 + 10, g.g_w1[3 * 169 + 10]),
            (1, 4, g.g_w2[4]),
            (2, 5, g.g_b1[5]),
        ];
        for (which, idx, ana) in cases {
            let mut up = net.clone();
            let mut dn = net.clone();
            match which {
                0 => {
                    up.w1[idx] += eps;
                    dn.w1[idx] -= eps;
                }
                1 => {
                    up.w2[idx] += eps;
                    dn.w2[idx] -= eps;
                }
                _ => {
                    up.b1[idx] += eps;
                    dn.b1[idx] -= eps;
                }
            }
            let lp = up.forward(&x).p[y].ln();
            let ld = dn.forward(&x).p[y].ln();
            let num = (lp - ld) / (2.0 * eps);
            assert!((num - ana).abs() < 1e-6, "梯度不匹配: {} vs {}", num, ana);
        }
    }
}
