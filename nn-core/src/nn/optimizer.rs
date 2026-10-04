//! Adam 与 SGD 优化器。
//!
//! # 符号约定（重要）
//!
//! `network::Network::backward` 返回的是**对数似然** `ln(p[y])`
//! 对参数的梯度（`dz2 = onehot(y) - p`），与交叉熵损失梯度符号相反。
//! 因此本模块的更新步是**梯度上升**：
//!
//! - SGD：`param += lr * grad`
//! - Adam：`param += lr * m_hat / (sqrt(v_hat) + eps)`（带偏差校正）
//!
//! 梯度上升对数似然 ≡ 梯度下降交叉熵损失。
//!
//! # 参数扁平布局
//!
//! 展平顺序为 `w1 ++ b1 ++ w2 ++ b2` 拼接；梯度 `Gradients` 采用相同顺序。

use super::network::{Gradients, Network};

/// Adam 一阶矩衰减率。
pub const ADAM_BETA1: f64 = 0.9;
/// Adam 二阶矩衰减率。
pub const ADAM_BETA2: f64 = 0.999;
/// Adam 数值稳定项。
pub const ADAM_EPS: f64 = 1e-8;

/// Adam 优化器状态：每参数一阶矩 `m`、二阶矩 `v`、已执行步数 `t`。
#[derive(Clone)]
pub struct AdamState {
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub t: u64,
}

impl AdamState {
    /// 为 `param_count` 个参数创建零初始化状态。
    pub fn new(param_count: usize) -> Self {
        Self {
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
            t: 0,
        }
    }

    /// 状态跟踪的参数个数。
    pub fn param_count(&self) -> usize {
        self.m.len()
    }
}

/// 网络总参数个数（扁平布局长度）。
pub fn param_count(net: &Network) -> usize {
    net.w1.len() + net.b1.len() + net.w2.len() + net.b2.len()
}

/// 按 `w1 ++ b1 ++ w2 ++ b2` 展平参数。
fn flatten_params(net: &Network) -> Vec<f64> {
    let mut p = Vec::with_capacity(param_count(net));
    p.extend_from_slice(&net.w1);
    p.extend_from_slice(&net.b1);
    p.extend_from_slice(&net.w2);
    p.extend_from_slice(&net.b2);
    p
}

/// 把扁平参数写回网络（与 [`flatten_params`] 逆操作）。
fn write_params(net: &mut Network, flat: &[f64]) {
    let mut it = flat.iter();
    for w in net.w1.iter_mut() {
        *w = *it.next().expect("flat length mismatch");
    }
    for b in net.b1.iter_mut() {
        *b = *it.next().expect("flat length mismatch");
    }
    for w in net.w2.iter_mut() {
        *w = *it.next().expect("flat length mismatch");
    }
    for b in net.b2.iter_mut() {
        *b = *it.next().expect("flat length mismatch");
    }
    assert!(it.next().is_none(), "flat length mismatch");
}

/// 按 `w1 ++ b1 ++ w2 ++ b2` 展平梯度。
fn flatten_grads(grads: &Gradients) -> Vec<f64> {
    let mut g = Vec::with_capacity(
        grads.g_w1.len() + grads.g_b1.len() + grads.g_w2.len() + grads.g_b2.len(),
    );
    g.extend_from_slice(&grads.g_w1);
    g.extend_from_slice(&grads.g_b1);
    g.extend_from_slice(&grads.g_w2);
    g.extend_from_slice(&grads.g_b2);
    g
}

/// Adam 单步（梯度上升）。
///
/// `t` 先自增到当前步数，再做偏差校正：
/// `m_t = β1·m + (1-β1)·g`，`v_t = β2·v + (1-β2)·g²`，
/// `m̂ = m_t/(1-β1^t)`，`v̂ = v_t/(1-β2^t)`，
/// `param += lr · m̂ / (√v̂ + eps)`。
pub fn adam_step(net: &mut Network, state: &mut AdamState, grads: &Gradients, lr: f64) {
    let g = flatten_grads(grads);
    assert_eq!(
        g.len(),
        state.param_count(),
        "gradient count {} != state param count {}",
        g.len(),
        state.param_count()
    );
    state.t += 1;
    let t = state.t as f64;
    let bias1 = 1.0 - ADAM_BETA1.powf(t);
    let bias2 = 1.0 - ADAM_BETA2.powf(t);
    let mut p = flatten_params(net);
    assert_eq!(p.len(), g.len(), "param/gradient length mismatch");
    for i in 0..p.len() {
        state.m[i] = ADAM_BETA1 * state.m[i] + (1.0 - ADAM_BETA1) * g[i];
        state.v[i] = ADAM_BETA2 * state.v[i] + (1.0 - ADAM_BETA2) * g[i] * g[i];
        let m_hat = state.m[i] / bias1;
        let v_hat = state.v[i] / bias2;
        p[i] += lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
    }
    write_params(net, &p);
}

/// SGD 单步（梯度上升）：`param += lr * grad`。
pub fn sgd_step(net: &mut Network, grads: &Gradients, lr: f64) {
    assert_eq!(net.w1.len(), grads.g_w1.len(), "w1/grad length mismatch");
    assert_eq!(net.b1.len(), grads.g_b1.len(), "b1/grad length mismatch");
    assert_eq!(net.w2.len(), grads.g_w2.len(), "w2/grad length mismatch");
    assert_eq!(net.b2.len(), grads.g_b2.len(), "b2/grad length mismatch");
    for (w, g) in net.w1.iter_mut().zip(grads.g_w1.iter()) {
        *w += lr * g;
    }
    for (b, g) in net.b1.iter_mut().zip(grads.g_b1.iter()) {
        *b += lr * g;
    }
    for (w, g) in net.w2.iter_mut().zip(grads.g_w2.iter()) {
        *w += lr * g;
    }
    for (b, g) in net.b2.iter_mut().zip(grads.g_b2.iter()) {
        *b += lr * g;
    }
}

#[cfg(test)]
mod tests {
    use super::super::network::Arch;
    use super::*;

    /// 所有参数均为 0.25 的 2→2→2 玩具网络。
    fn toy_net() -> Network {
        Network {
            arch: Arch {
                inputs: 2,
                hidden: 2,
                outputs: 2,
            },
            w1: vec![0.25; 4],
            b1: vec![0.25; 2],
            w2: vec![0.25; 4],
            b2: vec![0.25; 2],
        }
    }

    /// 全为常数 `g` 的梯度（形状匹配 [`toy_net`]）。
    fn const_grads(g: f64) -> Gradients {
        Gradients {
            g_w1: vec![g; 4],
            g_b1: vec![g; 2],
            g_w2: vec![g; 4],
            g_b2: vec![g; 2],
        }
    }

    fn all_params(net: &Network) -> Vec<f64> {
        flatten_params(net)
    }

    #[test]
    fn adam_state_new_is_zeroed() {
        let s = AdamState::new(12);
        assert_eq!(s.t, 0);
        assert_eq!(s.param_count(), 12);
        assert!(s.m.iter().all(|&x| x == 0.0));
        assert!(s.v.iter().all(|&x| x == 0.0));
    }

    /// Adam 第一步：t=1 时偏差校正使 `m̂ = g`、`v̂ = g²`，
    /// 故 `|Δw| = lr · |g| / (|g| + eps) ≈ lr`。
    #[test]
    fn adam_first_step_delta_is_lr_after_bias_correction() {
        let lr = 0.01;
        for &g in &[0.5, -0.3] {
            let mut net = toy_net();
            let mut state = AdamState::new(param_count(&net));
            adam_step(&mut net, &mut state, &const_grads(g), lr);
            assert_eq!(state.t, 1, "t should increment to 1");
            // 矩更新：m = (1-β1)·g，v = (1-β2)·g²
            assert!((state.m[0] - (1.0 - ADAM_BETA1) * g).abs() < 1e-15);
            assert!((state.v[0] - (1.0 - ADAM_BETA2) * g * g).abs() < 1e-15);
            let expected_delta = lr * g / (g.abs() + ADAM_EPS);
            for w in all_params(&net) {
                let delta = w - 0.25;
                assert!(
                    (delta - expected_delta).abs() < 1e-12,
                    "delta {delta} != expected {expected_delta}"
                );
                // |Δw| ≈ lr（eps 只引入 ~2e-8 相对误差）
                assert!(
                    (delta.abs() - lr).abs() < 1e-9,
                    "|delta| {} not ≈ lr {lr}",
                    delta.abs()
                );
            }
            // 上升方向：正梯度→参数增大，负梯度→参数减小
            assert_eq!(all_params(&net)[0] > 0.25, g > 0.0);
        }
    }

    /// Adam 第二步 t=2 且继续按偏差校正更新。
    #[test]
    fn adam_second_step_increments_t() {
        let mut net = toy_net();
        let mut state = AdamState::new(param_count(&net));
        let grads = const_grads(0.5);
        adam_step(&mut net, &mut state, &grads, 0.01);
        adam_step(&mut net, &mut state, &grads, 0.01);
        assert_eq!(state.t, 2);
        // 两步同向梯度：m̂ = 0.5（恒定梯度下偏差校正后 m̂ ≡ g）
        let t = 2.0f64;
        let bias1 = 1.0 - ADAM_BETA1.powf(t);
        let m_expected = (1.0 - ADAM_BETA1) * 0.5 * (1.0 + ADAM_BETA1) / bias1;
        assert!((state.m[0] / bias1 - m_expected).abs() < 1e-15);
    }

    /// SGD：`Δw = lr · grad`（上升）。
    #[test]
    fn sgd_step_is_lr_times_grad() {
        let mut net = toy_net();
        let lr = 0.1;
        sgd_step(&mut net, &const_grads(0.5), lr);
        for w in all_params(&net) {
            assert!(
                (w - (0.25 + lr * 0.5)).abs() < 1e-15,
                "w {w} != 0.25 + lr*0.5"
            );
        }
        // 负梯度：参数减小
        let mut net2 = toy_net();
        sgd_step(&mut net2, &const_grads(-2.0), lr);
        for w in all_params(&net2) {
            assert!((w - (0.25 - lr * 2.0)).abs() < 1e-15);
        }
    }

    /// 扁平布局顺序：w1 ++ b1 ++ w2 ++ b2。
    #[test]
    fn flat_layout_order_is_w1_b1_w2_b2() {
        let mut net = Network {
            arch: Arch {
                inputs: 2,
                hidden: 2,
                outputs: 2,
            },
            w1: vec![1.0, 2.0, 3.0, 4.0],
            b1: vec![5.0, 6.0],
            w2: vec![7.0, 8.0, 9.0, 10.0],
            b2: vec![11.0, 12.0],
        };
        // grads 全 1：SGD(lr=1) 后每个参数 +1，顺序即扁平顺序
        sgd_step(
            &mut net,
            &const_grads(1.0),
            1.0,
        );
        assert_eq!(
            all_params(&net),
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
        );
    }

    /// 100 步 Adam 训练：交叉熵损失（= -ln p[y]）显著下降。
    #[test]
    fn hundred_adam_steps_decrease_loss() {
        let arch = Arch {
            inputs: 169,
            hidden: 16,
            outputs: 10,
        };
        let mut net = Network::new(arch, 42);
        // 确定性非零输入
        let x: Vec<f64> = (0..169)
            .map(|i| 0.1 + 0.8 * (((i * 37 + 11) % 100) as f64) / 100.0)
            .collect();
        let y = 3usize;
        let loss = |net: &Network| -net.forward(&x).p[y].ln();
        let l0 = loss(&net);
        assert!(l0 > 1.0, "sanity: initial loss {l0} should be well above 0");

        let mut state = AdamState::new(param_count(&net));
        for _ in 0..100 {
            let f = net.forward(&x);
            let g = net.backward(&x, y, &f);
            adam_step(&mut net, &mut state, &g, 0.01);
        }
        let l100 = loss(&net);
        assert!(
            l100 < l0 - 0.3,
            "loss did not decrease enough: {l0} -> {l100}"
        );
    }
}
