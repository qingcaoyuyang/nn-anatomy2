//! 神经网络模块：MLP 169→16→10（LeakyReLU + Softmax + 对数似然梯度，
//! 符号与交叉熵损失相反）。
//!
//! 本模块纯 Rust、零 Tauri 依赖；`Network::new` 使用手写 mulberry32 + Box-Muller
//! 做 He 初始化，算法细节见 `network.rs` 顶部的 RNG 确定性契约注释（供 Task 5
//! 的 JS 实现逐位复现）。

pub mod network;

pub use network::{Arch, ForwardOut, Gradients, Network};
