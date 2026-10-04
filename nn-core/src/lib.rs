//! nn-core：纯 Rust 神经网络计算核心（无 Tauri 依赖）。
//!
//! Task 1 仅提供占位；后续任务（Task 2+）将在此添加 `nn` 模块
//! （MLP 169→16→10、RNG、训练循环等），并与 `src/nn/*.js` 的 JS
//! 实现做逐位一致性校验。

/// 占位函数：验证纯逻辑 crate 可编译、可单元测试。
pub fn placeholder() -> &'static str {
    "nn-core 就绪"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_works() {
        assert_eq!(placeholder(), "nn-core 就绪");
    }
}
