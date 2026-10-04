# 归档说明：Tauri v1 方案（已停止）

2026-10-04 深夜，项目从 **Tauri 2 桌面应用** 方案切换为
**纯浏览器 + File System Access API** 方案（用户决策）。

## 为什么切换

- 169→16→10 的网络太小：1000 样本 × 15 epoch 纯 JS 几百毫秒跑完，
  Rust 批量引擎 + 双语言交叉验证属于过度设计。
- 纯浏览器 = 零安装：教室电脑打开 Chrome 就能投屏；
  不再需要 Rust 工具链、Xcode、.dmg 签名。
- 云端可用无头 Chromium 全链路验证（含截图），Tauri 部分在云端验证不了。

## 这里是什么

- `nn-core/`：Rust 网络核心（前向/反向/Adam/SGD/预处理），Task 2–4 的成果，
  测试全绿。算法文档（符号约定、RNG 规范）已迁移到新方案的 JS 实现中。
- `src-tauri/`：Tauri 应用壳（未完成的 Task 1 收尾）。
- `main.js` / `tauri-bridge.js`：Task 1 的前端问候 stub。
- `Cargo.toml` / `Cargo.lock`：工作区清单（`Cargo.lock` 当时未入库）。

## 保留的 JS 成果（未归档，仍在用）

- `js/preprocess.js`：Task 4 的 JS 预处理实现（Rust↔JS 交叉验证 max|Δ|=0），
  新方案直接沿用。
- `css/app.css`：Task 1 的基础样式。

Git 历史完整保留（`git log --follow` 可查）。
