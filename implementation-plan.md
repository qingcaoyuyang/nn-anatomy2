# nn-anatomy 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 构建桌面端神经网络解剖教室（Tauri 2），手写数字识别全程可视化，数据以 PNG/JSON 文件落盘。

**Architecture:** Tauri 2 应用：Rust 后端承担计算内核（前向/反向/优化器）与工作区文件系统（PNG 数据集、模型 JSON）；前端为原生 ES Modules + Canvas 2D 的 WebView UI。计算分双层引擎：JS 教学引擎（逐样本动画、暴露全部中间值）与 Rust 批量引擎（快进 epoch），固定种子交叉验证保证数值一致。

**Tech Stack:** Rust（Tauri 2）、JavaScript ES Modules、Canvas 2D、CSS、cargo test + node --test。

## Global Constraints

- 网络结构固定：169 输入 → 16 隐藏 → 10 输出，LeakyReLU(0.01)，Softmax + 交叉熵。
- 优化器默认 Adam（lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8），训练界面可切 SGD。
- 预处理管线：260×260 画板 → 重心居中 → 降采样 13×13 → 0~1 灰度。
- 工作区目录结构、manifest 字段、模型文件命名 `.nnmodel.json` 按 spec 第 3 节执行。
- 界面语言：全中文；桌面优先（投屏 1024px+），平板（触屏）兼容，768px 以下单列。
- 设计系统：主色 #F97316 暖橙、背景 #FFF7ED 米白、深灰文字、单一强调色、单一圆角系统（8px）、全页面单一浅色主题。
- 数值（像素值、概率、loss、准确率）一律等宽字体。
- 前端零 npm 运行时依赖、无 CDN；字体本地打包（Baloo 2 + Noto Sans SC + 等宽）。
- JS 与 Rust 引擎数值一致性：固定种子训练 3 epoch，权重逐元素误差 < 1e-9。
- 代码与文档全部位于 nn-anatomy/ 单目录。

---

### Task 1: Tauri 2 项目骨架

**Files:**
- Create: src-tauri/Cargo.toml
- Create: src-tauri/tauri.conf.json
- Create: src-tauri/src/main.rs
- Create: src-tauri/build.rs
- Create: src-tauri/icons/（cargo tauri icon 生成）
- Create: index.html
- Create: src/main.js
- Create: src/tauri-bridge.js
- Create: src/ui/app.css
- Create: .gitignore

**Interfaces:**
- Produces: 可运行的 cargo tauri dev 骨架，后续任务在其上叠加
- Produces: Rust command `greet(name: String) -> String`（占位，验证前后端桥接）

- [ ] **Step 1: 初始化**

```bash
cd /Users/juewei/Documents/Codex/OtherWork/nn-anatomy
cargo install tauri-cli --locked
mkdir -p src-tauri/src src/ui src/nn tests scripts
touch index.html src/main.js src/tauri-bridge.js src/ui/app.css src-tauri/src/main.rs src-tauri/build.rs src-tauri/Cargo.toml src-tauri/tauri.conf.json .gitignore
```

- [ ] **Step 2: 写配置与骨架代码**

```toml
# src-tauri/Cargo.toml
[package]
name = "nn-anatomy"
version = "0.1.0"
edition = "2021"

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = [] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
image = "0.25"
tempdir = { version = "0.3", optional = true }

[dev-dependencies]
tempdir = "0.3"
```

```rust
// src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[tauri::command]
fn greet(name: &str) -> String {
    format!("你好，{}！神经网络解剖教室准备就绪。", name)
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

```json
// src-tauri/tauri.conf.json
{
  "productName": "nn-anatomy",
  "version": "0.1.0",
  "identifier": "com.juewei.nn-anatomy",
  "build": { "frontendDist": "../", "devUrl": "http://localhost:5173" },
  "app": {
    "withGlobalTauri": true,
    "windows": [{ "title": "nn-anatomy 神经网络解剖教室", "width": 1280, "height": 800 }]
  },
  "bundle": { "active": true, "targets": ["dmg", "app"] }
}
```

```rust
// src-tauri/build.rs
fn main() { tauri_build::build() }
```

```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>nn-anatomy 神经网络解剖教室</title>
  <link rel="stylesheet" href="src/ui/app.css" />
</head>
<body>
  <div id="app"></div>
  <script type="module" src="src/main.js"></script>
</body>
</html>
```

```js
// src/main.js
import { greet } from './tauri-bridge.js';
document.querySelector('#app').textContent = await greet('舟哥');
```

```js
// src/tauri-bridge.js
export async function invoke(cmd, args) {
  return window.__TAURI__.core.invoke(cmd, args);
}
export async function greet(name) { return invoke('greet', { name }); }
```

- [ ] **Step 3: 首次编译验证**

Run: cd /Users/juewei/Documents/Codex/OtherWork/nn-anatomy && cargo tauri dev
Expected: 窗口打开，显示中文问候

- [ ] **Step 4: 应用图标（教学视觉一致性）**

```bash
# 用 Canvas 画 1024×1024 节点连线图标导出 PNG 后：
cargo tauri icon icon-1024.png
```

- [ ] **Step 5: 提交 git commit -m "feat: tauri skeleton with bridge"**

### Task 2: Rust 网络核心（前向/反向/初始化）

**Files:**
- Create: src-tauri/src/nn/mod.rs
- Create: src-tauri/src/nn/network.rs
- Test: src-tauri/src/nn/network.rs 内联 #[cfg(test)]

**Interfaces:**
- Produces: `Network { arch: Arch, w1: Vec<f64>, b1: Vec<f64>, w2: Vec<f64>, b2: Vec<f64> }`
- Produces: `Network::new(arch: Arch, seed: u64) -> Network`（He 初始化）
- Produces: `Network::forward(&self, x: &[f64]) -> ForwardOut { z1, a1, z2, p }`
- Produces: `Network::backward(&self, x: &[f64], y: usize, f: &ForwardOut) -> Gradients`

- [ ] **Step 1: 写失败测试（含数值微分校验）**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_outputs_valid_softmax() {
        let net = Network::new(Arch { inputs: 169, hidden: 16, outputs: 10 }, 42);
        let x = vec![0.5; 169];
        let f = net.forward(&x);
        assert_eq!(f.p.len(), 10);
        let sum: f64 = f.p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn backward_matches_numerical_gradient() {
        let net = Network::new(Arch { inputs: 169, hidden: 16, outputs: 10 }, 7);
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
            let mut up = net.clone(); let mut dn = net.clone();
            match which { 0 => { up.w1[idx] += eps; dn.w1[idx] -= eps; }
                          1 => { up.w2[idx] += eps; dn.w2[idx] -= eps; }
                          _ => { up.b1[idx] += eps; dn.b1[idx] -= eps; } }
            let lp = up.forward(&x).p[y].ln();
            let ld = dn.forward(&x).p[y].ln();
            let num = (lp - ld) / (2.0 * eps);
            assert!((num - ana).abs() < 1e-6, "梯度不匹配: {} vs {}", num, ana);
        }
    }
}
```

- [ ] **Step 2: cargo test 确认失败**

Run: cd src-tauri && cargo test

- [ ] **Step 3: 实现 network.rs**

```rust
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Arch { pub inputs: usize, pub hidden: usize, pub outputs: usize }

#[derive(Clone, Serialize, Deserialize)]
pub struct Network { pub arch: Arch, pub w1: Vec<f64>, pub b1: Vec<f64>, pub w2: Vec<f64>, pub b2: Vec<f64> }

#[derive(Clone)]
pub struct ForwardOut { pub z1: Vec<f64>, pub a1: Vec<f64>, pub z2: Vec<f64>, pub p: Vec<f64> }

#[derive(Clone)]
pub struct Gradients { pub g_w1: Vec<f64>, pub g_b1: Vec<f64>, pub g_w2: Vec<f64>, pub g_b2: Vec<f64> }
```
（实现 new/forward/backward：LeakyReLU + Softmax + 交叉熵，权重 He 初始化，RNG 用手写 mulberry32）

- [ ] **Step 4: cargo test 确认通过**
- [ ] **Step 5: 提交 git commit -m "feat: rust network core"**

### Task 3: Rust 优化器（Adam + SGD）

**Files:**
- Create: src-tauri/src/nn/optimizer.rs
- Test: 内联 #[cfg(test)]

**Interfaces:**
- Consumes: Network/Gradients from Task 2
- Produces: `AdamState { m: Vec<f64>, v: Vec<f64>, t: u64 }`
- Produces: `AdamState::new(param_count: usize)`
- Produces: `adam_step(net: &mut Network, state: &mut AdamState, grads: &Gradients, lr: f64)`
- Produces: `sgd_step(net: &mut Network, grads: &Gradients, lr: f64)`

- [ ] **Step 1: 失败测试**：Adam 第一步 |Δw| ≈ lr（偏差校正后）；SGD Δw = lr*grad；100 步 loss 下降。
- [ ] **Step 2: cargo test 确认失败**
- [ ] **Step 3: 实现**：m/v 一阶二阶矩、t 步数、偏差校正；参数扁平化布局 w1+b1+w2+b2 拼接。
- [ ] **Step 4: cargo test 确认通过**
- [ ] **Step 5: 提交 git commit -m "feat: rust optimizers"**

### Task 4: 预处理（Rust + JS 双实现）

**Files:**
- Create: src-tauri/src/nn/preprocess.rs
- Create: src/nn/preprocess.js
- Test: 双方各写测试（cargo test / node --test）

**Interfaces:**
- Produces (Rust): `center_of_mass(img: &[u8], w: usize, h: usize) -> (f64, f64)`
- Produces (Rust): `preprocess(img: &[u8], w: usize, h: usize) -> Vec<f64>`（居中+降采样 13×13）
- Produces (JS): centerOfMass(img, w, h)、resampleTo13(img, w, h)、preprocess(img, w, h)

- [ ] **Step 1: 失败测试**（两语言同断言）：全零图像重心=画布中心；单亮点重心=亮点处；居中后降采样重心≈(6,6)；输出 0~1。
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现双份**（算法逐行对应：双线性采样 + 2×2 块平均）
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交 git commit -m "feat: dual-language preprocessing"**

### Task 5: JS 教学引擎（前向/反向/Adam）与双引擎一致性

**Files:**
- Create: src/nn/network.js
- Create: src/nn/optimizer.js
- Create: tests/fixtures/（交叉验证权重 JSON）
- Test: tests/network.test.js、tests/optimizer.test.js

**Interfaces:**
- Produces: createNetwork(arch, seed) → {arch, W1, b1, W2, b2}
- Produces: forward(net, x) → {z1, a1, z2, p}
- Produces: backward(net, x, y, f) → {gW1, gb1, gW2, gb2}
- Produces: createAdam() / adamStep(net, state, grads, lr) / sgdStep(net, grads, lr)

- [ ] **Step 1: 写失败测试**（含 100 样本 3 epoch loss 下降测试）
- [ ] **Step 2: node --test 确认失败**
- [ ] **Step 3: 实现**（与 Rust 逻辑严格对应：同 seed 生成同权重，mulberry32+Box-Muller）
- [ ] **Step 4: node --test 确认通过**
- [ ] **Step 4b: 双引擎一致性测试**：固定 seed=42、固定 20 样本，JS 训练 3 epoch 导出权重至 tests/fixtures/js-weights.json；Rust 同 seed 同样本训练 3 epoch，逐元素比较 |Δw| < 1e-9（cargo test 读取 fixture 文件）
- [ ] **Step 5: 提交 git commit -m "feat: js teaching engine with cross-validation"**

### Task 6: 工作区文件系统（Rust 命令）

**Files:**
- Create: src-tauri/src/workspace.rs
- Create: src-tauri/src/commands.rs
- Test: workspace 单元测试（tempdir）

**Interfaces:**
- Produces: `Workspace::open(path) / Workspace::create(path) / Workspace::load_manifests()`
- Produces commands: `workspace_open`、`workspace_create`、`list_samples(split)`、`add_sample(split, label, png_bytes, width, height, source)`、`remove_sample(id, split)`、`import_sample_dataset()`、`list_models`、`save_model`、`load_model`、`delete_model`
- PNG 编解码用 image crate

- [ ] **Step 1: 失败测试**：tempdir 创建工作区 → add_sample 写 PNG → list 读回 → manifest round-trip → remove 删除文件。
- [ ] **Step 2: cargo test 确认失败**
- [ ] **Step 3: 实现**（manifest 按 split 分文件；id 用 timestamp+counter；手写 260×260、MNIST 13×13）
- [ ] **Step 4: cargo test 确认通过**
- [ ] **Step 5: 提交 git commit -m "feat: workspace filesystem"**

### Task 7: 示例数据集生成与打包

**Files:**
- Create: scripts/generate-dataset.js
- Create: src-tauri/resources/sample-dataset.json
- Create: src-tauri/src/sample-data.rs（include_str! 嵌入）

**Interfaces:**
- Produces: JSON {version, train: [{id,label,pixels,source:"mnist-import"}], test: [...]}

- [ ] **Step 1: 实现 idx 解析 + 每类采样 100/30 + 居中降采样 13×13 + 2 位小数量化**
- [ ] **Step 2: 运行生成** Run: node scripts/generate-dataset.js
- [ ] **Step 3: Rust 端训练验收**（cargo test：include 资源 → 训练 15 epoch → acc > 0.9）
- [ ] **Step 4: 提交 git commit -m "feat: bundled sample dataset"**

### Task 8: 训练引擎（Rust command 层）

**Files:**
- Create: src-tauri/src/trainer.rs
- Test: 内联 #[cfg(test)]

**Interfaces:**
- Consumes: Network/Adam/Gradients/preprocess/sample-dataset
- Produces: `train_epochs(net, adam, samples, epochs, lr, on_epoch)`
- Produces: `evaluate(net, samples) -> { acc, per_class, confusion }`
- Produces commands: `train_bulk(model_name, epochs, lr, optimizer)`（异步、可取消）、`infer_13(x13: Vec<f64>)`（返回全部中间值）

- [ ] **Step 1: 失败测试**：100 样本 3 epoch loss 下降；confusion 10×10 对角线正确；train_bulk 取消后保留已完成 epoch。
- [ ] **Step 2: cargo test 确认失败**
- [ ] **Step 3: 实现**：每轮 shuffle（seed 固定）、每 epoch 回调；train_bulk 状态机 Mutex<TrainingState>。
- [ ] **Step 4: cargo test 确认通过（含 1000 样本 15 epoch acc>0.9）**
- [ ] **Step 5: 提交 git commit -m "feat: rust training engine"**

### Task 9: UI 骨架与手写画板

**Files:**
- Create: src/ui/drawing.js
- Modify: index.html、src/main.js、src/ui/app.css

**Interfaces:**
- Consumes: preprocess（JS）
- Produces: initDrawing(canvas, onStroke) → {clear, getPixels13}

- [ ] **Step 1: 三栏布局骨架**（左采集/中舞台/右训练，全中文，CSS 变量设计 token）
- [ ] **Step 2: 画板实现**（260×260，pointer 事件+lineJoin round+线宽 16；实时 13×13 网格预览）
- [ ] **Step 3: 验收**（手写"7"→13×13 显示笔画骨架；清空按钮；触屏兼容）
- [ ] **Step 4: 提交 git commit -m "feat: ui skeleton and drawing board"**

### Task 10: 神经网络舞台（核心可视化）

**Files:**
- Create: src/ui/stage.js
- Create: src/ui/particles.js

**Interfaces:**
- Consumes: forward（JS 引擎）、preprocess、tauri-bridge
- Produces: initStage(canvas, net) → {playInference(x), playTrainingStep(before, after), highlightNode(layer, idx)}

- [ ] **Step 1: 静态渲染**（169 输入按 13×13 像素墙、16 隐藏、10 输出；每隐藏节点默认画 |w| top-12 连线，粗细=|w|、蓝正红负）
- [ ] **Step 2: 推理动画**（粒子沿边流动，每层 300ms，粒子亮度=激活值；输出柱从均匀 10% 收敛到 p）
- [ ] **Step 3: 训练动画**（反向红色误差粒子 + 权重更新量脉冲 + before/after 概率对比条）
- [ ] **Step 4: 交互**（点击隐藏节点 → 169 权重热力图；prefers-reduced-motion 退化瞬达）
- [ ] **Step 5: 浏览器验收**（手写 7 → 逐层点亮 → 输出 7 柱条最高）
- [ ] **Step 6: 提交 git commit -m "feat: neural network stage with propagation animations"**

### Task 11: 训练控制面板 + 混淆矩阵 + 数据集管理

**Files:**
- Create: src/ui/panel.js
- Create: src/ui/dataset-viewer.js

**Interfaces:**
- Consumes: trainer commands、workspace commands、stage
- Produces: initPanel(root, {onTrainOne, onTrainEpoch, onEvaluate})、initDatasetViewer(root, workspace)

- [ ] **Step 1: 训练控制**（模型列表+新建/重命名/删除、优化器切换、lr 滑条、epoch 数、单样本"保存并训练"、批量训练（首轮动画+快进）、暂停/继续）
- [ ] **Step 2: 指标面板**（loss 曲线、总准确率、每类准确率条形、10×10 混淆矩阵热力表）
- [ ] **Step 3: 混淆矩阵交互**（点击格子 → 弹出样本原图列表 → 可删除错标）
- [ ] **Step 4: 数据集浏览**（训练/测试集网格、样本卡片 hover 翻面、一键导入示例数据集）
- [ ] **Step 5: 浏览器验收**（导入示例集 → 训练 15 epoch → acc > 90% 展示；Finder 查看 PNG 文件夹）
- [ ] **Step 6: 提交 git commit -m "feat: training panel, confusion matrix, dataset management"**

### Task 12: 教学内容层 + 端到端验收 + GitHub 发布

**Files:**
- Create: src/ui/lesson.js
- Modify: index.html、README.md
- Test: tests/e2e.test.js

**Interfaces:**
- Consumes: 全部
- Produces: initLesson(root, stageBus)

- [ ] **Step 1: 四阶段引导条**（像素化 → 前向 → 训练 → 泛化，高亮当前阶段，操作完成后自动推进）
- [ ] **Step 2: 端到端验收**（新建工作区 → 导入示例集 → 新建模型 → 训练 15 epoch → acc > 0.9 → 模型 round-trip）
- [ ] **Step 3: UI 走查清单**（空数据集、空模型、训练中按钮态、NaN 恢复、reduced-motion、768px 以下、触屏）
- [ ] **Step 4: gh repo create nn-anatomy --public --source=. --push**
- [ ] **Step 5: 提交 git commit -m "feat: lesson guide and e2e validation"**

## Self-Review 记录

- Spec 覆盖：Tauri 骨架（T1）、Rust 网络与优化器（T2/T3/T8）、预处理双实现（T4）、JS 教学引擎与交叉验证（T5）、工作区（T6）、示例数据集（T7）、画板（T9）、舞台（T10）、面板（T11）、教学引导与发布（T12）。
- 类型一致性：Rust Network 字段 w1/b1/w2/b2；JS createNetwork 返回 {arch, W1, b1, W2, b2}，映射在桥接层完成。
- 交叉验证：固定 seed 3 epoch 逐元素 < 1e-9，写入 T5 Step 4b。
