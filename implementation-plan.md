# nn-anatomy2 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 构建纯浏览器神经网络解剖教室，手写数字识别全程可视化，数据以 PNG/JSON 文件落盘到用户选择的真实文件夹。

**Architecture（2026-10-04  pivoted，见下方 PIVOT 记录）:** 纯浏览器应用，零后端：ES Modules + Canvas 2D + File System Access API。单 JS 计算引擎（网络/优化器/训练一体，169→16→10 小网络 JS 几百毫秒跑完 1000 样本×15 epoch，无需 Rust 批量引擎）。工作区 = 用户通过目录选择器授权的真实文件夹（PNG 数据集、`.nnmodel.json` 模型、manifest），可直接拷贝传播。存储层抽象为 Backend 接口：生产用真实目录 handle，测试用 OPFS（`navigator.storage.getDirectory()`，无头 Chromium 无需手势即可用）。

**Tech Stack:** JavaScript ES Modules、Canvas 2D、CSS、File System Access API、node --test + 无头 Chromium（截图/E2E）。

**运行方式：** 静态文件，需 http(s) 或 localhost（secure context）。Mac 上双击 `tools/serve.command` 或 `python3 -m http.server` 后用 Chrome 打开。file:// 下 `isSecureContext` 为 true 可降级运行，但 ES module 在 file:// 下受 CORS 限制，故以本地 server 为标准方式。

**浏览器要求：** Chromium 内核（Chrome/Edge）。Firefox 不支持 `showDirectoryPicker`。

## Global Constraints

- 网络结构固定：169 输入 → 16 隐藏 → 10 输出，LeakyReLU(0.01)，Softmax + 交叉熵。
- 优化器默认 Adam（lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8），训练界面可切 SGD。
- 符号约定（load-bearing，沿用 Tauri 版 Task 2 裁决）：`backward` 返回对数似然 ln(p[y]) 的梯度（dz2 = onehot − p），优化器做梯度**上升** `param += lr·g`（等价于交叉熵下降）。
- RNG：mulberry32，seed 截断到 u32；He 初始化 scale=√(2/fan_in)；Box-Muller 生成正态；抽取顺序 w1 行主序 → b1 零 → w2 行主序 → b2 零。
- 预处理管线：260×260 画板 → 重心居中 → 质心保持的块平均降采样 13×13 → 0~1 灰度（`js/preprocess.js` 已实现并经 Rust↔JS 交叉验证 max|Δ|=0）。
- 工作区目录结构、manifest 字段、模型文件命名 `.nnmodel.json` 按 spec 第 3 节执行。
- 界面语言：全中文；桌面优先（投屏 1024px+），平板（触屏）兼容，768px 以下单列。
- 设计系统：主色 #F97316 暖橙、背景 #FFF7ED 米白、深灰文字、单一强调色、单一圆角系统（8px）、全页面单一浅色主题。
- 数值（像素值、概率、loss、准确率）一律等宽字体。
- 前端零 npm 运行时依赖、无 CDN；字体本地打包（Baloo 2 + Noto Sans SC woff2 + 等宽）。
- 代码与文档全部位于 nn-anatomy2/ 单目录；`archive/tauri-v1/` 为已停止的 Tauri 方案归档。

## PIVOT 记录（2026-10-04 深夜，用户决策）

原方案 Tauri 2（Rust 双引擎）已停止，原因：
1. 169→16→10 网络太小，JS 单引擎足够快，Rust 批量引擎 + 双语言 1e-9 交叉验证是过度设计；
2. 纯浏览器零安装，教室 Chrome 即开即投屏，无需 Rust/Xcode/签名；
3. 云端可用无头 Chromium 全链路验证（含截图），Tauri GUI 在云端验证不了。

- Task 1（骨架）：完成，其 `index.html`/`css/app.css` 保留，Tauri 桥接部分已归档。
- Task 2/3（Rust 网络核心/优化器）：完成并测试全绿，已归档到 `archive/tauri-v1/`；算法文档（符号约定、RNG 规范）迁移到 JS 实现。
- Task 4（预处理）：Rust 实现归档；**JS 实现 `js/preprocess.js` 保留沿用**（评审中）。
- 旧的"JS↔Rust 固定种子 3 epoch 权重误差 < 1e-9"一致性约束作废（单引擎无需交叉验证）。

---

### Task 5: JS 引擎核心（RNG + 网络 + 优化器）

**Files:**
- Create: js/rng.js
- Create: js/network.js
- Create: js/optimizer.js
- Test: tests/rng.test.js、tests/network.test.js、tests/optimizer.test.js

**Interfaces:**
- Produces: `createRng(seed)` → mulberry32（seed 截断 u32），`nextF64()`，`nextNormal()`
- Produces: `createNetwork(arch, seed)` → {arch, w1, b1, w2, b2}（He 初始化，抽取顺序 w1→b1→w2→b2）
- Produces: `forward(net, x)` → {z1, a1, z2, p}
- Produces: `backward(net, x, y, f)` → 对数似然梯度 {gW1, gb1, gW2, gb2}
- Produces: `createAdam(paramCount)` / `adamStep(net, state, grads, lr)` / `sgdStep(net, grads, lr)`（梯度上升）

- [ ] **Step 1: 写失败测试**：softmax 和为 1；数值微分校验 backward（W1/W2/b1 各抽一点，eps=1e-5，误差<1e-6）；He 初始化方差量级正确；Adam 第一步 |Δw|≈lr（偏差校正后）；100 样本 3 epoch 对数似然上升。
- [ ] **Step 2: node --test 确认失败**
- [ ] **Step 3: 实现**（严格按 Global Constraints 的符号约定与 RNG 规范；与归档的 Rust 实现逻辑对应）
- [ ] **Step 4: node --test 确认通过**
- [ ] **Step 5: 提交 git commit -m "feat: js engine core (rng, network, optimizer)"**

### Task 6: 工作区文件系统（File System Access API）

**Files:**
- Create: js/workspace.js
- Test: tests/workspace.test.js（内存 Mock Backend，node 可跑）

**Interfaces:**
- Produces: `Backend` 抽象（方法：listDir/getFile/putFile/remove/entryKind），`DirHandleBackend`（生产）与 `MemoryBackend`（测试）
- Produces: `Workspace.open(dirHandle)` / `Workspace.createNew(dirHandle)` / `requestWorkspaceDir()`（showDirectoryPicker）
- Produces: `listSamples(split)`、`addSample(split, label, pngBlob, width, height, source)`、`removeSample(id, split)`、`listModels()`、`saveModel(name, net, meta)`、`loadModel(name)`、`deleteModel(name)`
- Produces: 最近工作区持久化（IndexedDB 存 dirHandle，启动时 queryPermission/requestPermission 恢复授权）
- PNG 编解码走 Canvas：读用 `createImageBitmap`，写用 canvas `toBlob`

- [ ] **Step 1: 写失败测试**（MemoryBackend）：open/create → addSample 写 PNG → list 读回 → manifest round-trip → remove 删文件 → model save/load round-trip。
- [ ] **Step 2: node --test 确认失败**
- [ ] **Step 3: 实现**（manifest 按 split 分文件；id=timestamp+counter；目录结构按 spec 第 3 节）
- [ ] **Step 4: node --test 确认通过**
- [ ] **Step 5: 无头 Chromium 集成验证**：OPFS Backend 跑同样用例（脚本见 tests/browser/），确认真实 API 路径可用
- [ ] **Step 6: 提交 git commit -m "feat: workspace filesystem on File System Access API"**

### Task 7: 示例数据集生成与打包

**Files:**
- Create: tools/build-dataset.py
- Create: data/sample-digits/train/{0..9}/*.png、data/sample-digits/test/{0..9}/*.png（每类 train 100 / test 30，共 1300 张）
- Create: js/dataset.js（`importSampleDataset(workspace)`：把 data/ 下 PNG 拷贝进工作区）

**Interfaces:**
- Consumes: MNIST idx 文件（https://storage.googleapis.com/cvdf-datasets/mnist/，4 文件）
- Produces: 13×13 灰度 PNG（导入时在 Python 端做重心居中 + 面积平均降采样；手写样本仍存 260×260 原图，加载时走 js/preprocess.js 标准管线）

- [ ] **Step 1: 实现 tools/build-dataset.py**（PIL：解析 idx → 每类采样 → 写 PNG；固定采样 seed 保证可复现）
- [ ] **Step 2: 运行生成** Run: python3 tools/build-dataset.py；抽查 PNG 可读且标签分布正确
- [ ] **Step 3: 实现 js/dataset.js**（fetch 打包 PNG → workspace.addSample；source="mnist-import"）
- [ ] **Step 4: 提交 git commit -m "feat: bundled sample dataset (1300 PNGs)"**

### Task 8: 训练引擎

**Files:**
- Create: js/trainer.js
- Test: tests/trainer.test.js

**Interfaces:**
- Consumes: network/optimizer/preprocess/workspace
- Produces: `trainEpochs(net, optState, samples, {epochs, lr, optimizer, seed, onEpoch})`（可取消：onEpoch 返回 false 即停，保留已完成 epoch）
- Produces: `evaluate(net, samples)` → {acc, perClass, confusion[10][10]}
- Produces: `inferWithTrace(net, x13)` → 全部中间值（供舞台动画）

- [ ] **Step 1: 写失败测试**：100 样本 3 epoch 对数似然上升；confusion 10×10 对角线；取消后保留已完成 epoch；evaluate 在固定小样本上 acc 可复现。
- [ ] **Step 2: node --test 确认失败**
- [ ] **Step 3: 实现**（每轮 shuffle 用独立 seed 流；默认 Adam，可切 SGD）
- [ ] **Step 4: node --test 确认通过**
- [ ] **Step 5: 提交 git commit -m "feat: training engine"**

### Task 9: UI 骨架与手写画板

**Files:**
- Create: js/ui/app.js（应用壳：首次启动引导选工作区文件夹）
- Create: js/ui/drawing.js
- Modify: index.html、css/app.css

**Interfaces:**
- Consumes: preprocess（JS）、workspace
- Produces: `initDrawing(canvas, onStroke)` → {clear, getPixels13}
- Produces: 三栏布局（左采集/中舞台/右训练），首次启动引导用户选择或新建工作区文件夹

- [ ] **Step 1: 三栏布局骨架**（全中文，CSS 变量设计 token，工作区引导流程：选文件夹 → 授权 → 恢复最近）
- [ ] **Step 2: 画板实现**（260×260，pointer 事件+lineJoin round+线宽 16；实时 13×13 网格预览）
- [ ] **Step 3: 无头 Chromium 验收截图**（布局三栏可见；画板可绘制；清空按钮；13×13 预览）
- [ ] **Step 4: 提交 git commit -m "feat: ui skeleton and drawing board"**

### Task 10: 神经网络舞台（核心可视化）

**Files:**
- Create: js/ui/stage.js
- Create: js/ui/particles.js

**Interfaces:**
- Consumes: network（forward）、preprocess、trainer.inferWithTrace
- Produces: `initStage(canvas, net)` → {playInference(x), playTrainingStep(before, after), highlightNode(layer, idx)}

- [ ] **Step 1: 静态渲染**（169 输入按 13×13 像素墙、16 隐藏、10 输出；每隐藏节点默认画 |w| top-12 连线，粗细=|w|、蓝正红负）
- [ ] **Step 2: 推理动画**（粒子沿边流动，每层 300ms，粒子亮度=激活值；输出柱从均匀 10% 收敛到 p）
- [ ] **Step 3: 训练动画**（反向红色误差粒子 + 权重更新量脉冲 + before/after 概率对比条）
- [ ] **Step 4: 交互**（点击隐藏节点 → 169 权重热力图；prefers-reduced-motion 退化瞬达）
- [ ] **Step 5: 无头 Chromium 验收截图**（手写 7 → 逐层点亮 → 输出 7 柱条最高）
- [ ] **Step 6: 提交 git commit -m "feat: neural network stage with propagation animations"**

### Task 11: 训练控制面板 + 混淆矩阵 + 数据集管理

**Files:**
- Create: js/ui/panel.js
- Create: js/ui/dataset-viewer.js

**Interfaces:**
- Consumes: trainer、workspace、stage
- Produces: `initPanel(root, {onTrainOne, onTrainEpoch, onEvaluate})`、`initDatasetViewer(root, workspace)`

- [ ] **Step 1: 训练控制**（模型列表+新建/重命名/删除、优化器切换、lr 滑条、epoch 数、单样本"保存并训练"、批量训练（首轮动画+快进）、暂停/继续）
- [ ] **Step 2: 指标面板**（loss 曲线、总准确率、每类准确率条形、10×10 混淆矩阵热力表）
- [ ] **Step 3: 混淆矩阵交互**（点击格子 → 弹出样本原图列表 → 可删除错标）
- [ ] **Step 4: 数据集浏览**（训练/测试集网格、样本卡片 hover 翻面、一键导入示例数据集）
- [ ] **Step 5: 无头 Chromium 验收**（导入示例集 → 训练 15 epoch → acc > 90% 展示；截图确认）
- [ ] **Step 6: 提交 git commit -m "feat: training panel, confusion matrix, dataset management"**

### Task 12: 教学内容层 + 端到端验收 + README

**Files:**
- Create: js/ui/lesson.js
- Create: tests/e2e.test.js（无头 Chromium 驱动）
- Create: README.md
- Create: tools/serve.command（Mac 双击启动本地 server）

**Interfaces:**
- Consumes: 全部
- Produces: `initLesson(root, stageBus)`

- [ ] **Step 1: 四阶段引导条**（像素化 → 前向 → 训练 → 泛化，高亮当前阶段，操作完成后自动推进）
- [ ] **Step 2: 端到端验收**（无头 Chromium + OPFS 工作区：导入示例集 → 新建模型 → 训练 15 epoch → acc > 0.9 → 模型 save/load round-trip；全程截图）
- [ ] **Step 3: UI 走查清单**（空数据集、空模型、训练中按钮态、NaN 恢复、reduced-motion、768px 以下、触屏 pointer）
- [ ] **Step 4: README**（中文：运行方式、浏览器要求、工作区说明、架构、开发命令）
- [ ] **Step 5: 提交 git commit -m "feat: lesson guide, e2e validation and README"**

> 注：原计划的 `gh repo create --public --push`（GitHub 发布）默认不执行，等用户明确确认后再做。

## Self-Review 记录

- Spec 覆盖：引擎核心（T5）、工作区文件系统（T6）、示例数据集（T7）、训练引擎（T8）、画板（T9）、舞台（T10）、面板（T11）、教学引导与 E2E（T12）。
- 单 JS 引擎：数值一致性约束已随双引擎一起作废；可复现性由固定 seed + RNG 规范保证。
- 工作区：真实目录 handle（生产）/ OPFS（测试）双 Backend，同一套逻辑。
- 验收：node --test（逻辑）+ 无头 Chromium（UI 截图 + E2E）。
