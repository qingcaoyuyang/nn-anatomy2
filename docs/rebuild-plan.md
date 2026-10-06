# nn-anatomy2 60×60 重构实施计划

用户已批准（2026-10-06）：60×60 输入 + 32 隐藏 + 训练集扩到每类 300 张（已入库 3300 张 PNG）+
底部过程面板 + 教学流（空模型→10 样本→测试→全量训练→测试→实时手写）。

## 已完成
- [x] 数据集：`tools/build-dataset.py` → 60×60、train 每类 300（3000）、test 每类 30（300），commit 9e3c9c2
- [x] `js/dataset.js`：SAMPLE_COUNTS={train:300,test:30}，SAMPLE_SIZE=60，decode 3600 维
- [x] `js/preprocess.js`：参数化 `preprocess(img,w,h,outSize=60)`，新增 `preprocessWithInfo`（返回重心/平移）
- [x] `js/ui/drawing.js`：`getPixels60()` + `onDraw` 回调（实时推理用）+ `getLastPreprocessInfo()`
- [x] `js/ui/app.js`：ARCH={3600,32,10}，60×60 预览，实时推理节流接线，过程面板挂载点 `#panel-process`
- [x] node 单测 54/54（preprocess/dataset/decode 已更新）

## 接口约定（各模块必须遵守）

### ARCH
`{ inputs: 3600, hidden: 32, outputs: 10 }`

### 网络对象（不变）
`{ arch, w1: Float64Array(32*3600), b1: Float64Array(32), w2: Float64Array(10*32), b2: Float64Array(10) }`

### Trace（inferWithTrace 返回，不变）
`{ x: Float64Array(3600), z1: Float64Array(32), a1: Float64Array(32), z2: Float64Array(10), p: Float64Array(10) }`

### Stage API（js/ui/stage.js）
- `initStage(canvas, net)` → stage；`stage.setNet(net)`；`stage.render()`
- `stage.playInference(x: 3600数组)` → Promise（校验长度；播粒子动画）
- `stage.playTrainingStep(before, after)` → Promise；before/after = `{ trace, w1: 切片, w2: 切片, y }`
- `stage.highlightNode('hidden', idx)` / `stage.clearHighlight()`
- `stage.getLastTrace()` → 最近一次推理的 trace（过程面板用）
- 布局画布 680×440：左 6×6 分块（每块 10×10 像素，颜色=平均墨量）｜中 32 隐藏节点（2 列×16）｜右 10 输出概率柱
- 连线：input→hidden 每节点 top-8 |w|（共 256 条，从像素位置画）；hidden→output 32×10 全画
- 点击隐藏节点 → 左侧切该节点 60×60 权重热力图（蓝正/红负）；再点/点空白 → 回分块
- 点击分块 → 显示该块 100 像素小图 + Σ|w| 最大的 top-3 隐藏节点
- 必须保留：粒子系统、prefers-reduced-motion 退化、动画重入 guard、训练 before/after 对比态
- 诚实标注（画在左下）：“36 块只是帮人眼看清墨在哪；网络实际把 3600 像素一次性全连到每个隐藏节点”

### 过程面板（js/ui/process-panel.js，新建）
- `initProcessPanel(root)` → `{ showInference(trace, preInfo), showTraining(info), clear() }`
- `showInference(trace, preInfo)`：preInfo=`{cx,cy,dx,dy}`；渲染 ①原图→②重心→③平移→④60×60→⑤z₁=W₁x+b₁→⑥LeakyReLU→⑦z₂→⑧softmax，全部用真实数值
- `showTraining({before, after, y, lr, optimizer, epoch})`：loss=-ln(p[y])、输出误差、反向传播、梯度、Adam(lr/m/v/t)、|Δw| top 边、before/after 概率对比
- app.js 在 stage.playInference 后调 showInference；panel.js 在 trainSingleStep/batchTrain 后调 showTraining

### 权重视图（js/ui/weights-view.js，新建）
- `initWeightsView(api)` → `{ open() }`；modal 展示 W1（32 张 60×60 热力图）+ W2（10×32 矩阵）
- panel.js 模型区加「查看权重」按钮；空模型（随机初始化小值）打开即为教学第 0 步

### 实时推理（app.js 已接线）
- `#realtime-toggle`（默认开）→ pointermove 经 rAF 节流 → getPixels60 → forward → `#live-digit` + `#live-bars`（top-3）
- 落笔结束（pointerup）→ 完整 stage.playInference 粒子动画
- 性能：前向 ~115k MAC ≈ 0.2ms；预处理 ~5ms；每帧 <10ms，M4 无压力。重动画不参与实时循环。

## 待做
- [ ] stage.js 重写（分工 A）
- [ ] process-panel.js + weights-view.js（分工 B）
- [ ] 接线：panel.js 调过程面板；权重按钮；CSS（新增类）
- [ ] 讲稿 docs/lecture-script.md 重写到新流程
- [ ] E2E 重跑（目标 ~90%+）
  - node 基准（2026-10-06，seed23/runSeed1）：60×60→32→10，3000 训/300 测，
    15 轮 acc=0.9300（99.8s），30 轮 acc=0.9233（199.4s）→ 15 轮为甜点，E2E 阈值 ≥0.90
- [ ] UI walkthrough 更新
- [ ] Mac 包重打
