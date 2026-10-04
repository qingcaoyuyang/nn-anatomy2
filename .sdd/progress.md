# SDD ledger — plan: /home/hatch/workspace/nn-anatomy2/implementation-plan.md

## Pre-flight conflict scan (2026-10-04, controller)

| # | Conflict | Ruling |
|---|---|---|
| 1 | Plan paths reference /Users/juewei/... (Mac); user said dir "nn-anatomy2" in cloud, plan says "nn-anatomy/" | Ruling: use ~/workspace/nn-anatomy2 — user's explicit name wins. Cost if wrong: rename later, cheap. |
| 2 | Task 1: `cargo install tauri-cli` + `cargo tauri dev` cannot run on headless Linux VM (no display, no WebKitGTK) | Ruling: restructure Rust as `nn-core` lib crate (pure computation, no tauri dep) + `src-tauri` bin crate (Tauri shell). Verify nn-core with `cargo test`; Tauri GUI compile/run deferred to user's Mac. Skip tauri-cli install here. Cost if wrong: none for logic; GUI wiring verified on Mac. |
| 3 | Task 1 Step 4: `cargo tauri icon` needs tauri-cli | Ruling: generate placeholder 1024px icon via script now; regenerate with `cargo tauri icon` on Mac. |
| 4 | Task 12 Step 4: `gh repo create --public --push` — publish side effect outside worktree | Ruling: ASK USER before sleeping; default = do NOT publish. |
| 5 | Task 7 needs MNIST idx files | Ruling: download from https://storage.googleapis.com/cvdf-datasets/mnist/ (4 files). Fallback: alternate mirror. |
| 6 | Task 5 Step 4b: JS/Rust weight diff < 1e-9 needs bit-identical RNG + op order | Ruling: both engines must implement mulberry32 + Box-Muller identically (plan mandates); emphasized in Task 2/5 briefs. Test with fixed seed=42, 20 samples, 3 epochs. |
| 7 | Global constraint: fonts local (Baloo 2 + Noto Sans SC + mono) | Ruling: download woff2 (Baloo 2, Noto Sans SC Regular/Bold, mono=system ui-monospace fallback + bundled if small). |
| 8 | Tasks 9-12 "browser acceptance" | Ruling: verify with headless Chromium screenshots + stubbed window.__TAURI__ bridge; real Tauri WebView check on Mac. |

## Task 1: complete
- Task 1: complete (commits f3b688c..d8f58d4, review clean)
- Controller verified: `cargo test -p nn-core` 1/1 passing; icon.png is 1024x1024 RGB on disk.
- Task 1: minor (deferred): tauri.conf.json single-line minified, no trailing newline — pretty-print later.
- Task 1: minor (deferred): src-tauri/Cargo.toml has unused `tempdir` optional dependency — remove or wire when first used.
- Task 1: minor (deferred): nn-core/Cargo.toml empty [dependencies] section — remove until Task 2 adds serde.
- Task 1: minor (deferred): empty dirs src/nn, tests, scripts not tracked — add .gitkeep or let Task 2+ populate.
- Ruling: commit Cargo.lock for the binary workspace before final review (standard practice for applications) — cost if wrong: none, easily reverted.
- Ruling: src-tauri `cargo check` deferred to user's Mac (WebKitGTK missing on this VM) — recorded in Task 1 report.

## PIVOT 裁决（2026-10-04 深夜，用户决策：按建议切换到纯浏览器 + File System Access API）

- Tauri 2 方案停止。`nn-core/`、`src-tauri/`、`Cargo.toml`、`Cargo.lock`、`src/main.js`、`src/tauri-bridge.js` 已 `git mv` 到 `archive/tauri-v1/`（历史保留，附 NOTE.md）；`target/` 已删除（gitignored 构建产物）。
- 保留沿用：`js/preprocess.js`（Task 4 JS 实现）、`css/app.css`、`index.html`（路径已更新）。
- 作废：旧 Task 5（JS↔Rust 双引擎一致性 <1e-9）——单引擎无需交叉验证；旧 Task 1 的"Mac 上 cargo check Tauri"、旧 Task 12 的 GitHub 发布默认不做（等用户确认）。
- 新计划：implementation-plan.md 已重写为 Task 5–12（纯浏览器）。旧 brief 只有 1–4，无需归档。
- 验证能力升级：已安装无头 Chromium（`~/workspace/tools/chrome-headless-shell-linux64/`，v154），冒烟测试通过：`showDirectoryPicker` 可用、OPFS 可用、`isSecureContext=true`（file:// 下亦然）。
- 运行方式：静态文件 + 本地 server（secure context）；要求 Chromium 内核浏览器。

## Task 12: complete (FINAL REVIEW: SHIP)
- Task 12: complete (commit 37bdab2) — lesson guide + E2E + README + serve.command.
- Controller verified: 52/52 node tests; E2E 重跑 pass=8 fail=0 acc=0.9033；walkthrough 8/8；截图目检通过。
- 修：导入/删除后左栏状态行过期（9c9ff26）；README 截图占位补实图（4268c47）。
- Final review verdict: **SHIP**（无 Critical/阻塞性 Important；4 条 Minor 不阻塞）。报告：.superpowers/sdd/implementation-plan/final-review.md。
- 全部分支干净。GitHub 发布未执行，等用户明确批准。
- 真机待用户确认：260×260 手写全链路、真实目录持久化、原生 rAF 流畅度、serve.command 双击。

## Task 11: complete (review APPROVED, fixes applied)
- Task 11: complete (commit a8c8d20) — training panel + confusion matrix + dataset management.
- **极性约定（load-bearing，已落实）**：网络输入统一"墨=1"。手写 PNG（黑字白纸）`x = 1 − gray/255` 再走 `preprocess()`；MNIST PNG（白字黑底）`x = gray/255`（13×13 直接，不取反）。
- Controller verified: 51/51 node tests (dataset-decode.test.js: handwritten 260×260 →墨=1→preprocess 全链路在 node 覆盖）; smoke CDP 真机时间 15/15（×4）；面板/modal 截图目检通过。
- 260×260 PNG→createImageBitmap 端到端解码在本环境挂死（AGENTS.md 已记），真机需补测。
- Reviewer verdict: Approved. Fixed: pausedState 绑定模型名（切模型丢弃暂停态）+ 单步/批量互斥 + 标签校验（commit 08f2661）。Minor 剩余 3（runEvaluation 双 forward、augmented 幽灵分支、argmaxP 重复）不阻塞。
- 1300 张全量导入性能、真实目录持久化、原生 rAF 流畅度 → 真机/Task 12。

## Task 10: complete
- Task 10: complete (commit 7f287d1, review APPROVED)
- Controller verified: 46/46 node tests; smoke {"pass":15,"fail":0}; inference + training screenshots reviewed, legible.
- Review follow-ups: stage JSDoc 补快照拷贝要求 + 状态语义文档。

## Task 9: complete
- Task 9: complete (commit 303bfb3, review APPROVED after fix round)
- Controller verified: 46/46 node tests; smoke page {"pass":14,"fail":0}; desktop + guide screenshots reviewed clean.
- Review follow-ups: 恢复按钮改 onclick 赋值（防重复挂载）+ 每次重入先解禁；boot() 加 catch；状态行 id 截短显示。

## Task 8: complete
- Task 8: complete (commit 260596b, review APPROVED)
- Controller verified: 46/46 green.
- Review follow-up (commit 8099d63): trainEpochs 对非整数 epochs 做 floor（之前会虚报 epochsDone）。

## Task 7: complete
- Task 7: complete (commit a25e83b, review APPROVED)
- Controller verified: 33/33 tests; 1300 PNGs on disk, per-class counts correct; PIL spot-checks 13×13 'L'.
- Review follow-up: build-dataset.py 改用 round-half-up（与注释"四舍五入"一致）+ idx 图片/标签跨文件计数断言；重跑后 PNG 逐字节一致（说明原数据不受影响）。

## Task 6: complete
- Task 6: complete (commit 0c81c19, review APPROVED)
- Controller verified: `node --test tests/*.test.js` → 28/28 green; browser PNG codec page → {"pass":1,"fail":0}.
- Review follow-ups (commit 6e7f0d4): saveModel JSON pretty-print (spec §3.2 可读性); id 加随机后缀防多实例碰撞 + 套件正则同步; workspace-opfs.html 补 IndexedDB 最近工作区用例（随真机 Chrome 跑）。
- 缺口（诚实记录）: OPFS 真机验证 + IndexedDB 用例只能在用户 Mac 的真实 Chrome 跑，云端 headless-shell 的 storage 子系统不可用。

## Task 5: complete
- Task 5: complete (commit b00cfe9, review APPROVED — reviewer did independent line-by-line math verification: sign convention, mulberry32 canonical algorithm re-implemented in Python, Adam bias correction, flat layout)
- Controller verified: `node --test tests/*.test.js` → 26/26 passing.
- Review follow-up (commit c4be873): added `ADAM_DEFAULT_LR = 0.001` export; fixed rng test comment wording.
- 流程备注：Node 24 下 `node --test tests/`（裸目录）报 MODULE_NOT_FOUND，后续任务验收一律用 `node --test tests/*.test.js` glob 形式。

## Task 4: complete
- Task 4: complete (commit ccb1d0b, review APPROVED)
- Controller verified: Rust 14/14 + JS 5/5 passing; cross-check max|Δ|=0 bit-identical.
- Review follow-up (commit b4a6ad8): added anti-aliasing regression guard to tests/preprocess.test.js — asserts the single-pixel pipeline output contains exactly one positive element (the (6,6) assertion alone would pass vacuously on all-zeros via the m=0 fallback). All 5 tests green.
- Also fixed stale `../src/nn/preprocess.js` imports in both test files after the pivot reorg (moved to `js/`).
- Note: only the JS half (`js/preprocess.js`) carries forward; Rust half archived.

## Task 2: complete
- Task 2: complete (commits d8f58d4..6ed5083, review clean after fix round 1/5 — all findings addressed, no new breakage)
- Ruling (load-bearing, Task 3/5 must follow): Task 2's `backward` returns gradients of LOG-LIKELIHOOD ln(p[y]) (dz2 = onehot − p), NOT cross-entropy loss gradients. Optimizer steps must ASCEND: param += lr·g (equivalent to CE descent). Documented in nn-core/src/nn/network.rs header + backward docstring.
- Ruling: RNG bit-identity contract scoped — exact bit-identity guaranteed only through next_f64 (mulberry32 int stream + u32→f64/2^32); ln/cos may differ 1–2 ulp across Rust libm vs V8; Task 5 must compare with tolerance (< 1e-9 budget covers it). u1==0 intentionally unhandled in next_normal — JS must mirror, not "fix".

## Task 3: complete
- Task 3: complete (commits 6ed5083..8d371a3, review clean)
- Controller verified: `cargo test -p nn-core` 9/9 passing.
- Task 3: minor (deferred): no DEFAULT_LR constant — add `pub const DEFAULT_LR: f64 = 0.001` with Adam constants later.
- Task 3: minor (deferred): AdamState fields fully pub — consider pub(crate) later.
- Ruling: sign convention reaffirmed — optimizers ASCEND log-likelihood (param += lr·g); Task 5 JS and Task 8 Rust must use the same convention.
