/**
 * 图像预处理：重心居中 + 降采样到 13×13 + 归一化（Rust 与 JS 双实现，逐位一致）。
 * ES module，零 npm 依赖。
 *
 * # 精确算法定义（Rust / JS 共用，两实现逐行对应）
 *
 * 输入：灰度字节数组（0–255），尺寸 w×h，行优先（索引 `y*w + x`）。
 * 输出：13×13 = 169 个 f64，范围 [0, 1]。
 *
 * ## Step 1 — 重心（center of mass）
 * 以灰度值为权重，x、y 为整数像素坐标（0..w-1 / 0..h-1），按 y=0..h、x=0..w
 * 的行优先顺序累加：
 * ```text
 * m  = Σ v
 * sx = Σ x·v
 * sy = Σ y·v
 * ```
 * 若 m = 0（全零图像）：(cx, cy) = ((w−1)/2, (h−1)/2)（画布中心）。
 * 否则：cx = sx / m，cy = sy / m。
 * `centerOfMass` 同时接受 Uint8Array（0–255）和 Float64Array/普通数组输入。
 *
 * ## Step 2 — 平移居中（translate to center）
 * ```text
 * dx = (w−1)/2 − cx
 * dy = (h−1)/2 − cy
 * T(x, y) = bilinearSample(I, x − dx, y − dy)    （x ∈ 0..w，y ∈ 0..h）
 * ```
 * 越界采样返回 0。
 *
 * ## 双线性采样（bilinearSample：仅含 +、−、×，严格按括号顺序求值）
 * ```text
 * x0 = floor(sx),  y0 = floor(sy)
 * fx = sx − x0,    fy = sy − y0
 * v00 = I(x0, y0), v10 = I(x0+1, y0), v01 = I(x0, y0+1), v11 = I(x0+1, y0+1)
 *     （任一坐标越界则该采样点取 0）
 * out = (v00·(1−fx) + v10·fx)·(1−fy) + (v01·(1−fx) + v11·fx)·fy
 * ```
 *
 * ## Step 3 — 降采样到 13×13（块平均，质量守恒）
 * 将平移后的图像 T（w×h）划分为 13×13 个互不重叠的矩形块；
 * 输出像素 (ox, oy)（ox、oy ∈ 0..13）为其块内像素的算术平均：
 * ```text
 * x0 = floor(ox·w/13),  x1 = floor((ox+1)·w/13)
 * y0 = floor(oy·h/13),  y1 = floor((oy+1)·h/13)
 * O[ox, oy] = (Σ T[y·w+x]) / ((x1−x0)·(y1−y0))，其中 x ∈ [x0, x1)，y ∈ [y0, y1)
 * ```
 * 求和按 y = y0..y1−1 外层、x = x0..x1−1 内层的行优先顺序左结合；
 * 除数为块内像素个数的精确 f64 值；空块（w<13 或 h<13 时可能出现）输出 0.0。
 * 当 w = h = 260 时，每块恰为 20×20（400 像素）。
 * 块之间无重叠、无遗漏（x1(ox) = x0(ox+1)），总质量守恒。
 *
 * ## 为什么不用“双线性逐点重采样到 26×26 再 2×2 平均”
 * 双线性重采样是逐点采样：260→26 时每 10 像素才取一个采样点，
 * 2 像素宽的笔画或单像素点可能正好落在采样点之间而被完全丢弃（混叠）；
 * 此时“居中后重心 ≈ (6,6)”的测试会因输出全零而被 m=0 兜底虚假通过。
 * 块平均对块内全部像素求平均，单像素点必定落入其所在块，
 * 测试真实有效，且更适合手写数字这种高频细节输入。
 *
 * ## Step 4 — 归一化
 * ```text
 * out[i] = O[i] / 255.0
 * ```
 *
 * # 跨语言一致性契约（Task 4 双实现必读）
 * - 全程仅使用 f64 的 +、−、×、/ 与 floor：IEEE-754 正确舍入，
 *   Rust 与 V8（JS）逐位一致；u8→f64 转换精确（u8 < 2^53）。
 * - 禁止使用 sqrt、exp、ln、三角函数等超越函数（本管线不需要）。
 * - 求和一律按文档顺序左结合；本文件与 `nn-core/src/nn/preprocess.rs` 逐行对应。
 * - 交叉验证：`tests/preprocess-crosscheck.test.js` 读取 Rust 测试生成的
 *   `tests/fixtures/preprocess-crosscheck.json`，逐元素 |Δ| < 1e-12。
 */

/** 灰度图重心（像素坐标系）。全零图像返回画布中心 ((w−1)/2, (h−1)/2)。
 *  接受 Uint8Array（0–255）或 Float64Array/普通数组输入。 */
export function centerOfMass(img, w, h) {
  if (img.length !== w * h) throw new Error('centerOfMass: img.length 必须等于 w*h');
  return centerOfMassF64(Float64Array.from(img), w, h);
}

/** 居中 + 块平均到 13×13（未归一化，值域 [0, 255]）。 */
export function resampleTo13(img, w, h) {
  if (img.length !== w * h) throw new Error('resampleTo13: img.length 必须等于 w*h');
  const f = Float64Array.from(img);
  const [cx, cy] = centerOfMassF64(f, w, h);
  const t = translateToCenter(f, w, h, cx, cy);
  return blockAverageTo13(t, w, h);
}

/** 完整预处理管线：居中 + 降采样到 13×13 + 归一化到 [0, 1]，返回 169 个 f64。 */
export function preprocess(img, w, h) {
  const o = resampleTo13(img, w, h);
  const out = new Float64Array(169);
  for (let i = 0; i < 169; i++) out[i] = o[i] / 255;
  return out;
}

/** f64 图像的重心（内部用；与 Rust 的 center_of_mass_f64 逐行对应）。 */
function centerOfMassF64(img, w, h) {
  let m = 0;
  let sx = 0;
  let sy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = img[y * w + x];
      m += v;
      sx += x * v;
      sy += y * v;
    }
  }
  if (m === 0) return [(w - 1) / 2, (h - 1) / 2];
  return [sx / m, sy / m];
}

/** 双线性采样：严格按头注释公式与括号顺序求值；越界采样点取 0。 */
function bilinearSample(img, w, h, sx, sy) {
  const x0 = Math.floor(sx);
  const y0 = Math.floor(sy);
  const fx = sx - x0;
  const fy = sy - y0;
  const at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : img[y * w + x]);
  const v00 = at(x0, y0);
  const v10 = at(x0 + 1, y0);
  const v01 = at(x0, y0 + 1);
  const v11 = at(x0 + 1, y0 + 1);
  return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
}

/** 平移居中：T(x, y) = bilinearSample(I, x − dx, y − dy)。 */
function translateToCenter(img, w, h, cx, cy) {
  const dx = (w - 1) / 2 - cx;
  const dy = (h - 1) / 2 - cy;
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[y * w + x] = bilinearSample(img, w, h, x - dx, y - dy);
    }
  }
  return out;
}

/** 块平均到 13×13：块边界 x0 = floor(ox·w/13)、x1 = floor((ox+1)·w/13)
 *  （y 同理）；块内行优先左结合求和，除以块内像素个数；空块输出 0.0。 */
function blockAverageTo13(t, w, h) {
  const out = new Float64Array(13 * 13);
  for (let oy = 0; oy < 13; oy++) {
    for (let ox = 0; ox < 13; ox++) {
      const x0 = Math.floor((ox * w) / 13);
      const x1 = Math.floor(((ox + 1) * w) / 13);
      const y0 = Math.floor((oy * h) / 13);
      const y1 = Math.floor(((oy + 1) * h) / 13);
      let sum = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          sum += t[y * w + x];
        }
      }
      const count = (x1 - x0) * (y1 - y0);
      out[oy * 13 + ox] = count === 0 ? 0 : sum / count;
    }
  }
  return out;
}
