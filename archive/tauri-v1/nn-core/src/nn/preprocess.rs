//! 图像预处理：重心居中 + 降采样到 13×13 + 归一化（Rust 与 JS 双实现，逐位一致）。
//!
//! # 精确算法定义（Rust / JS 共用，两实现逐行对应）
//!
//! 输入：灰度字节数组（0–255），尺寸 w×h，行优先（索引 `y*w + x`）。
//! 输出：13×13 = 169 个 f64，范围 [0, 1]。
//!
//! ## Step 1 — 重心（center of mass）
//! 以灰度值为权重，x、y 为整数像素坐标（0..w-1 / 0..h-1），按 y=0..h、x=0..w
//! 的行优先顺序累加：
//! ```text
//! m  = Σ v
//! sx = Σ x·v
//! sy = Σ y·v
//! ```
//! 若 m = 0（全零图像）：(cx, cy) = ((w−1)/2, (h−1)/2)（画布中心）。
//! 否则：cx = sx / m，cy = sy / m。
//!
//! ## Step 2 — 平移居中（translate to center）
//! ```text
//! dx = (w−1)/2 − cx
//! dy = (h−1)/2 − cy
//! T(x, y) = bilinear_sample(I, x − dx, y − dy)    （x ∈ 0..w，y ∈ 0..h）
//! ```
//! 越界采样返回 0。
//!
//! ## 双线性采样（bilinear_sample：仅含 +、−、×，严格按括号顺序求值）
//! ```text
//! x0 = floor(sx),  y0 = floor(sy)
//! fx = sx − x0,    fy = sy − y0
//! v00 = I(x0, y0), v10 = I(x0+1, y0), v01 = I(x0, y0+1), v11 = I(x0+1, y0+1)
//!     （任一坐标越界则该采样点取 0）
//! out = (v00·(1−fx) + v10·fx)·(1−fy) + (v01·(1−fx) + v11·fx)·fy
//! ```
//!
//! ## Step 3 — 降采样到 13×13（块平均，质量守恒）
//! 将平移后的图像 T（w×h）划分为 13×13 个互不重叠的矩形块；
//! 输出像素 (ox, oy)（ox、oy ∈ 0..13）为其块内像素的算术平均：
//! ```text
//! x0 = floor(ox·w/13),  x1 = floor((ox+1)·w/13)
//! y0 = floor(oy·h/13),  y1 = floor((oy+1)·h/13)
//! O[ox, oy] = (Σ T[y·w+x]) / ((x1−x0)·(y1−y0))，其中 x ∈ [x0, x1)，y ∈ [y0, y1)
//! ```
//! 求和按 y = y0..y1−1 外层、x = x0..x1−1 内层的行优先顺序左结合；
//! 除数为块内像素个数的精确 f64 值；空块（w<13 或 h<13 时可能出现）输出 0.0。
//! 当 w = h = 260 时，每块恰为 20×20（400 像素）。
//! 块之间无重叠、无遗漏（x1(ox) = x0(ox+1)），总质量守恒。
//!
//! ## 为什么不用“双线性逐点重采样到 26×26 再 2×2 平均”
//! 双线性重采样是逐点采样：260→26 时每 10 像素才取一个采样点，
//! 2 像素宽的笔画或单像素点可能正好落在采样点之间而被完全丢弃（混叠）；
//! 此时“居中后重心 ≈ (6,6)”的测试会因输出全零而被 m=0 兜底虚假通过。
//! 块平均对块内全部像素求平均，单像素点必定落入其所在块，
//! 测试真实有效，且更适合手写数字这种高频细节输入。
//!
//! ## Step 4 — 归一化
//! ```text
//! out[i] = O[i] / 255.0
//! ```
//!
//! # 跨语言一致性契约（Task 4 双实现必读）
//! - 全程仅使用 f64 的 +、−、×、/ 与 floor：IEEE-754 正确舍入，
//!   Rust 与 V8（JS）逐位一致；u8→f64 转换精确（u8 < 2^53）。
//! - 禁止使用 sqrt、exp、ln、三角函数等超越函数（本管线不需要）。
//! - 求和一律按文档顺序左结合；JS（`src/nn/preprocess.js`）与本文件逐行对应。
//! - 交叉验证：`tests/preprocess-crosscheck.test.js` 读取本文件测试生成的
//!   `tests/fixtures/preprocess-crosscheck.json`，逐元素 |Δ| < 1e-12。

/// 灰度图重心（像素坐标系）。全零图像返回画布中心 ((w−1)/2, (h−1)/2)。
pub fn center_of_mass(img: &[u8], w: usize, h: usize) -> (f64, f64) {
    assert_eq!(img.len(), w * h, "center_of_mass: img.len() 必须等于 w*h");
    let f: Vec<f64> = img.iter().map(|&v| v as f64).collect();
    center_of_mass_f64(&f, w, h)
}

/// 居中 + 块平均到 13×13（未归一化，值域 [0, 255]）。
pub fn resample_to_13(img: &[u8], w: usize, h: usize) -> Vec<f64> {
    assert_eq!(img.len(), w * h, "resample_to_13: img.len() 必须等于 w*h");
    let f: Vec<f64> = img.iter().map(|&v| v as f64).collect();
    let (cx, cy) = center_of_mass_f64(&f, w, h);
    let t = translate_to_center(&f, w, h, cx, cy);
    block_average_to_13(&t, w, h)
}

/// 完整预处理管线：居中 + 降采样到 13×13 + 归一化到 [0, 1]，返回 169 个 f64。
pub fn preprocess(img: &[u8], w: usize, h: usize) -> Vec<f64> {
    resample_to_13(img, w, h)
        .iter()
        .map(|&v| v / 255.0)
        .collect()
}

/// f64 图像的重心（内部/测试用；`center_of_mass` 先转 f64 再调它）。
/// 行优先 y=0..h、x=0..w 累加；m = 0 时返回画布中心。
fn center_of_mass_f64(img: &[f64], w: usize, h: usize) -> (f64, f64) {
    let mut m = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let v = img[y * w + x];
            m += v;
            sx += x as f64 * v;
            sy += y as f64 * v;
        }
    }
    if m == 0.0 {
        ((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0)
    } else {
        (sx / m, sy / m)
    }
}

/// 双线性采样：严格按头注释公式与括号顺序求值；越界采样点取 0。
fn bilinear_sample(img: &[f64], w: usize, h: usize, sx: f64, sy: f64) -> f64 {
    let x0 = sx.floor() as isize;
    let y0 = sy.floor() as isize;
    let fx = sx - x0 as f64;
    let fy = sy - y0 as f64;
    let w_i = w as isize;
    let h_i = h as isize;
    let at = |x: isize, y: isize| -> f64 {
        if x < 0 || y < 0 || x >= w_i || y >= h_i {
            0.0
        } else {
            img[(y * w_i + x) as usize]
        }
    };
    let v00 = at(x0, y0);
    let v10 = at(x0 + 1, y0);
    let v01 = at(x0, y0 + 1);
    let v11 = at(x0 + 1, y0 + 1);
    (v00 * (1.0 - fx) + v10 * fx) * (1.0 - fy) + (v01 * (1.0 - fx) + v11 * fx) * fy
}

/// 平移居中：T(x, y) = bilinear_sample(I, x − dx, y − dy)。
fn translate_to_center(img: &[f64], w: usize, h: usize, cx: f64, cy: f64) -> Vec<f64> {
    let dx = (w as f64 - 1.0) / 2.0 - cx;
    let dy = (h as f64 - 1.0) / 2.0 - cy;
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = bilinear_sample(img, w, h, x as f64 - dx, y as f64 - dy);
        }
    }
    out
}

/// 块平均到 13×13：块边界 x0 = floor(ox·w/13)、x1 = floor((ox+1)·w/13)
///（y 同理）；块内行优先左结合求和，除以块内像素个数；空块输出 0.0。
fn block_average_to_13(t: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0; 13 * 13];
    for oy in 0..13 {
        for ox in 0..13 {
            let x0 = ((ox * w) as f64 / 13.0).floor() as usize;
            let x1 = (((ox + 1) * w) as f64 / 13.0).floor() as usize;
            let y0 = ((oy * h) as f64 / 13.0).floor() as usize;
            let y1 = (((oy + 1) * h) as f64 / 13.0).floor() as usize;
            let mut sum = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += t[y * w + x];
                }
            }
            let count = ((x1 - x0) * (y1 - y0)) as f64;
            out[oy * 13 + ox] = if count == 0.0 { 0.0 } else { sum / count };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 260;
    const H: usize = 260;

    fn zeros() -> Vec<u8> {
        vec![0u8; W * H]
    }

    fn single_pixel() -> Vec<u8> {
        let mut img = zeros();
        img[201 * W + 37] = 255;
        img
    }

    /// 对角线渐变：v(x,y) = (x+y)*255/(w+h−2)（整数除法，与 JS 测试同公式）。
    fn diagonal_gradient() -> Vec<u8> {
        (0..H)
            .flat_map(|y| (0..W).map(move |x| ((x + y) * 255 / (W + H - 2)) as u8))
            .collect()
    }

    /// 全零图像 → 重心 = 画布中心。
    #[test]
    fn zeros_center_of_mass_is_canvas_center() {
        assert_eq!(center_of_mass(&zeros(), W, H), (129.5, 129.5));
    }

    /// 单亮点 → 重心 = 亮点坐标（精确相等）。
    #[test]
    fn single_pixel_center_of_mass_is_pixel() {
        assert_eq!(center_of_mass(&single_pixel(), W, H), (37.0, 201.0));
    }

    /// 居中 + 降采样后，13×13 网格上的重心 ≈ (6, 6)。
    #[test]
    fn centered_downsampled_center_of_mass_near_6_6() {
        let out = resample_to_13(&single_pixel(), W, H);
        assert_eq!(out.len(), 169);
        let (cx, cy) = center_of_mass_f64(&out, 13, 13);
        assert!(
            (cx - 6.0).abs() < 1e-6 && (cy - 6.0).abs() < 1e-6,
            "重心应 ≈ (6,6)，实际 ({cx}, {cy})"
        );
    }

    /// 输出全部落在 [0, 1]，且非常零图像有正输出。
    #[test]
    fn outputs_in_unit_interval() {
        let out = preprocess(&diagonal_gradient(), W, H);
        assert_eq!(out.len(), 169);
        assert!(
            out.iter().all(|&v| (0.0..=1.0).contains(&v)),
            "输出必须全部落在 [0, 1]"
        );
        assert!(out.iter().any(|&v| v > 0.0), "渐变图应有正输出");
    }

    /// 交叉验证固件：把 3 张合成图的 preprocess 输出写入
    /// `tests/fixtures/preprocess-crosscheck.json`，供 JS 测试逐元素对比
    /// （|Δ| < 1e-12）。f64 用最短往返字符串序列化，JSON.parse 可逐位还原。
    #[test]
    fn write_crosscheck_fixture() {
        let cases: [(&str, Vec<u8>); 3] = [
            ("zeros", zeros()),
            ("single_pixel", single_pixel()),
            ("diagonal_gradient", diagonal_gradient()),
        ];
        let mut s = String::from("{\"w\":260,\"h\":260,\"cases\":[");
        for (i, (name, img)) in cases.iter().enumerate() {
            let out = preprocess(img, W, H);
            assert_eq!(out.len(), 169);
            assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("{{\"name\":\"{name}\",\"output\":["));
            for (j, v) in out.iter().enumerate() {
                if j > 0 {
                    s.push(',');
                }
                s.push_str(&format!("{v}"));
            }
            s.push_str("]}");
        }
        s.push_str("]}");
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures");
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(format!("{dir}/preprocess-crosscheck.json"), s).unwrap();
    }
}
