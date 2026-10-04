/**
 * mulberry32 伪随机数发生器 + Box-Muller 正态采样。
 * ES module，零 npm 依赖，无 DOM。
 *
 * # 确定性契约
 * - `createRng(seed)` 先把 seed 截断到 u32（`seed >>> 0`）。
 * - 单步为标准 mulberry32（Math.imul 语义），输出为无符号 u32。
 * - `nextF64()` = u32 输出 / 2^32，范围 [0, 1)。
 * - `nextNormal()` 用标准 Box-Muller：消耗一对连续均匀数（u1 先、u2 后），
 *   只取 z0 = sqrt(-2·ln(u1))·cos(2π·u2)，丢弃 z1（不缓存）。
 *   `u1 == 0` 的极端情况**有意不处理**（与旧 Rust 实现一致）：
 *   Math.log(0) = -Infinity 会自然传播，任何"修复"都会让 RNG 流分叉。
 */

/** mulberry32 单步：返回无符号 u32，state 就地更新。 */
export function createRng(seed) {
  let state = seed >>> 0;

  function nextU32() {
    let t = (state + 0x6D2B79F5) | 0;
    state = t;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t = (t ^ ((t + Math.imul(t ^ (t >>> 7), t | 61)) | 0)) | 0;
    return (t ^ (t >>> 14)) >>> 0;
  }

  function nextF64() {
    return nextU32() / 4294967296;
  }

  function nextNormal() {
    const u1 = nextF64();
    const u2 = nextF64();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  return { nextU32, nextF64, nextNormal };
}
