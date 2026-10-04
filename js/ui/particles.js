/**
 * 粒子系统：神经网络舞台的前向 / 反向传播动画共用。
 *
 * ES module，零 npm 依赖，无 DOM。舞台（js/ui/stage.js）每帧调用
 * update(dt) 推进、draw(ctx) 绘制。
 *
 * 粒子：{ x1,y1,x2,y2, t, speed, size, color, alpha }
 *   - 位置 = (x1,y1)→(x2,y2) 的线性插值，t 从 0→1，t>=1 自动回收；
 *   - speed：每秒走完的"边长比例"（如 3 表示约 1/3 秒走完一条边）；
 *   - color：'r,g,b' 字符串，绘制时拼成 rgba()；
 *   - 绘制用"双层圆"模拟辉光（刻意不用 shadowBlur：352 条边 + 数百粒子
 *     同屏时 shadowBlur 会让帧率崩掉）。
 */

const TAU = Math.PI * 2;

/** 默认粒子上限：前向两段 + 反向两段同时存在也不会超。 */
export const MAX_PARTICLES = 600;

/**
 * 创建一个粒子池。
 * @param {number} [max=MAX_PARTICLES] 粒子上限；满时挤掉最老的粒子。
 */
export function createParticleSystem(max = MAX_PARTICLES) {
  const particles = [];

  return {
    /** 当前存活粒子数。 */
    count() {
      return particles.length;
    },

    /**
     * 生成一个粒子。
     * @param {{x1:number,y1:number,x2:number,y2:number}} edge 运动的边（起止点）
     * @param {object} [opts] { t0=0, speed=3, size=2.2, color='255,160,60', alpha=0.9 }
     */
    spawn(edge, opts = {}) {
      if (particles.length >= max) particles.shift(); // 满了挤掉最老
      particles.push({
        x1: edge.x1,
        y1: edge.y1,
        x2: edge.x2,
        y2: edge.y2,
        t: opts.t0 || 0,
        speed: opts.speed || 3,
        size: opts.size || 2.2,
        color: opts.color || '255,160,60',
        alpha: opts.alpha == null ? 0.9 : opts.alpha,
      });
    },

    /** 推进 dt 秒（dt 为秒数）。 */
    update(dt) {
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.t += p.speed * dt;
        if (p.t >= 1) particles.splice(i, 1);
      }
    },

    /** 绘制全部存活粒子（ctx 已做 DPR 缩放，坐标为逻辑像素）。 */
    draw(ctx) {
      for (const p of particles) {
        const x = p.x1 + (p.x2 - p.x1) * p.t;
        const y = p.y1 + (p.y2 - p.y1) * p.t;
        // 尾端 15% 淡出，避免粒子在终点"闪没"
        const fade = p.t < 0.85 ? 1 : (1 - p.t) / 0.15;
        const a = p.alpha * fade;
        const r = p.size;
        // 外层光晕（低透明大圆）
        ctx.beginPath();
        ctx.arc(x, y, r * 2.2, 0, TAU);
        ctx.fillStyle = `rgba(${p.color},${(a * 0.22).toFixed(3)})`;
        ctx.fill();
        // 内核（高透明小圆）
        ctx.beginPath();
        ctx.arc(x, y, r, 0, TAU);
        ctx.fillStyle = `rgba(${p.color},${a.toFixed(3)})`;
        ctx.fill();
      }
    },

    /** 清空全部粒子。 */
    clear() {
      particles.length = 0;
    },
  };
}
