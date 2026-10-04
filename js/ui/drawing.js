/**
 * 手写画板：260×260 绘图画布 + pointer 事件绘制 + 13×13 预处理。
 *
 * ES module，零 npm 依赖。绘图逻辑与取数逻辑分离：
 * - initDrawing(canvas, onStroke)：挂接 pointer 事件，返回 { clear, getPixels13 }；
 * - getPixels13()：从画布读像素 → 灰度化 → js/preprocess.js 的 preprocess()。
 *
 * 画布内部固定 260×260（backing store），CSS 可任意缩放显示，
 * 绘图坐标按 (canvas.width / rect.width) 比例换算，因此显示缩放不影响精度。
 */
import { preprocess } from '../preprocess.js';

/** 画板边长（像素）。 */
export const BOARD_SIZE = 260;
/** 画笔宽度（像素）。 */
export const LINE_WIDTH = 16;
/** 画笔颜色。 */
export const STROKE_COLOR = '#1c1917';

/**
 * 初始化手写画板。
 * @param {HTMLCanvasElement} canvas 已存在于 DOM 的画布（内部尺寸会被设为 260×260）。
 * @param {() => void} [onStroke] 每次落笔结束（pointerup）时调用，用于刷新 13×13 预览等。
 * @returns {{ clear: () => void, getPixels13: () => Float64Array, canvas: HTMLCanvasElement }}
 */
export function initDrawing(canvas, onStroke) {
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error('initDrawing 需要一个 <canvas> 元素。');
  }
  canvas.width = BOARD_SIZE;
  canvas.height = BOARD_SIZE;

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('当前环境不支持 Canvas 2D。');

  // 白底：getImageData 灰度化时空白区域为 255 → 预处理归一化后为 0（纸面）。
  const paintBackground = () => {
    ctx.save();
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, BOARD_SIZE, BOARD_SIZE);
    ctx.restore();
  };
  paintBackground();

  ctx.lineWidth = LINE_WIDTH;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = STROKE_COLOR;

  let drawing = false;
  let lastX = 0;
  let lastY = 0;

  /** 事件坐标 → 画布像素坐标（考虑 CSS 缩放）。 */
  const toCanvasCoords = (event) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  };

  const onPointerDown = (event) => {
    event.preventDefault();
    // setPointerCapture：指针滑出画布后 move/up 仍能收到（触屏/触控笔必备）。
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch (e) {
      // 旧浏览器不支持时忽略，不影响基本绘制
    }
    drawing = true;
    const p = toCanvasCoords(event);
    lastX = p.x;
    lastY = p.y;
    // 点一下也留下一个圆点
    ctx.beginPath();
    ctx.arc(p.x, p.y, LINE_WIDTH / 2, 0, Math.PI * 2);
    ctx.fillStyle = STROKE_COLOR;
    ctx.fill();
  };

  const onPointerMove = (event) => {
    if (!drawing) return;
    event.preventDefault();
    const p = toCanvasCoords(event);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    lastX = p.x;
    lastY = p.y;
  };

  const endStroke = (event) => {
    if (!drawing) return;
    drawing = false;
    if (typeof onStroke === 'function') {
      try {
        onStroke();
      } catch (e) {
        // onStroke 是外部回调，出错不应打断画板状态
        console.error('onStroke 回调出错：', e);
      }
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', endStroke);
  canvas.addEventListener('pointercancel', endStroke);

  return {
    canvas,

    /** 清空画板（恢复白底）。 */
    clear() {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, BOARD_SIZE, BOARD_SIZE);
      ctx.restore();
      paintBackground();
    },

    /**
     * 当前画布 → 13×13 灰度（0~1）。
     * @returns {Float64Array} 169 个值，行优先。
     */
    getPixels13() {
      const imageData = ctx.getImageData(0, 0, BOARD_SIZE, BOARD_SIZE);
      const rgba = imageData.data;
      const gray = new Uint8Array(BOARD_SIZE * BOARD_SIZE);
      for (let i = 0; i < gray.length; i++) {
        const r = rgba[i * 4];
        const g = rgba[i * 4 + 1];
        const b = rgba[i * 4 + 2];
        // 白底黑字：255(白) → 0(纸面)，0(黑) → 1(墨迹)
        gray[i] = 255 - Math.round((r + g + b) / 3);
      }
      return preprocess(gray, BOARD_SIZE, BOARD_SIZE);
    },
  };
}
