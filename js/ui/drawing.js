/**
 * 手写画板：260×260 绘图画布 + pointer 事件绘制 + 60×60 预处理。
 *
 * ES module，零 npm 依赖。绘图逻辑与取数逻辑分离：
 * - initDrawing(canvas, callbacks)：挂接 pointer 事件，
 *   返回 { clear, getPixels60, getPixels13, getLastPreprocessInfo }；
 * - getPixels60()：从画布读像素 → 灰度化 → js/preprocess.js 的 preprocess()（60×60）；
 * - callbacks.onStroke：每次落笔结束（pointerup）时调用；
 * - callbacks.onDraw：绘制中每次 pointermove 时调用（供实时推理节流使用）。
 *   为兼容旧调用，callbacks 也可直接传一个函数，视为 onStroke。
 *
 * 画布内部固定 260×260（backing store），CSS 可任意缩放显示，
 * 绘图坐标按 (canvas.width / rect.width) 比例换算，因此显示缩放不影响精度。
 */
import { preprocess, preprocessWithInfo } from '../preprocess.js';

/** 画板边长（像素）。 */
export const BOARD_SIZE = 260;
/** 画笔宽度（像素）。 */
export const LINE_WIDTH = 16;
/** 画笔颜色。 */
export const STROKE_COLOR = '#1c1917';

/**
 * 初始化手写画板。
 * @param {HTMLCanvasElement} canvas 已存在于 DOM 的画布（内部尺寸会被设为 260×260）。
 * @param {(() => void)|{ onStroke?: () => void, onDraw?: () => void }} [callbacks]
 *        绘制回调：onStroke 落笔结束时调用；onDraw 绘制中每次 move 时调用。
 *        直接传函数视为 { onStroke }（兼容旧调用）。
 * @returns {{ clear: () => void, getPixels60: () => Float64Array,
 *             getPixels13: () => Float64Array, getLastPreprocessInfo: () => object|null,
 *             canvas: HTMLCanvasElement }}
 */
export function initDrawing(canvas, callbacks) {
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error('initDrawing 需要一个 <canvas> 元素。');
  }
  const onStroke = typeof callbacks === 'function' ? callbacks : callbacks && callbacks.onStroke;
  const onDraw = typeof callbacks === 'function' ? undefined : callbacks && callbacks.onDraw;
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
    if (typeof onDraw === 'function') {
      try {
        onDraw();
      } catch (e) {
        console.error('onDraw 回调出错：', e);
      }
    }
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

  /** 最近一次 getPixels60() 的预处理中间信息 { cx, cy, dx, dy }（无则 null）。 */
  let lastInfo = null;

  /** 画布 → 墨=1 的灰度字节数组（白底黑字取反）。 */
  const readInkGray = () => {
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
    return gray;
  };

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
     * 当前画布 → 60×60 灰度（0~1），行优先 3600 维。
     * 同时记录本次预处理的重心/平移信息（供过程面板展示）。
     * @returns {Float64Array} 3600 个值，行优先。
     */
    getPixels60() {
      const gray = readInkGray();
      const info = preprocessWithInfo(gray, BOARD_SIZE, BOARD_SIZE, 60);
      lastInfo = { cx: info.cx, cy: info.cy, dx: info.dx, dy: info.dy };
      return info.x;
    },

    /**
     * 当前画布 → 13×13 灰度（0~1）。保留供旧测试；应用内统一用 60×60。
     * @returns {Float64Array} 169 个值，行优先。
     */
    getPixels13() {
      const gray = readInkGray();
      return preprocess(gray, BOARD_SIZE, BOARD_SIZE, 13);
    },

    /** 最近一次 getPixels60() 的预处理中间信息 { cx, cy, dx, dy }（无则 null）。 */
    getLastPreprocessInfo() {
      return lastInfo;
    },
  };
}
