/**
 * 教学引导条：四阶段 stepper（Task 12）。
 *
 * 四个阶段：像素化 → 前向传播 → 训练 → 泛化。
 * 推进只靠真实事件（app.js 调 notify 送入），不用定时器：
 *   drawn     ← 画板落笔且有墨迹
 *   inference ← 点"推理"跑完一次前向
 *   trained   ← 单步训练或批量训练完成
 *   evaluated ← 测试集评估完成（测试样本 > 0）
 *
 * 事件可以乱序到达：每个事件只标记自己的阶段完成，
 * 高亮永远指向"第一个未完成的阶段"。全部完成显示祝贺态。
 * 本体是静态 DOM，不播动画，reduced-motion 下无影响。
 */

/** @returns {{ notify(evt:string):void, reset():void }} */
export function initLesson(root, _api) {
  if (!root) throw new Error('initLesson 需要引导条容器。');

  const STAGES = [
    { id: 'drawn', title: '像素化', desc: '在左边的画板上写一个数字' },
    { id: 'inference', title: '前向传播', desc: '点「▶ 推理」，看信号流过网络' },
    { id: 'trained', title: '训练', desc: '点「保存并训练」，看网络修正错误' },
    { id: 'evaluated', title: '泛化', desc: '导入示例数据集，看测试集准确率' },
  ];

  root.innerHTML = `
    <ol class="lesson-steps" aria-label="教学引导">
      ${STAGES.map(
        (s, i) => `
        <li class="lesson-step" data-stage="${s.id}">
          <span class="lesson-dot" aria-hidden="true">${i + 1}</span>
          <span class="lesson-text">
            <span class="lesson-title">${s.title}</span>
            <span class="lesson-desc">${s.desc}</span>
          </span>
        </li>`,
      ).join('<li class="lesson-sep" aria-hidden="true">→</li>')}
    </ol>
    <p class="lesson-cheer hidden" id="lesson-cheer">🎉 四个阶段都体验过啦，继续自由探索吧！</p>
    <button class="btn btn-ghost btn-sm" id="lesson-skip" type="button">跳过引导</button>`;

  const stepEls = Array.from(root.querySelectorAll('.lesson-step'));
  const cheerEl = root.querySelector('#lesson-cheer');
  const skipBtn = root.querySelector('#lesson-skip');
  const done = new Set();

  function render() {
    const firstOpen = STAGES.findIndex((s) => !done.has(s.id));
    stepEls.forEach((el, i) => {
      const id = STAGES[i].id;
      el.classList.toggle('done', done.has(id));
      el.classList.toggle('active', i === firstOpen);
      const dot = el.querySelector('.lesson-dot');
      dot.textContent = done.has(id) ? '✓' : String(i + 1);
    });
    const allDone = done.size === STAGES.length;
    cheerEl.classList.toggle('hidden', !allDone);
    skipBtn.classList.toggle('hidden', allDone);
  }

  skipBtn.addEventListener('click', () => {
    root.classList.add('hidden');
  });

  render();

  return {
    /** 送入真实事件；未知事件忽略。 */
    notify(evt) {
      if (!STAGES.some((s) => s.id === evt)) return;
      if (done.has(evt)) return;
      done.add(evt);
      render();
    },
    /** 回到初始态（并重新显示引导条）。 */
    reset() {
      done.clear();
      root.classList.remove('hidden');
      render();
    },
  };
}
