/**
 * 应用入口（薄）：直接启动应用壳。
 */
import { boot } from './ui/app.js';

boot().catch((e) => console.error('[nn-anatomy2] 启动失败：', e));
