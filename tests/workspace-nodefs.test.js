/**
 * DirHandleBackend 的真实句柄逻辑验证（node）。
 *
 * 用 tests/helpers/node-fs-shim.js 把 node:fs 包装成 FileSystemDirectoryHandle，
 * 让**真正的** DirHandleBackend（生产代码，未经修改）跑共享测试套件，
 * 验证 _navigate / _parentOf / list / writeFile / remove / exists 的句柄导航逻辑。
 * （浏览器 OPFS 在此 headless 环境不可用，见 task-6 评审记录；此测试是等价替代。）
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { DirHandleBackend } from '../js/workspace.js';
import { runWorkspaceSuite } from './workspace-suite.js';
import { NodeDirHandle } from './helpers/node-fs-shim.js';

test('工作区文件系统共享套件（DirHandleBackend + node:fs 真实目录）', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'nnws-'));
  let n = 0;
  try {
    const result = await runWorkspaceSuite(async () => {
      const sub = path.join(root, `case-${n++}`);
      await mkdir(sub, { recursive: true });
      return new DirHandleBackend(new NodeDirHandle(sub));
    }, assert);
    assert.strictEqual(
      result.fail,
      0,
      `${result.fail} 项失败:\n${result.errors.join('\n')}`,
    );
    assert.ok(result.pass > 0, '套件应当至少跑过一个用例');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
