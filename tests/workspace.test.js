import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MemoryBackend } from '../js/workspace.js';
import { runWorkspaceSuite } from './workspace-suite.js';

test('工作区文件系统共享套件（MemoryBackend）', async () => {
  const result = await runWorkspaceSuite(
    () => Promise.resolve(new MemoryBackend()),
    assert,
  );
  assert.strictEqual(result.fail, 0,
    `${result.fail} 项失败:\n${result.errors.join('\n')}`);
  assert.ok(result.pass > 0, '套件应当至少跑过一个用例');
});
