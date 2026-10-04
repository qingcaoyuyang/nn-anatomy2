/**
 * node:fs 之上的 File System Access API 垫片（仅测试用）。
 *
 * 目标：让浏览器里才能拿到的 FileSystemDirectoryHandle 在 node 里可被
 * js/workspace.js 的 DirHandleBackend 原样使用，从而在 node 中验证
 * DirHandleBackend 的真实句柄导航逻辑（_navigate/_parentOf/list/write/remove/exists）。
 *
 * 实现的 API 面与 DirHandleBackend 使用到的严格对应：
 * - kind / name / getDirectoryHandle(name, {create}) / getFileHandle(name, {create})
 * - values()（async iterator，产出带 name/kind 的句柄）
 * - removeEntry(name, {recursive}) / getFile() / createWritable()
 * - 错误以 .name 区分：NotFoundError / TypeMismatchError
 *
 * 注意：这不是浏览器的完整实现，只覆盖被测代码路径。
 */
import { promises as fs } from 'node:fs';
import path from 'node:path';

class FsApiError extends Error {
  constructor(name, message) {
    super(message);
    this.name = name;
  }
}

export class NodeDirHandle {
  constructor(absPath, name) {
    this._path = absPath;
    this.name = name ?? path.basename(absPath);
    this.kind = 'directory';
  }

  async getDirectoryHandle(name, opts = {}) {
    const p = path.join(this._path, name);
    let st = null;
    try {
      st = await fs.stat(p);
    } catch (e) {
      if (e && e.code === 'ENOENT') {
        if (opts.create) {
          await fs.mkdir(p, { recursive: true });
          return new NodeDirHandle(p, name);
        }
        throw new FsApiError('NotFoundError', `目录不存在：${name}`);
      }
      throw e;
    }
    if (!st.isDirectory()) throw new FsApiError('TypeMismatchError', `不是目录：${name}`);
    return new NodeDirHandle(p, name);
  }

  async getFileHandle(name, opts = {}) {
    const p = path.join(this._path, name);
    let st = null;
    try {
      st = await fs.stat(p);
    } catch (e) {
      if (e && e.code === 'ENOENT') {
        if (opts.create) {
          await fs.writeFile(p, Buffer.alloc(0));
          return new NodeFileHandle(p, name);
        }
        throw new FsApiError('NotFoundError', `文件不存在：${name}`);
      }
      throw e;
    }
    if (!st.isFile()) throw new FsApiError('TypeMismatchError', `不是文件：${name}`);
    return new NodeFileHandle(p, name);
  }

  async *values() {
    const entries = await fs.readdir(this._path, { withFileTypes: true });
    for (const e of entries) {
      const p = path.join(this._path, e.name);
      yield e.isDirectory() ? new NodeDirHandle(p, e.name) : new NodeFileHandle(p, e.name);
    }
  }

  async removeEntry(name, opts = {}) {
    const p = path.join(this._path, name);
    try {
      await fs.rm(p, { recursive: !!opts.recursive, force: false });
    } catch (e) {
      if (e && e.code === 'ENOENT') throw new FsApiError('NotFoundError', `路径不存在：${name}`);
      throw e;
    }
  }
}

export class NodeFileHandle {
  constructor(absPath, name) {
    this._path = absPath;
    this.name = name ?? path.basename(absPath);
    this.kind = 'file';
  }

  async getFile() {
    let buf;
    try {
      buf = await fs.readFile(this._path);
    } catch (e) {
      if (e && e.code === 'ENOENT') throw new FsApiError('NotFoundError', `文件不存在：${this.name}`);
      throw e;
    }
    return new Blob([buf]);
  }

  async createWritable() {
    return new NodeWritable(this._path);
  }
}

/**
 * WritableFileStream 垫片：浏览器语义是打开即截断、write 追加、close 落盘。
 * 这里用内存缓冲 + close 时一次性写入，等价。
 */
class NodeWritable {
  constructor(absPath) {
    this._path = absPath;
    this._chunks = [];
    this._closed = false;
  }

  async write(data) {
    if (this._closed) throw new Error('Writable 已关闭。');
    if (data instanceof Blob) {
      this._chunks.push(Buffer.from(await data.arrayBuffer()));
    } else if (typeof data === 'string') {
      this._chunks.push(Buffer.from(data, 'utf8'));
    } else if (ArrayBuffer.isView(data)) {
      this._chunks.push(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
    } else {
      throw new Error(`不支持的写入类型：${typeof data}`);
    }
  }

  async close() {
    if (this._closed) return;
    this._closed = true;
    await fs.writeFile(this._path, Buffer.concat(this._chunks));
  }
}
