/**
 * 工作区文件系统：File System Access API 之上的工作区抽象。
 *
 * 纯浏览器架构：工作区 = 用户授权的真实文件夹，结构遵循 design-spec.md 第 3 节：
 *   workspace.json / datasets/{train,test}/{0..9}/{id}.png + manifest.json / models/*.nnmodel.json / exports/
 *
 * # Backend 抽象
 * 所有 IO 都经过 Backend 接口（方法均为 async），路径为 `/` 分隔的相对路径：
 * - DirHandleBackend：包装 FileSystemDirectoryHandle。生产环境传入用户选择的目录，
 *   测试环境传入 OPFS 根（`navigator.storage.getDirectory()`）——两者是同一个类。
 * - MemoryBackend：Map 实现，供 node 测试使用。
 *
 * # 说明
 * ES module，零依赖。DOM/Canvas 相关函数（readSamplePixels、canvasToPngBlob）
 * 做了特性检测，node 环境下调用会抛中文错误，绝不影响纯逻辑测试。
 */

/** 支持的数据集划分。 */
export const SPLITS = ['train', 'test'];
/** 样本来源取值。 */
export const SOURCES = ['handwritten', 'mnist-import', 'augmented'];
/** workspace.json 的版本号。 */
export const WORKSPACE_VERSION = 1;

/** 把 `/` 分隔的相对路径切成段数组；空路径返回 []。 */
function splitPath(path) {
  if (typeof path !== 'string') throw new Error(`路径必须为字符串：${path}`);
  return path.split('/').filter((s) => s.length > 0);
}

/**
 * DirHandleBackend：基于 FileSystemDirectoryHandle 的 Backend。
 * 内部按段逐级 navigate（getDirectoryHandle / getFileHandle），create 按需创建。
 */
export class DirHandleBackend {
  /** @param {FileSystemDirectoryHandle} handle 目录句柄（用户目录或 OPFS 根）。 */
  constructor(handle) {
    if (!handle || handle.kind !== 'directory' || typeof handle.getFileHandle !== 'function') {
      throw new Error('DirHandleBackend 需要一个 FileSystemDirectoryHandle。');
    }
    this.handle = handle;
  }

  /** 逐级进入目录；create 为 true 时逐级创建。 */
  async _navigate(segments, create) {
    let dir = this.handle;
    for (const seg of segments) {
      dir = await dir.getDirectoryHandle(seg, { create });
    }
    return dir;
  }

  /** 返回 { dir, name }：path 的父目录句柄与末段名。 */
  async _parentOf(path, create) {
    const segments = splitPath(path);
    const name = segments.pop();
    if (name === undefined) throw new Error('不能对工作区根目录执行该操作。');
    const dir = segments.length > 0 ? await this._navigate(segments, create) : this.handle;
    return { dir, name };
  }

  /** 递归创建目录。 */
  async mkdir(path) {
    await this._navigate(splitPath(path), true);
  }

  /** 列出直接子项，按名称排序：[{ name, kind: 'file' | 'directory' }]。 */
  async list(path) {
    let dir;
    try {
      dir = await this._navigate(splitPath(path), false);
    } catch (e) {
      if (e && e.name === 'NotFoundError') throw new Error(`目录不存在：${path}`);
      throw e;
    }
    const entries = [];
    // 注意：values() 产出的是句柄本身（FileSystemHandle），不是 [name, handle] 对；
    // [name, handle] 对要用 entries()。句柄自带 .name 与 .kind。
    for await (const handle of dir.values()) {
      entries.push({ name: handle.name, kind: handle.kind });
    }
    entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    return entries;
  }

  /** 读取文件，返回 Blob。 */
  async readFile(path) {
    try {
      const { dir, name } = await this._parentOf(path, false);
      const fileHandle = await dir.getFileHandle(name);
      return await fileHandle.getFile();
    } catch (e) {
      if (e && e.name === 'NotFoundError') throw new Error(`文件不存在：${path}`);
      throw e;
    }
  }

  /** 写入文件；data 为 Blob 或 string（string 按 UTF-8 编码）。父目录自动创建。 */
  async writeFile(path, data) {
    const { dir, name } = await this._parentOf(path, true);
    const fileHandle = await dir.getFileHandle(name, { create: true });
    const writable = await fileHandle.createWritable();
    try {
      const blob = data instanceof Blob
        ? data
        : new Blob([data], { type: 'application/json;charset=utf-8' });
      await writable.write(blob);
    } finally {
      await writable.close();
    }
  }

  /** 删除文件或目录（递归）。 */
  async remove(path) {
    try {
      const { dir, name } = await this._parentOf(path, false);
      await dir.removeEntry(name, { recursive: true });
    } catch (e) {
      if (e && e.name === 'NotFoundError') throw new Error(`路径不存在：${path}`);
      throw e;
    }
  }

  /** 路径是否存在（文件或目录均可）。 */
  async exists(path) {
    const segments = splitPath(path);
    if (segments.length === 0) return true;
    let dir;
    let name;
    try {
      ({ dir, name } = await this._parentOf(path, false));
    } catch (e) {
      if (e && e.name === 'NotFoundError') return false;
      throw e;
    }
    // 先后尝试文件与目录：TypeMismatchError 表示"存在但是另一种类型"，继续试另一种。
    const lookup = async (kind) => {
      try {
        if (kind === 'file') await dir.getFileHandle(name);
        else await dir.getDirectoryHandle(name);
        return true;
      } catch (e) {
        if (e && (e.name === 'NotFoundError' || e.name === 'TypeMismatchError')) return false;
        throw e;
      }
    };
    if (await lookup('file')) return true;
    return lookup('directory');
  }
}

/**
 * MemoryBackend：基于 Map 的 Backend，供 node 测试使用。
 * 行为与 DirHandleBackend 对齐（错误信息为中文）。
 */
export class MemoryBackend {
  constructor() {
    /** @type {Map<string, { kind: 'file' | 'directory', data: Blob | null }>} */
    this._entries = new Map();
  }

  _norm(path) {
    return splitPath(path).join('/');
  }

  async mkdir(path) {
    const segments = splitPath(path);
    let acc = '';
    for (const seg of segments) {
      acc = acc === '' ? seg : `${acc}/${seg}`;
      this._entries.set(acc, { kind: 'directory', data: null });
    }
  }

  async list(path) {
    const prefix = this._norm(path);
    if (prefix !== '' && this._entries.get(prefix)?.kind !== 'directory') {
      throw new Error(`目录不存在：${path}`);
    }
    const out = [];
    for (const [key, val] of this._entries) {
      if (prefix === '') {
        if (!key.includes('/')) out.push({ name: key, kind: val.kind });
      } else if (key.startsWith(`${prefix}/`)) {
        const rest = key.slice(prefix.length + 1);
        if (!rest.includes('/')) out.push({ name: rest, kind: val.kind });
      }
    }
    out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    return out;
  }

  async readFile(path) {
    const entry = this._entries.get(this._norm(path));
    if (!entry || entry.kind !== 'file') throw new Error(`文件不存在：${path}`);
    return entry.data;
  }

  async writeFile(path, data) {
    const norm = this._norm(path);
    if (norm === '') throw new Error('不能写入工作区根路径。');
    const slash = norm.lastIndexOf('/');
    await this.mkdir(slash === -1 ? '' : norm.slice(0, slash));
    const blob = data instanceof Blob
      ? data
      : new Blob([data], { type: 'application/json;charset=utf-8' });
    this._entries.set(norm, { kind: 'file', data: blob });
  }

  async remove(path) {
    const norm = this._norm(path);
    if (norm === '') throw new Error('不能删除工作区根目录。');
    if (!this._entries.has(norm)) throw new Error(`路径不存在：${path}`);
    for (const key of [...this._entries.keys()]) {
      if (key === norm || key.startsWith(`${norm}/`)) this._entries.delete(key);
    }
  }

  async exists(path) {
    const norm = this._norm(path);
    if (norm === '') return true;
    return this._entries.has(norm);
  }
}

const FILENAME_FORBIDDEN_RE = /[/\\:*?"<>|]/g;

/**
 * 模型名 → 安全文件名（去 `/\:*?"<>|`）。过滤后为空或为 `.`/`..` 时抛错。
 * 注意：返回的只是文件名体，不含扩展名。
 */
export function sanitizeFileName(name) {
  if (typeof name !== 'string') throw new Error('名称必须为字符串。');
  const safe = name.replace(FILENAME_FORBIDDEN_RE, '_').trim();
  if (safe === '' || safe === '.' || safe === '..') {
    throw new Error(`名称不合法（过滤非法字符后为空）：${name}`);
  }
  return safe;
}

/**
 * 工作区：datasets / models 的高层读写，manifest 内存缓存 + 全量写回。
 *
 * 构造接受 Backend；`Workspace.create(handle)` / `Workspace.open(handle)`
 * 也可直接接受 FileSystemDirectoryHandle（内部自动包装为 DirHandleBackend）。
 */
export class Workspace {
  /** @param {DirHandleBackend | MemoryBackend} backend */
  constructor(backend) {
    if (!backend || typeof backend.readFile !== 'function') {
      throw new Error('Workspace 需要一个 Backend（DirHandleBackend 或 MemoryBackend）。');
    }
    this.backend = backend;
    this.manifests = { train: [], test: [] };
    this._idCounter = 0;
  }

  static _isDirHandle(x) {
    return !!x && x.kind === 'directory' && typeof x.getFileHandle === 'function';
  }

  static _toBackend(handleOrBackend) {
    return Workspace._isDirHandle(handleOrBackend)
      ? new DirHandleBackend(handleOrBackend)
      : handleOrBackend;
  }

  /** 初始化一个新工作区：建目录、写 workspace.json 与空 manifest。 */
  static async create(handleOrBackend) {
    const ws = new Workspace(Workspace._toBackend(handleOrBackend));
    await ws.initialize();
    return ws;
  }

  /** 打开已有工作区：校验 workspace.json 并加载 manifests。 */
  static async open(handleOrBackend) {
    const ws = new Workspace(Workspace._toBackend(handleOrBackend));
    await ws.loadFromDisk();
    return ws;
  }

  /** 初始化目录结构与元文件（已存在时不覆盖 manifests）。 */
  async initialize() {
    for (const split of SPLITS) {
      for (let d = 0; d <= 9; d++) {
        await this.backend.mkdir(`datasets/${split}/${d}`);
      }
    }
    await this.backend.mkdir('models');
    await this.backend.mkdir('exports');
    await this.backend.writeFile('workspace.json', JSON.stringify({
      version: WORKSPACE_VERSION,
      createdAt: new Date().toISOString(),
      app: 'nn-anatomy2',
    }, null, 2));
    for (const split of SPLITS) {
      if (await this.backend.exists(`datasets/${split}/manifest.json`)) {
        this.manifests[split] = await this._readManifest(split);
      } else {
        await this._saveManifest(split);
      }
    }
  }

  /** 从磁盘加载 workspace.json 与 manifests；不是有效工作区时抛中文错。 */
  async loadFromDisk() {
    let meta;
    try {
      const blob = await this.backend.readFile('workspace.json');
      meta = JSON.parse(await blob.text());
    } catch (e) {
      throw new Error('所选文件夹不是有效的工作区：缺少或损坏的 workspace.json。');
    }
    if (!meta || meta.version !== WORKSPACE_VERSION) {
      throw new Error(`不支持的工作区版本：${meta && meta.version}（本应用仅支持版本 ${WORKSPACE_VERSION}）。`);
    }
    for (const split of SPLITS) {
      this.manifests[split] = await this._readManifest(split);
    }
  }

  async _readManifest(split) {
    const blob = await this.backend.readFile(`datasets/${split}/manifest.json`);
    const arr = JSON.parse(await blob.text());
    if (!Array.isArray(arr)) {
      throw new Error(`manifest.json 格式错误：datasets/${split}/manifest.json`);
    }
    return arr;
  }

  async _saveManifest(split) {
    await this.backend.writeFile(
      `datasets/${split}/manifest.json`,
      JSON.stringify(this.manifests[split], null, 2),
    );
  }

  _checkSplit(split) {
    if (!SPLITS.includes(split)) {
      throw new Error(`未知的数据集划分：${split}（应为 train 或 test）。`);
    }
  }

  _newId() {
    // 时间戳 + 实例计数器 + 随机后缀：同一毫秒内多实例并发 add 也不碰撞。
    return `${Date.now()}-${this._idCounter++}-${Math.random().toString(36).slice(2, 8)}`;
  }

  /** 返回某划分的样本条目（拷贝）：[{ id, label, file, source, createdAt, ... }]。 */
  listSamples(split) {
    this._checkSplit(split);
    return this.manifests[split].map((e) => ({ ...e }));
  }

  /**
   * 添加样本：写 PNG 文件并追加 manifest 条目，返回 id。
   * @param {Blob} pngBlob PNG 数据
   * @param {{ width?: number, height?: number, source: string }} opts
   */
  async addSample(split, label, pngBlob, { width, height, source } = {}) {
    this._checkSplit(split);
    if (!/^[0-9]$/.test(String(label))) {
      throw new Error(`标签必须为 0-9 的单个数字：${label}`);
    }
    if (!(pngBlob instanceof Blob)) throw new Error('样本数据必须为 PNG Blob。');
    if (!SOURCES.includes(source)) {
      throw new Error(`未知的样本来源：${source}（应为 ${SOURCES.join(' / ')}）。`);
    }
    const id = this._newId();
    const file = `${id}.png`;
    await this.backend.writeFile(`datasets/${split}/${label}/${file}`, pngBlob);
    const entry = {
      id,
      label: String(label),
      file,
      source,
      createdAt: new Date().toISOString(),
    };
    if (Number.isFinite(width) && Number.isFinite(height)) {
      entry.width = width;
      entry.height = height;
    }
    this.manifests[split].push(entry);
    await this._saveManifest(split);
    return id;
  }

  /** 删除样本：删 PNG 文件并移除 manifest 条目。 */
  async removeSample(split, id) {
    this._checkSplit(split);
    const idx = this.manifests[split].findIndex((e) => e.id === id);
    if (idx === -1) throw new Error(`样本不存在：${id}`);
    const entry = this.manifests[split][idx];
    await this.backend.remove(`datasets/${split}/${entry.label}/${entry.file}`);
    this.manifests[split].splice(idx, 1);
    await this._saveManifest(split);
  }

  /**
   * 读取样本像素（仅浏览器）：createImageBitmap + canvas 解码 PNG。
   * @returns {Promise<{ data: Uint8ClampedArray, width: number, height: number }>}
   */
  async readSamplePixels(split, id) {
    if (typeof createImageBitmap === 'undefined' || typeof document === 'undefined') {
      throw new Error('读取样本像素仅在浏览器环境中可用。');
    }
    this._checkSplit(split);
    const entry = this.manifests[split].find((e) => e.id === id);
    if (!entry) throw new Error(`样本不存在：${id}`);
    const blob = await this.backend.readFile(`datasets/${split}/${entry.label}/${entry.file}`);
    const bitmap = await createImageBitmap(blob);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(bitmap, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      return { data: imageData.data, width: canvas.width, height: canvas.height };
    } finally {
      if (typeof bitmap.close === 'function') bitmap.close();
    }
  }

  _modelFile(name) {
    return `models/${sanitizeFileName(name)}.nnmodel.json`;
  }

  /**
   * 列出模型：扫描 models/*.nnmodel.json，返回 [{ name, id, architecture, metrics, trainedAt }]。
   * 损坏的文件会被跳过。
   */
  async listModels() {
    const entries = await this.backend.list('models');
    const out = [];
    for (const e of entries) {
      if (e.kind !== 'file' || !e.name.endsWith('.nnmodel.json')) continue;
      try {
        const blob = await this.backend.readFile(`models/${e.name}`);
        const doc = JSON.parse(await blob.text());
        out.push({
          name: typeof doc.name === 'string' ? doc.name : e.name.slice(0, -'.nnmodel.json'.length),
          id: doc.id,
          architecture: doc.architecture,
          metrics: doc.metrics,
          trainedAt: doc.trainedAt,
        });
      } catch (e2) {
        // 损坏的模型文件：跳过，不中断列表
      }
    }
    out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    return out;
  }

  /**
   * 保存模型：同名覆盖。net 为纯数据 { arch, w1, b1, w2, b2 }，无需 import network.js。
   * 文件内容字段：id、name、architecture、weights、optimizerState、metrics、trainedAt。
   */
  async saveModel(name, net, optimizerState, metrics) {
    if (!net || !net.arch || !Array.isArray(net.w1) || !Array.isArray(net.b1)
        || !Array.isArray(net.w2) || !Array.isArray(net.b2)) {
      throw new Error('net 必须为 { arch, w1, b1, w2, b2 } 纯数据对象。');
    }
    const doc = {
      id: this._newId(),
      name,
      architecture: { ...net.arch },
      weights: { w1: net.w1, b1: net.b1, w2: net.w2, b2: net.b2 },
      optimizerState: optimizerState === undefined ? null : optimizerState,
      metrics: metrics === undefined ? null : metrics,
      trainedAt: new Date().toISOString(),
    };
    await this.backend.writeFile(this._modelFile(name), JSON.stringify(doc, null, 2));
  }

  /** 加载模型，返回 { id, name, architecture, net, optimizerState, metrics, trainedAt }。 */
  async loadModel(name) {
    let doc;
    try {
      const blob = await this.backend.readFile(this._modelFile(name));
      doc = JSON.parse(await blob.text());
    } catch (e) {
      throw new Error(`模型不存在或已损坏：${name}`);
    }
    const w = doc.weights || {};
    return {
      id: doc.id,
      name: doc.name,
      architecture: doc.architecture,
      net: { arch: doc.architecture, w1: w.w1, b1: w.b1, w2: w.w2, b2: w.b2 },
      optimizerState: doc.optimizerState,
      metrics: doc.metrics,
      trainedAt: doc.trainedAt,
    };
  }

  /** 删除模型；不存在时抛错。 */
  async deleteModel(name) {
    const file = this._modelFile(name);
    if (!(await this.backend.exists(file))) throw new Error(`模型不存在：${name}`);
    await this.backend.remove(file);
  }
}

/**
 * 请求用户选择工作区目录（需在用户手势中调用）。
 * 不支持 File System Access API 的浏览器会抛中文友好错误。
 * @returns {Promise<FileSystemDirectoryHandle>}
 */
export async function requestDirectory() {
  if (typeof globalThis.showDirectoryPicker !== 'function') {
    throw new Error('当前浏览器不支持文件系统访问（File System Access API），请使用桌面版 Chrome 或 Edge 打开本应用。');
  }
  return globalThis.showDirectoryPicker({ mode: 'readwrite' });
}

/**
 * 确保拥有目录的读写权限：先 queryPermission，已授权直接返回 true，
 * 否则 requestPermission（会弹窗请求用户确认）。返回是否获得授权。
 */
export async function ensurePermission(handle) {
  if (!handle) return false;
  try {
    if (typeof handle.queryPermission === 'function') {
      if (await handle.queryPermission({ mode: 'readwrite' }) === 'granted') return true;
    }
    if (typeof handle.requestPermission === 'function') {
      return await handle.requestPermission({ mode: 'readwrite' }) === 'granted';
    }
    return true;
  } catch (e) {
    return false;
  }
}

const RECENT_DB = 'nn-anatomy2';
const RECENT_STORE = 'kv';
const RECENT_KEY = 'recent-workspace';

function openRecentDb() {
  return new Promise((resolve, reject) => {
    const idb = globalThis.indexedDB;
    if (!idb) {
      reject(new Error('当前环境不支持 IndexedDB，无法记住最近的工作区。'));
      return;
    }
    const req = idb.open(RECENT_DB, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(RECENT_STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(new Error('打开本地数据库失败，无法记住最近的工作区。'));
  });
}

/** 把目录句柄存入 IndexedDB，供下次启动恢复。 */
export async function saveRecentHandle(handle) {
  const db = await openRecentDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(RECENT_STORE, 'readwrite');
    tx.objectStore(RECENT_STORE).put(handle, RECENT_KEY);
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { db.close(); reject(new Error('保存最近工作区失败。')); };
  });
}

/** 读取上次保存的目录句柄；没有则返回 null。 */
export async function loadRecentHandle() {
  const db = await openRecentDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(RECENT_STORE, 'readonly');
    const req = tx.objectStore(RECENT_STORE).get(RECENT_KEY);
    req.onsuccess = () => { db.close(); resolve(req.result === undefined ? null : req.result); };
    req.onerror = () => { db.close(); reject(new Error('读取最近工作区失败。')); };
  });
}

/**
 * Canvas → PNG Blob（仅浏览器）。供手写板"保存样本"使用。
 * @returns {Promise<Blob>}
 */
export function canvasToPngBlob(canvas) {
  if (!canvas || typeof canvas.toBlob !== 'function') {
    return Promise.reject(new Error('当前环境不支持 Canvas，无法编码 PNG。'));
  }
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('Canvas 编码 PNG 失败。'));
    }, 'image/png');
  });
}
