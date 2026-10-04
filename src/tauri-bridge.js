// Tauri 前后端桥接：统一封装 invoke 调用。
// 要求 Tauri 2 且 tauri.conf.json 中 withGlobalTauri: true（提供 window.__TAURI__）。

export async function invoke(cmd, args) {
  return window.__TAURI__.core.invoke(cmd, args);
}

export async function greet(name) {
  return invoke('greet', { name });
}
