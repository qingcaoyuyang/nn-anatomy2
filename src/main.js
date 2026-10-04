import { greet } from './tauri-bridge.js';

document.querySelector('#app').textContent = await greet('舟哥');
