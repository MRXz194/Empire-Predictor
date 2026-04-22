/**
 * bridge.js - Isolated world content script
 * Nhận postMessage từ MAIN world, forward sang background
 * Chạy ở isolated world nên có quyền dùng chrome.runtime
 */

window.addEventListener('message', function (event) {
  if (
    event.source !== window ||
    !event.data ||
    event.data.source !== 'csgoempire_tracker'
  ) return;

  // Forward sang background service worker
  chrome.runtime.sendMessage(event.data).catch(() => {});
});
