/**
 * background.js - Service Worker v2
 * Nhận roll events từ content script, lưu vào storage + gửi tới Python server
 */

const LOG_PREFIX = '[CSGOEmpire BG]';
const SERVER_URL = 'http://localhost:8000/api/roll';

// ── Nhận message từ content script (qua postMessage bridge) ─────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.source !== 'csgoempire_tracker') return;

  if (message.type === 'roll_event') {
    handleRollEvent(message.data);
    sendResponse({ ok: true });
  }

  if (message.type === 'get_history') {
    getRollHistory().then(history => sendResponse({ history }));
    return true; // async response
  }

  if (message.type === 'clear_history') {
    chrome.storage.local.remove('roll_history', () => {
      sendResponse({ ok: true });
    });
    return true;
  }
});

// ── Lưu roll event vào storage + gửi tới server ─────────────────────────
async function handleRollEvent(rollData) {
  try {
    const result = await chrome.storage.local.get('roll_history');
    const history = result.roll_history || [];

    // Thêm vào đầu mảng (mới nhất trước)
    history.unshift({
      ...rollData,
      savedAt: Date.now()
    });

    // Giữ tối đa 5000 records
    if (history.length > 5000) history.splice(5000);

    await chrome.storage.local.set({ roll_history: history });

    console.log(`${LOG_PREFIX} Saved roll #${history.length}:`, rollData.extracted);

    // Notify popup nếu đang mở
    chrome.runtime.sendMessage({
      type: 'new_roll',
      data: rollData
    }).catch(() => { }); // popup có thể không mở

    // ── GỬI TỚI PYTHON SERVER ──────────────────────────────────────────
    sendToServer(rollData);

  } catch (e) {
    console.error(`${LOG_PREFIX} Error saving:`, e);
  }
}

// ── POST roll data tới local Python server ───────────────────────────────
async function sendToServer(rollData) {
  const ext = rollData.extracted || {};
  const payload = {
    round_id: ext.round || 0,
    outcome: ext.winner ?? 0,
    color: mapCoin(ext.coin),
    timestamp: rollData.timestamp || Date.now()
  };

  try {
    const resp = await fetch(SERVER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (resp.ok) {
      const data = await resp.json();
      console.log(`${LOG_PREFIX} ✅ Server received. Prediction:`, data.prediction?.color, data.prediction?.confidence);
    }
  } catch (e) {
    // Server offline — silently ignore, data is in storage as backup
    console.log(`${LOG_PREFIX} ⚠️ Server offline, data saved locally`);
  }
}

function mapCoin(coin) {
  if (!coin) return null;
  const c = coin.toLowerCase();
  if (c === 't' || c === 'orange') return 'T';
  if (c === 'ct' || c === 'blue') return 'CT';
  if (c === 'bonus' || c === 'green') return 'Bonus';
  return null;
}

// ── Lấy lịch sử ─────────────────────────────────────────────────────────
async function getRollHistory() {
  const result = await chrome.storage.local.get('roll_history');
  return result.roll_history || [];
}

console.log(`${LOG_PREFIX} Service worker v2 started (+ server integration)`);
