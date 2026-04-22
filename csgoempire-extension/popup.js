/**
 * popup.js - Logic cho extension popup
 */

let rollHistory = [];

// ── Load data khi popup mở ───────────────────────────────────────────────
async function loadHistory() {
  const response = await chrome.runtime.sendMessage({
    source: 'csgoempire_tracker',
    type: 'get_history'
  });
  rollHistory = response?.history || [];
  renderAll();
}

// ── Render toàn bộ UI ────────────────────────────────────────────────────
function renderAll() {
  renderStats();
  renderRollList();
}

function renderStats() {
  const total = rollHistory.length;
  const ct = rollHistory.filter(r => isCT(r)).length;
  const t  = rollHistory.filter(r => isT(r)).length;

  document.getElementById('total-count').textContent = total;
  document.getElementById('ct-count').textContent = ct;
  document.getElementById('t-count').textContent = t;
}

function isCT(roll) {
  const v = roll.extracted?.roll;
  const c = (roll.extracted?.color || '').toLowerCase();
  return c.includes('ct') || c.includes('blue') || (v !== null && v >= 1 && v <= 7);
}

function isT(roll) {
  const v = roll.extracted?.roll;
  const c = (roll.extracted?.color || '').toLowerCase();
  return c.includes('t') || c.includes('orange') || (v !== null && v >= 8 && v <= 14);
}

function getColorClass(roll) {
  const v = roll.extracted?.roll;
  if (v === 0) return 'roll-zero';
  if (isCT(roll)) return 'roll-ct';
  if (isT(roll)) return 'roll-t';
  return 'roll-ct';
}

function renderRollList() {
  const list = document.getElementById('roll-list');
  const empty = document.getElementById('empty-state');

  if (rollHistory.length === 0) {
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';

  // Hiện 30 roll gần nhất
  const recent = rollHistory.slice(0, 30);
  list.innerHTML = recent.map((roll, i) => {
    const val = roll.extracted?.roll ?? '?';
    const colorClass = getColorClass(roll);
    const time = new Date(roll.timestamp).toLocaleTimeString();
    const rawStr = JSON.stringify(roll.raw || {}, null, 2).slice(0, 300);

    return `
      <div class="roll-item">
        <div class="roll-num ${colorClass}">${val}</div>
        <div class="roll-info">
          <div class="roll-event">${roll.eventName || 'unknown event'}</div>
          <div class="roll-time">${time}</div>
          <span class="raw-toggle" onclick="toggleRaw(${i})">raw data</span>
          <div class="raw-data" id="raw-${i}">${rawStr}</div>
        </div>
      </div>
    `;
  }).join('');
}

function toggleRaw(i) {
  const el = document.getElementById(`raw-${i}`);
  el.style.display = el.style.display === 'block' ? 'none' : 'block';
}

// ── Listen realtime updates ──────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'new_roll') {
    rollHistory.unshift(message.data);
    renderAll();
  }
});

// ── Export JSON ──────────────────────────────────────────────────────────
document.getElementById('btn-export').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(rollHistory, null, 2)], {
    type: 'application/json'
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `csgoempire_rolls_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// ── Clear data ───────────────────────────────────────────────────────────
document.getElementById('btn-clear').addEventListener('click', async () => {
  if (!confirm('Xóa toàn bộ dữ liệu?')) return;
  await chrome.runtime.sendMessage({
    source: 'csgoempire_tracker',
    type: 'clear_history'
  });
  rollHistory = [];
  renderAll();
});

// ── Init ─────────────────────────────────────────────────────────────────
loadHistory();
