/**
 * content.js - VERSION 3 - Fix Socket.io namespace format
 * Format thực tế: 42/roulette,["event",{...}]
 */
(function () {
  const LOG_PREFIX = '[CSGOEmpire Tracker]';
  const OriginalWebSocket = window.WebSocket;

  window.WebSocket = function (url, protocols) {
    const ws = protocols
      ? new OriginalWebSocket(url, protocols)
      : new OriginalWebSocket(url);

    const shortUrl = (url.split('/')[2] || url).slice(0, 40);
    console.log(`${LOG_PREFIX} WebSocket opened:`, url);

    ws.addEventListener('open', () => console.log(`${LOG_PREFIX} WS connected: [${shortUrl}]`));
    ws.addEventListener('close', () => console.log(`${LOG_PREFIX} WS closed: [${shortUrl}]`));

    ws.addEventListener('message', function (event) {
      try {
        const raw = event.data;
        if (typeof raw !== 'string' || raw.length <= 3) return;
        parseSocketIO(raw, shortUrl);
      } catch (e) { }
    });

    return ws;
  };

  Object.keys(OriginalWebSocket).forEach(key => {
    try { window.WebSocket[key] = OriginalWebSocket[key]; } catch (e) { }
  });
  window.WebSocket.prototype = OriginalWebSocket.prototype;
  window.WebSocket.CONNECTING = 0;
  window.WebSocket.OPEN = 1;
  window.WebSocket.CLOSING = 2;
  window.WebSocket.CLOSED = 3;

  // ── Parser đúng cho Socket.io với namespace ──────────────────────────────
  // Format: 42/namespace,["eventName", {...payload}]
  // Ví dụ:  42/roulette,["roll", {"winner":14, "coin":"ct", ...}]
  function parseSocketIO(raw, source) {
    // Chỉ xử lý message type 42 (EVENT)
    if (!raw.startsWith('42')) return;

    let jsonStr = raw.slice(2); // bỏ "42"

    // Có namespace: 42/roulette,["event",...]
    // Không có namespace: 42["event",...]
    if (jsonStr.startsWith('/')) {
      const commaIdx = jsonStr.indexOf(',');
      if (commaIdx === -1) return;
      const namespace = jsonStr.slice(1, commaIdx); // "roulette"
      jsonStr = jsonStr.slice(commaIdx + 1);        // ["event",...]

      // Chỉ quan tâm namespace roulette
      if (namespace !== 'roulette') return;
    }

    let parsed;
    try {
      parsed = JSON.parse(jsonStr);
    } catch (e) { return; }

    if (!Array.isArray(parsed) || parsed.length < 2) return;

    const eventName = parsed[0];
    const payload = parsed[1];

    // ── Chỉ lấy 3 events cần thiết ──────────────────────────────────────
    if (eventName === 'roll') {
      // Event chính - có winner và coin
      console.log(`%c${LOG_PREFIX} ROLL: round=${payload.round} winner=${payload.winner} coin=${payload.coin}`,
        'color:#00ff88;font-weight:bold', payload);
      sendRoll('roll', payload);

    } else if (eventName === 'spin') {
      // Báo hiệu vòng mới bắt đầu quay
      console.log(`%c${LOG_PREFIX} SPIN: round=${payload.round} (đang quay...)`,
        'color:#63b3ed;font-weight:bold');

    } else if (eventName === 'end') {
      // Kết thúc vòng - confirm winner
      console.log(`%c${LOG_PREFIX} END: round=${payload.round} winner=${payload.winner} coin=${payload.coin}`,
        'color:#f6ad55;font-weight:bold');

    } else if (eventName === 'rolling') {
      // Vòng mới bắt đầu betting
      console.log(`%c${LOG_PREFIX} NEW ROUND: ${payload.round} - Betting open`,
        'color:#9f7aea;font-weight:bold');
    }
  }

  // ── Kịch Kim 4.8: Sync chuyển hoàn toàn sang Socket History ────────────────

  // ── Gửi roll data lên background ────────────────────────────────────────
  function sendRoll(eventName, payload) {
    const rollData = {
      timestamp: Date.now(),
      eventName: eventName,
      raw: payload,
      extracted: {
        round: payload.round ?? null,
        winner: payload.winner ?? null,   // số 0-14
        coin: payload.coin ?? null,   // "ct", "t", "bonus"
        nextRound: payload.nextRound ?? null,
        rolls: payload.rolls ?? [],     // mảng animation
      }
    };

    window.postMessage({
      source: 'csgoempire_tracker',
      type: 'roll_event',
      data: rollData
    }, '*');
  }

  // Gỡ bỏ scrape cũ - Kịch Kim 4.8 chuyển sang Socket Sync
  console.log(`%c${LOG_PREFIX} v4.8 Active - Socket History Sync Enabled`, 'color:#68d391;font-weight:bold');
})();
