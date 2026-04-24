/**
 * dashboard.js — Empire Predictor WebSocket Dashboard
 * Realtime updates, chart rendering, backtest/Monte Carlo integration
 */

const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

let ws = null;
let reconnectTimer = null;
let recentRolls = [];
let lastHandledRoundId = 0; // Kịch Kim 4.5: Anti-duplicate guard


// ── WebSocket Connection ────────────────────────────────────────────────────

function connectWS() {
    if (ws && ws.readyState <= 1) return;

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('[WS] Connected');
        setConnectionStatus(true);
        clearInterval(reconnectTimer);
        // Ping keepalive
        setInterval(() => {
            if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'ping' }));
        }, 30000);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWSMessage(data);
        } catch (e) {
            console.error('[WS] Parse error:', e);
        }
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected');
        setConnectionStatus(false);
        reconnectTimer = setInterval(connectWS, 3000);
    };

    ws.onerror = () => setConnectionStatus(false);
}

function setConnectionStatus(connected) {
    const el = document.getElementById('connection-status');
    if (connected) {
        el.className = 'status-badge connected';
        el.querySelector('span:last-child').textContent = 'Connected';
    } else {
        el.className = 'status-badge';
        el.querySelector('span:last-child').textContent = 'Disconnected';
    }
}

// ── Handle WebSocket Messages ───────────────────────────────────────────────

function handleWSMessage(data) {
    if (data.type === 'init') {
        recentRolls = data.recent || [];
        updatePrediction(data.prediction);
        updateHealth(data.health);
        updateStats(data.stats);
        updateModelWeights(data.model_weights);
        renderRollDots(recentRolls);
    }

    if (data.type === 'roll') {
        const rid = data.roll?.round_id || 0;
        if (rid <= lastHandledRoundId && rid !== 0) {
            console.log(`[WS] Skipping already handled round_id: ${rid} (Last: ${lastHandledRoundId})`);
            return;
        }
        lastHandledRoundId = rid;

        // New roll received!
        if (data.recent) recentRolls = data.recent;
        renderRollDots(recentRolls, true);
        updatePrediction(data.prediction);
        updateHealth(data.health);

        // Update round ID
        if (data.roll) {
            document.getElementById('round-id').textContent = `#${data.roll.round_id}`;
        }

        // Flash effect
        flashPanel('panel-realtime');
    }

    if (data.type === 'sync') {
        recentRolls = data.recent || [];
        lastHandledRoundId = data.last_id || lastHandledRoundId;
        renderRollDots(recentRolls);
        console.log('[WS] 🔄 Sequence synchronized with live floor. Ready.');
    }

    if (data.type === 'pong') return;
}

function flashPanel(panelId) {
    const panel = document.getElementById(panelId);
    panel.style.boxShadow = '0 0 30px rgba(100, 126, 234, 0.3)';
    setTimeout(() => panel.style.boxShadow = '', 500);
}

// ── Update Prediction Display ───────────────────────────────────────────────

function updatePrediction(pred) {
    if (!pred) return;

    const hero = document.getElementById('prediction-hero');
    const badge = document.getElementById('pred-badge');
    const action = document.getElementById('pred-action');
    const confFill = document.getElementById('confidence-fill');
    const confText = document.getElementById('pred-confidence');
    const betAmount = document.getElementById('bet-amount');

    // Reset classes
    hero.className = 'prediction-hero';
    badge.className = 'pred-color-badge';

    if (pred.action === 'SKIP') {
        hero.classList.add('skip');
        badge.textContent = '—';
        action.textContent = `SKIP: ${pred.skip_reason || 'Low confidence'}`;
        betAmount.textContent = '—';
    } else {
        const color = pred.color;
        hero.classList.add(`bet-${color.toLowerCase()}`);
        badge.classList.add(`badge-${color.toLowerCase()}`);
        badge.textContent = color;
        action.textContent = `BET ${color} (${pred.risk_level || ''})`;
        betAmount.textContent = pred.bet_amount ? pred.bet_amount.toFixed(1) : '—';
    }

    const conf = (pred.confidence || 0) * 100;
    confFill.style.width = `${conf}%`;
    confText.textContent = `${conf.toFixed(1)}%`;

    // Probability bars
    const probs = pred.probs || {};
    updateProbBar('prob-t', 'prob-t-pct', probs.T || 0);
    updateProbBar('prob-ct', 'prob-ct-pct', probs.CT || 0);
    updateProbBar('prob-bonus', 'prob-bonus-pct', probs.Bonus || 0);

    // Model votes
    const votes = pred.model_votes || {};
    // Item: Dynamic loop for all 7 models
    const models = ['mamba', 'tft', 'foundation', 'lstm', 'markov', 'statistical', 'rl_agent'];

    models.forEach(model => {
        const el = document.getElementById(`vote-${model}`);
        if (!el) return;

        const info = votes[model] || { vote: '—', weight: 0 };
        const vBadge = el.querySelector('.vote-badge');
        const vWeight = el.querySelector('.vote-weight');

        const vote = info.vote || '—';
        vBadge.textContent = vote;
        vBadge.className = `vote-badge vb-${vote.toLowerCase()}`;
        vWeight.textContent = `w: ${(info.weight * 100).toFixed(0)}%`;

        // Interactive Probability Tooltip
        el.onclick = () => {
            if (!info.probs) return;
            const toast = document.createElement('div');
            toast.className = 'glass-panel';
            toast.style.position = 'fixed';
            toast.style.top = '100px';
            toast.style.left = '50%';
            toast.style.transform = 'translateX(-50%)';
            toast.style.zIndex = '10000';
            toast.style.padding = '15px 25px';
            toast.style.borderRadius = '8px';
            toast.style.border = '1px solid rgba(255, 255, 255, 0.1)';
            toast.style.boxShadow = '0 10px 30px rgba(0,0,0,0.5)';
            toast.style.pointerEvents = 'none';
            toast.style.transition = 'opacity 0.3s ease';
            toast.style.textAlign = 'center';

            toast.innerHTML = `<h3 style="margin-top:0; color:#48c774;">${model.toUpperCase()} PROBABILITIES</h3>
                               <div style="display:flex; justify-content:center; gap:15px; font-family: 'JetBrains Mono', monospace;">
                                  <span style="color:#FFA07A">T: ${(info.probs.T * 100).toFixed(1)}%</span>
                                  <span style="color:#87CEFA">CT: ${(info.probs.CT * 100).toFixed(1)}%</span>
                                  <span style="color:#FFD700">B: ${(info.probs.Bonus * 100).toFixed(1)}%</span>
                               </div>`;
            document.body.appendChild(toast);
            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        };
        el.style.cursor = 'pointer';

        // Add "Kịch Kim" class to highlight high performers or the True Temporal indicator
        if (model === 'tft' && info.weight > 0) {
            el.classList.add('temporal-active');
        }
    });
}

function updateProbBar(barId, pctId, value) {
    const bar = document.getElementById(barId);
    const pct = document.getElementById(pctId);
    if (bar) bar.style.width = `${value * 100}%`;
    if (pct) pct.textContent = `${(value * 100).toFixed(0)}%`;
}

// ── Update Health Display ───────────────────────────────────────────────────

function updateHealth(health) {
    if (!health) return;

    // Health score gauge
    const score = health.score || 0;
    const label = health.label || 'Unknown';

    const gaugeText = document.getElementById('gauge-text');
    if (gaugeText) gaugeText.textContent = score;

    const healthLabel = document.getElementById('health-label');
    if (healthLabel) healthLabel.textContent = label;

    // Gauge arc
    const fillEl = document.getElementById('gauge-fill');
    if (fillEl) {
        const offset = 251 - (score / 100 * 251);
        fillEl.style.strokeDashoffset = offset;
        const gaugeColor = score >= 80 ? '#48c774' : score >= 60 ? '#667eea' : score >= 40 ? '#ffc107' : '#ff6b6b';
        fillEl.style.stroke = gaugeColor;
    }

    // Session stats
    const streak = health.streak || {};
    const sRounds = document.getElementById('session-rounds');
    if (sRounds) sRounds.textContent = health.stats?.session_rounds || 0;

    const sWinrate = document.getElementById('session-winrate');
    if (sWinrate) sWinrate.textContent = health.stats?.win_rate !== undefined ? `${(health.stats.win_rate * 100).toFixed(0)}%` : '—';

    const sStreak = document.getElementById('session-streak');
    if (sStreak) sStreak.textContent = streak.streak_len ? `${streak.streak_len}× ${streak.streak_color}` : '—';

    // Variance
    const variance = health.variance || {};
    const zEl = document.getElementById('var-zscore');
    if (zEl) {
        zEl.textContent = variance.z_score || '0';
        const z = Math.abs(variance.z_score || 0);
        zEl.style.color = z > 2 ? '#ff6b6b' : z > 1 ? '#ffc107' : '#48c774';
    }

    const rEl = document.getElementById('var-ratio');
    if (rEl) rEl.textContent = variance.ratio !== undefined ? `${(variance.ratio * 100).toFixed(1)}%` : '50%';
}

// ── Render Roll Dots ────────────────────────────────────────────────────────

function renderRollDots(colors, animate = false) {
    const container = document.getElementById('roll-dots');
    if (!container) return;

    const dots = colors.map((color, i) => {
        const cls = `dot-${color.toLowerCase()}`;
        const isNew = animate && i === colors.length - 1;
        return `<div class="roll-dot ${cls} ${isNew ? 'dot-new' : ''}" title="${color}">${colorToShort(color)}</div>`;
    });
    container.innerHTML = dots.join('');

    const totalEl = document.getElementById('total-rounds');
    if (totalEl) totalEl.textContent = colors.length;
}

function colorToShort(color) {
    return color === 'Bonus' ? 'B' : color;
}

// ── Update Stats (Analytics panel) ──────────────────────────────────────────

function updateStats(stats) {
    if (!stats) return;

    const overall = stats.overall || {};
    const total = overall.total_rounds || 1;

    const distT = document.getElementById('dist-t');
    const distCT = document.getElementById('dist-ct');
    const distB = document.getElementById('dist-bonus');

    const tPct = (overall.t_count || 0) / total;
    const ctPct = (overall.ct_count || 0) / total;
    const bPct = (overall.bonus_count || 0) / total;
    const maxPct = Math.max(tPct, ctPct, bPct, 0.01);

    if (distT) distT.style.height = `${(tPct / maxPct) * 100}%`;
    if (distCT) distCT.style.height = `${(ctPct / maxPct) * 100}%`;
    if (distB) distB.style.height = `${(bPct / maxPct) * 100}%`;

    const distTCount = document.getElementById('dist-t-count');
    const distCTCount = document.getElementById('dist-ct-count');
    const distBCount = document.getElementById('dist-bonus-count');

    if (distTCount) distTCount.textContent = `${overall.t_count || 0} (${(tPct * 100).toFixed(1)}%)`;
    if (distCTCount) distCTCount.textContent = `${overall.ct_count || 0} (${(ctPct * 100).toFixed(1)}%)`;
    if (distBCount) distBCount.textContent = `${overall.bonus_count || 0} (${(bPct * 100).toFixed(1)}%)`;

    const preds = stats.predictions || {};
    const accPct = document.getElementById('acc-pct');
    const accDetail = document.getElementById('acc-detail');
    if (preds.total_predictions > 0) {
        if (accPct) accPct.textContent = `${(preds.accuracy * 100).toFixed(1)}%`;
        if (accDetail) accDetail.textContent = `${preds.correct}/${preds.total_predictions} correct`;
    }

    const streaks = stats.streaks?.streaks || {};
    const stT = document.getElementById('streak-t-max');
    const stCT = document.getElementById('streak-ct-max');
    const stB = document.getElementById('streak-bonus-max');
    if (stT) stT.textContent = streaks.T?.max || '—';
    if (stCT) stCT.textContent = streaks.CT?.max || '—';
    if (stB) stB.textContent = streaks.Bonus?.max || '—';
}

function updateModelWeights(weights) {
    if (!weights) return;
    const models = ['mamba', 'tft', 'foundation', 'lstm', 'markov', 'statistical', 'rl_agent'];
    models.forEach(model => {
        const w = weights[model] || 0;
        const fillEl = document.getElementById(`w-${model}`);
        const pctEl = document.getElementById(`w-${model}-pct`);
        if (fillEl) fillEl.style.width = `${w * 100}%`;
        if (pctEl) pctEl.textContent = `${(w * 100).toFixed(0)}%`;
    });
}

// ── Backtest ────────────────────────────────────────────────────────────────

async function runBacktest() {
    const btn = document.getElementById('btn-backtest');
    btn.disabled = true;
    btn.textContent = 'Running...';

    const params = {
        strategy: document.getElementById('bt-strategy').value,
        bankroll: parseFloat(document.getElementById('bt-bankroll').value),
        confidence_threshold: parseFloat(document.getElementById('bt-confidence').value),
    };

    try {
        const resp = await fetch(`${API_BASE}/api/backtest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        const data = await resp.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Display results
        document.getElementById('bt-results').style.display = 'block';
        document.getElementById('bt-roi').textContent = data.roi_pct;
        document.getElementById('bt-roi').style.color = data.roi >= 0 ? '#48c774' : '#ff6b6b';
        document.getElementById('bt-winrate').textContent = `${(data.win_rate * 100).toFixed(1)}%`;
        document.getElementById('bt-drawdown').textContent = `${(data.max_drawdown * 100).toFixed(1)}%`;
        document.getElementById('bt-sharpe').textContent = data.sharpe_ratio;
        document.getElementById('bt-final').textContent = data.final_bankroll;
        document.getElementById('bt-final').style.color = data.final_bankroll >= params.bankroll ? '#48c774' : '#ff6b6b';
        document.getElementById('bt-bets').textContent = data.total_bets;

        // Draw bankroll curve
        drawBankrollChart(data.curve_sample || data.bankroll_curve);

    } catch (e) {
        alert('Server error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Backtest';
    }
}

async function runMonteCarlo() {
    const btn = document.getElementById('btn-montecarlo');
    btn.disabled = true;
    btn.textContent = 'Simulating...';

    const params = {
        n_simulations: 10000,
        n_rounds: 100,
        bankroll: parseFloat(document.getElementById('bt-bankroll').value),
        bet_fraction: 0.05,
    };

    try {
        const resp = await fetch(`${API_BASE}/api/montecarlo`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        const data = await resp.json();

        if (data.error) { alert(data.error); return; }

        document.getElementById('mc-results').style.display = 'block';
        document.getElementById('mc-mean').textContent = data.results.mean;
        document.getElementById('mc-median').textContent = data.results.median;
        document.getElementById('mc-p10').textContent = data.results.p10;
        document.getElementById('mc-p90').textContent = data.results.p90;
        document.getElementById('mc-ruin').textContent = `${(data.ruin_probability * 100).toFixed(1)}%`;
        document.getElementById('mc-profit').textContent = `${(data.profit_probability * 100).toFixed(1)}%`;

        drawHistogramChart(data.histogram);

    } catch (e) {
        alert('Server error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Monte Carlo (10K)';
    }
}

// ── Canvas Charts ───────────────────────────────────────────────────────────

function drawBankrollChart(data) {
    const canvas = document.getElementById('bt-chart');
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 400;
    const pad = { t: 20, r: 20, b: 30, l: 50 };

    ctx.clearRect(0, 0, w, h);
    ctx.scale(1, 1);

    if (!data || data.length < 2) return;

    const minVal = Math.min(...data) * 0.95;
    const maxVal = Math.max(...data) * 1.05;
    const range = maxVal - minVal || 1;

    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + plotH * (i / 4);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        const val = maxVal - (range * i / 4);
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = '18px JetBrains Mono';
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(0), pad.l - 8, y + 5);
    }

    // Starting line (bankroll = initial)
    const startY = pad.t + plotH * (1 - (data[0] - minVal) / range);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad.l, startY); ctx.lineTo(w - pad.r, startY); ctx.stroke();
    ctx.setLineDash([]);

    // Line chart
    const gradient = ctx.createLinearGradient(0, pad.t, 0, h - pad.b);
    gradient.addColorStop(0, '#667eea');
    gradient.addColorStop(1, '#764ba2');

    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
        const x = pad.l + (i / (data.length - 1)) * plotW;
        const y = pad.t + plotH * (1 - (data[i] - minVal) / range);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill area
    const lastX = pad.l + plotW;
    const lastY = pad.t + plotH * (1 - (data[data.length - 1] - minVal) / range);
    ctx.lineTo(lastX, h - pad.b);
    ctx.lineTo(pad.l, h - pad.b);
    ctx.closePath();
    const fillGrad = ctx.createLinearGradient(0, pad.t, 0, h - pad.b);
    fillGrad.addColorStop(0, 'rgba(102, 126, 234, 0.15)');
    fillGrad.addColorStop(1, 'rgba(102, 126, 234, 0)');
    ctx.fillStyle = fillGrad;
    ctx.fill();
}

function drawHistogramChart(histogram) {
    if (!histogram) return;
    const canvas = document.getElementById('mc-chart');
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 400;
    const pad = { t: 20, r: 20, b: 30, l: 50 };

    ctx.clearRect(0, 0, w, h);

    const counts = histogram.counts || [];
    const edges = histogram.edges || [];
    const maxCount = Math.max(...counts, 1);
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    const barW = plotW / counts.length;

    for (let i = 0; i < counts.length; i++) {
        const barH = (counts[i] / maxCount) * plotH;
        const x = pad.l + i * barW;
        const y = h - pad.b - barH;

        const gradient = ctx.createLinearGradient(x, y, x, h - pad.b);
        gradient.addColorStop(0, 'rgba(102, 126, 234, 0.8)');
        gradient.addColorStop(1, 'rgba(118, 75, 162, 0.4)');

        ctx.fillStyle = gradient;
        ctx.fillRect(x + 1, y, barW - 2, barH);
    }

    // Axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '16px JetBrains Mono';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {
        const idx = Math.floor(i * (edges.length - 1) / 4);
        const x = pad.l + (idx / (edges.length - 1)) * plotW;
        ctx.fillText(edges[idx]?.toFixed(0) || '', x, h - 8);
    }
}

// ── Load initial data via REST (fallback if WS not connected) ───────────────

async function loadInitialData() {
    try {
        // Fetch stats
        const statsResp = await fetch(`${API_BASE}/api/stats`);
        if (statsResp.ok) {
            const stats = await statsResp.json();
            updateStats(stats);
            updateModelWeights(stats.model_weights);
        }

        // Fetch recent
        const recentResp = await fetch(`${API_BASE}/api/recent?limit=50`);
        if (recentResp.ok) {
            const data = await recentResp.json();
            const rolls = data.rolls || [];
            recentRolls = rolls.reverse().map(r => r.color);
            renderRollDots(recentRolls);
            document.getElementById('total-rounds').textContent = data.total || 0;
        }

        // Fetch prediction
        const predResp = await fetch(`${API_BASE}/api/predict`);
        if (predResp.ok) {
            const data = await predResp.json();
            updatePrediction(data.prediction);
        }

        // Fetch health
        const healthResp = await fetch(`${API_BASE}/api/health`);
        if (healthResp.ok) {
            const data = await healthResp.json();
            updateHealth(data);
        }

    } catch (e) {
        console.log('[Init] Server not available, waiting for WS...');
    }
}

// ── Bankroll input ──────────────────────────────────────────────────────────

document.getElementById('bankroll-input').addEventListener('change', async (e) => {
    const val = parseFloat(e.target.value);
    if (val > 0) {
        try {
            await fetch(`${API_BASE}/api/bankroll`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ bankroll: val })
            });
        } catch (_) { }
    }
});

// ── Add SVG gradient def programmatically ───────────────────────────────────

(function addSvgGradient() {
    const svg = document.querySelector('.health-gauge');
    if (!svg) return;
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    grad.setAttribute('id', 'gaugeGradient');
    const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop1.setAttribute('offset', '0%'); stop1.setAttribute('stop-color', '#ff6b6b');
    const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop2.setAttribute('offset', '50%'); stop2.setAttribute('stop-color', '#ffc107');
    const stop3 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop3.setAttribute('offset', '100%'); stop3.setAttribute('stop-color', '#48c774');
    grad.appendChild(stop1); grad.appendChild(stop2); grad.appendChild(stop3);
    defs.appendChild(grad);
    svg.insertBefore(defs, svg.firstChild);
})();

// ── Init ────────────────────────────────────────────────────────────────────

loadInitialData();
connectWS();

// Auto-refresh stats every 30s
setInterval(async () => {
    try {
        const resp = await fetch(`${API_BASE}/api/stats`);
        if (resp.ok) {
            const stats = await resp.json();
            updateStats(stats);
            updateModelWeights(stats.model_weights);
        }
    } catch (_) { }
}, 30000);

// Tilt and Onboarding removed in v2.0
