# Changelog — Empire-Predictor

All notable changes to this project will be documented in this file.

---

## [4.8.1] — 2026-04-24
- **Data Integrity**: Resolved a double-entry bug where incoming rounds were appended to the cache twice during socket-sync transitions.
- **Payload Sanitization**: Implemented `parseInt()` conversion in the extension to ensure outcome values are consistently treated as integers.
- **Reliability**: Optimized `_process_roll_sync` logic to prioritize live socket history over stale database records during sequence restoration.

---

## [4.8.0] — 2026-04-24
- **Socket History Sync**: Extension now extracts the full `rolls[100]` historical array directly from the platform's Socket.io stream.
- **Zero-Delay Bootstrapping**: The server can now instantly restore full predictive context within a single round by utilizing the socket-provided history.
- **Sequence Continuity Engine**: Automated gap detection and cache-flush logic to prevent estimation errors during network interruptions.

---

## [4.6.0] — 2026-04-23
- **Context Safeguards**: Implemented a "Warm-up Guard" that suspends complex sequence modules until 60 contiguous rounds of telemetry are captured.
- **Synchronized Visuals**: The dashboard now performs a full-history re-render upon receiving a `sync` event from the server.

---

## [4.0.0] — 2026-04-23
- **Engine Redesign**: Migrated to a **7-layer Ensemble Engine** utilizing parallel analytical modules.
- **Attention-based Sequences**: Integrated an Additive Attention layer into the Sequence Analyzer to improve focus on high-volatility transitions.
- **Parameterized States**: Expanded the Dynamic Learner's state-space to include regime-based buckets for market volatility detection.

---

## [1.0.0] — 2026-04-21
- **Initial Deployment**: Launched the core Empire-Predictor architecture.
- **Components**: Integrated the Chrome Extension capture tool, FastAPI analytics backend, and real-time visualization dashboard.
- **Baseline Logic**: Implemented basic probability modules and Orders-3 state transitioning.
