# API Reference — Empire-Predictor

**Base URL**: `http://localhost:8000`  
**WebSocket**: `ws://localhost:8000/ws`

---

## Endpoints

### `POST /api/roll`
Submits new round data for processing and analysis.

**Payload**:
```json
{
  "round_id": 12182716,
  "outcome": 11,
  "color": "CT",
  "history_full": ["CT", "T", "..."]
}
```

### `GET /api/predict`
Retrieves the current probability estimations and module analysis outcomes.

### `GET /api/stats`
Retrieves system performance metrics, database statistics, and operational health.

---

## WebSocket Communication

The system streams live updates via the `/ws` channel:
- **prediction**: Dynamic confidence levels and recommended actions.
- **roll**: Confirmation of the latest round outcome.
- **sync**: Command to trigger a complete UI history re-synchronization.
