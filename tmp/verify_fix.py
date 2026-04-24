import requests
import time
import json
import asyncio
import websockets

API_URL = "http://localhost:8000/api/roll"
WS_URL = "ws://localhost:8000/ws"

async def test_fix():
    print("[Test] Connecting to WebSocket...")
    try:
        async with websockets.connect(WS_URL) as ws:
            # Consume init message
            init_msg = await ws.recv()
            print("[Test] Received Init")
            
            # 1. Send Round #9001 (Color: T)
            print("[Test] Sending Round #9001: T")
            resp1 = requests.post(API_URL, json={
                "round_id": 9001,
                "outcome": 1, # T
                "color": "T",
                "history_full": ["CT", "T"]
            })
            print(f"[Test] Round #9001 status: {resp1.status_code}")
            
            msg1 = await ws.recv()
            data1 = json.parse(msg1) if hasattr(json, 'parse') else json.loads(msg1)
            print(f"[WS] Received: {data1.get('type')} Round: {data1.get('roll', {}).get('round_id')} Color: {data1.get('roll', {}).get('color')}")
            
            # 2. Send Round #9002 (Color: T) - SAME COLOR
            print("[Test] Sending Round #9002: T (SAME COLOR)")
            resp2 = requests.post(API_URL, json={
                "round_id": 9002,
                "outcome": 2, # T
                "color": "T",
                "history_full": ["CT", "T", "T"] # history_full updated
            })
            print(f"[Test] Round #9002 status: {resp2.status_code}")
            
            # Wait for WS message
            try:
                msg2 = await asyncio.wait_for(ws.recv(), timeout=5)
                data2 = json.loads(msg2)
                print(f"[WS] SUCCESS: Received Round #9002: {data2.get('roll', {}).get('color')}")
                print(f"[WS] Current Recent: {data2.get('recent')}")
                
                if data2['recent'][-1] == 'T' and data2['recent'][-2] == 'T':
                    print("[Test] ✅ VERIFIED: Both T rounds are present in the cache.")
                else:
                    print("[Test] ❌ FAILED: Sequence is not correct.")
            except asyncio.TimeoutError:
                print("[Test] ❌ FAILED: No WebSocket message received for same-color round.")

    except Exception as e:
        print(f"[Test] ❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_fix())
