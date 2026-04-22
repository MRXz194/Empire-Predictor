import requests
try:
    print("Triggering LSTM Training...")
    resp = requests.post("http://localhost:8000/api/train-lstm?epochs=5", timeout=300)
    print("Result:", resp.json())
except Exception as e:
    print("Error:", e)
