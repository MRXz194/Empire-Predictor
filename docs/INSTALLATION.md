# Installation Guide — Empire-Predictor

Empire-Predictor is optimized for a Windows-based environment.

## 1. Prerequisites
- **Python 3.11+**: Download from python.org.
- **Chrome Browser**: Required for the data capture extension.

## 2. Dependency Installation
Open your terminal in the project directory and execute:
```bash
pip install -r requirements.txt
```

## 3. Extension Setup
1. Navigate to `chrome://extensions/` in Google Chrome.
2. Enable **Developer mode**.
3. Click **Load unpacked** and select the `csgoempire-extension` folder.

## 4. Launch Procedure
1. Start the Backend Server: `python server/main.py`.
2. Open the Dashboard: Open `dashboard/index.html` in your browser.
3. Access the Market: Open the CSGOEmpire Roulette page to begin telemetry collection.

## 5. Operational Notes
- Ensure the server is running before initiating the data collection stream.
- The system requires a minimum of **60 consecutive rounds** to reach full operational capacity across all sequence-based modules.
