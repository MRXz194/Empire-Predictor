"""
alerts.py - Tầng 7: Cảnh báo & Notifications
Gửi Windows Desktop Notification khi có Tilt hoặc có Bet độ tin cậy cực cao.
"""
import threading

def _send_notification_async(title: str, msg: str, duration: int = 5):
    try:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(title, msg, duration=duration, threaded=True)
    except Exception as e:
        print(f"[Alert] Notification failed: {e}")

def trigger_alert(title: str, msg: str, duration: int = 5):
    """Gởi notification chạy trong thread riêng để không block FastAPI"""
    thread = threading.Thread(target=_send_notification_async, args=(title, msg, duration))
    thread.daemon = True
    thread.start()



def check_high_confidence_alert(prediction: dict):
    if not prediction or 'confidence' not in prediction:
        return
    conf = prediction['confidence']
    action = prediction.get('action', 'SKIP')
    color = prediction.get('color', 'T')
    
    # Alert if confidence > 75%
    if action != 'SKIP' and conf > 0.75:
        bet_amount = prediction.get('bet_amount', 0)
        trigger_alert("🔥 HIGH CONFIDENCE BET", f"Cơ hội ăn cao: {conf*100:.1f}%\nBet ngay: {bet_amount} coin vào {color}", duration=7)
