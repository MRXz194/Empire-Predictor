"""
session.py - Tầng 4: Session Analysis + Anomaly Detection
"""
import math
import time


class VarianceMonitor:
    """Z-score monitor for T/CT ratio deviation."""

    def __init__(self):
        self.expected_ratio = 0.5  # T vs CT expected 50/50 (excluding Bonus)
        self.alerts = []

    def compute(self, recent_colors: list[str], window: int = 100) -> dict:
        sample = recent_colors[-window:]
        tc_only = [c for c in sample if c in ('T', 'CT')]

        if len(tc_only) < 10:
            return {'z_score': 0, 'alert': False, 'ratio': 0.5}

        t_count = sum(1 for c in tc_only if c == 'T')
        n = len(tc_only)
        observed_ratio = t_count / n

        # Z-score for binomial proportion
        std = math.sqrt(self.expected_ratio * (1 - self.expected_ratio) / n)
        z = (observed_ratio - self.expected_ratio) / std if std > 0 else 0

        alert = abs(z) > 2.0
        if alert:
            direction = "T-heavy" if z > 0 else "CT-heavy"
            self.alerts.append({
                'time': int(time.time()),
                'z_score': round(z, 2),
                'direction': direction,
                'ratio': round(observed_ratio, 3)
            })

        return {
            'z_score': round(z, 3),
            'alert': alert,
            'ratio': round(observed_ratio, 3),
            't_count': t_count,
            'ct_count': n - t_count,
            'window': len(sample)
        }


class StreakDetector:
    """Consecutive same-color streak detector."""

    def compute(self, recent_colors: list[str]) -> dict:
        if not recent_colors:
            return {'streak_len': 0, 'streak_color': None, 'anomaly': False}

        streak_color = recent_colors[-1]
        streak_len = 1
        for i in range(len(recent_colors) - 2, -1, -1):
            if recent_colors[i] == streak_color:
                streak_len += 1
            else:
                break

        # Anomaly thresholds
        anomaly = False
        
        if streak_color == 'Bonus':
            anomaly = streak_len >= 2  # 2 Bonus in a row is very rare
        elif streak_len >= 7:
            anomaly = True

        # Longest streak in recent 100
        max_streak = streak_len
        if len(recent_colors) > 1:
            current = 1
            for i in range(len(recent_colors) - 1, 0, -1):
                if recent_colors[i] == recent_colors[i-1]:
                    current += 1
                    max_streak = max(max_streak, current)
                else:
                    current = 1

        return {
            'streak_len': streak_len,
            'streak_color': streak_color,
            'anomaly': anomaly,
            'max_streak_100': max_streak
        }


class RegimeDetector:
    """Detects current game 'regime' (Streak, Alternating, Random)."""

    def compute(self, recent_colors: list[str], window: int = 20) -> dict:
        sample = recent_colors[-window:]
        if len(sample) < 10:
            return {'regime': 'INITIALIZING', 'confidence': 0.0}

        # 1. Count switches (T -> CT or CT -> T)
        switches = 0
        for i in range(1, len(sample)):
            if sample[i] in ('T', 'CT') and sample[i-1] in ('T', 'CT'):
                if sample[i] != sample[i-1]:
                    switches += 1
        
        switch_rate = switches / (len(sample) - 1)

        # 2. Max streak in window
        max_s = 1
        curr = 1
        for i in range(1, len(sample)):
            if sample[i] == sample[i-1]:
                curr += 1
                max_s = max(max_s, curr)
            else:
                curr = 1

        # Logic
        if max_s >= 5:
            regime = 'STREAK'
            confidence = min(1.0, max_s / 8)
        elif switch_rate >= 0.7:
            regime = 'ALTERNATING'
            confidence = min(1.0, switch_rate)
        elif switch_rate <= 0.3 and max_s >= 3:
            regime = 'STREAK'
            confidence = 0.6
        else:
            regime = 'STABLE'
            confidence = 0.5

        return {
            'regime': regime,
            'confidence': round(confidence, 2),
            'switch_rate': round(switch_rate, 2),
            'max_streak': max_s,
            'streak_color': sample[-1] if max_s >= 3 else None
        }


class HealthScore:
    """Composite health score 0-100."""

    def __init__(self):
        self.variance_monitor = VarianceMonitor()
        self.streak_detector = StreakDetector()
        self.regime_detector = RegimeDetector()

    def compute(self, recent_colors: list[str], drift: bool = False) -> dict:
        variance = self.variance_monitor.compute(recent_colors)
        streak = self.streak_detector.compute(recent_colors)
        regime = self.regime_detector.compute(recent_colors)

        # Score components (each 0-100, higher = healthier)

        # Z-score component (60 weight): |z| > 3 = 0, |z| < 0.5 = 100
        z = abs(variance['z_score'])
        z_score = max(0, min(100, (3 - z) / 3 * 100))

        # Streak component (40 weight)
        s_len = streak['streak_len']
        s_score = max(0, min(100, (8 - s_len) / 8 * 100))

        health = round(z_score * 0.6 + s_score * 0.4)
        
        # Drift Penalty
        if drift:
            health -= 15
            
        health = max(0, min(100, health))

        # Label
        if health >= 80:
            label = 'EXCELLENT'
        elif health >= 60:
            label = 'GOOD'
        elif health >= 40:
            label = 'CAUTION'
        elif health >= 20:
            label = 'DANGER'
        else:
            label = 'CRITICAL'

        return {
            'score': health,
            'label': label,
            'components': {
                'variance': round(z_score, 1),
                'streak': round(s_score, 1),
                'drift_penalty': drift
            },
            'variance': variance,
            'streak': streak,
            'regime': regime,
            'stats': {
                'session_rounds': len(recent_colors),
                'win_rate': 0.5 # Placeholder since we don't track bankroll here anymore
            }
        }
