"""
rl_agent.py - Q-Learning Agent
State: (streak_bucket, last_color, freq_deviation_bucket, entropy_bucket)
Action: bet_T, bet_CT, bet_Bonus, skip
"""
import pickle
import os
import random
import math
import numpy as np
from collections import defaultdict, Counter

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'saved', 'q_table.pkl')

ACTIONS = ['bet_t', 'bet_ct', 'bet_bonus', 'skip']
COLOR_ODDS = {'t': 2.0, 'ct': 2.0, 'bonus': 14.0}  # payout multiplier


class QLearningAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.q_table = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.trained = False

    def _discretize_state(self, recent_colors: list[str], end_idx: int = None) -> tuple:
        """
        Convert recent history to a discrete state tuple.
        Item 12 Fix: Add bonus_since_bucket for richer context.
        """
        if end_idx is None:
            end_idx = len(recent_colors)
            
        if end_idx < 5:
            return (0, 0, 0, 0, 0)

        # 1. Streak bucket (0-5+)
        streak = 1
        last = recent_colors[end_idx - 1]
        for i in range(end_idx - 2, -1, -1):
            if recent_colors[i] == last:
                streak += 1
            else:
                break
        streak_bucket = min(streak, 5)

        # 2. Last color color
        last_color = 0 if last.upper() == 'T' else 1 if last.upper() == 'CT' else 2

        # 3. Frequency deviation bucket
        # Only look at the last 20 from end_idx
        sample_start = max(0, end_idx - 20)
        sample = recent_colors[sample_start:end_idx]
        c = Counter(sample)
        total = len(sample)
        t_freq = c.get('T', 0) / total
        ct_freq = c.get('CT', 0) / total
        dev = t_freq - ct_freq
        dev_bucket = 2 + int(np.clip(dev * 5, -2, 2))  # range 0-4

        # 4. Recent Entropy (High entropy = unpredictable, Low = pattern)
        import math
        entropy = 0
        for count in c.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        entropy_bucket = 0 if entropy < 1.0 else 1 if entropy < 1.4 else 2

        # 5. Rounds since last bonus (New context)
        bonus_since = 20
        for i in range(end_idx - 1, -1, -1):
            if recent_colors[i].upper() == 'BONUS':
                bonus_since = (end_idx - 1) - i
                break
        bonus_since_bucket = min(bonus_since // 5, 4) # 0, 1, 2, 3, 4+

        return (streak_bucket, last_color, dev_bucket, entropy_bucket, bonus_since_bucket)

    def choose_action(self, state: tuple) -> str:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_vals = self.q_table[state]
        return max(q_vals, key=q_vals.get)

    def _compute_reward(self, action: str, actual_color: str) -> float:
        """Compute reward for an action given the actual outcome."""
        if action == 'skip':
            return 0.0  # Neutral: let agent learn when to skip

        bet_color = action.replace('bet_', '').lower()
        actual_lower = actual_color.lower()
        if bet_color == actual_lower:
            return COLOR_ODDS[bet_color] - 1.0  # net profit
        else:
            return -1.0  # lost the bet

    def train(self, rolls: list[dict], episodes: int = 3):
        """Train by replaying historical data multiple times."""
        colors = [r['color'] for r in rolls]
        print(f"[RL Agent] Training on {len(colors)} rounds × {episodes} episodes...")

        for ep in range(episodes):
            total_reward = 0

            for i in range(20, len(colors) - 1):
                # Strict Continuity check: skip if there's a gap between i-1 and i
                if i > 0 and rolls[i]['round_id'] - rolls[i-1]['round_id'] > 1:
                    continue

                state = self._discretize_state(colors, end_idx=i)
                action = self.choose_action(state)
                actual = colors[i]
                reward = self._compute_reward(action, actual)

                next_state = self._discretize_state(colors, end_idx=i+1)
                best_next = max(self.q_table[next_state].values())

                # Q-learning update
                old_q = self.q_table[state][action]
                self.q_table[state][action] = old_q + self.alpha * (
                    reward + self.gamma * best_next - old_q
                )

                total_reward += reward

            # Decay epsilon each episode
            self.epsilon = max(0.01, self.epsilon * 0.95)
            print(f"  Episode {ep+1}/{episodes}: total_reward={total_reward:.1f}, "
                  f"states={len(self.q_table)}, epsilon={self.epsilon:.3f}")

        self.trained = True

    def predict(self, recent_colors: list[str]) -> dict:
        """
        Get action recommendation and convert to probability distribution.
        Returns {T: float, CT: float, Bonus: float}
        """
        if not self.trained or len(recent_colors) < 10:
            return {'T': 7/15, 'CT': 7/15, 'Bonus': 1/15}

        state = self._discretize_state(recent_colors)
        q_vals = self.q_table[state]

        # Convert Q-values to softmax probabilities (excluding skip)
        bet_actions = {a: q_vals[a] for a in ['bet_t', 'bet_ct', 'bet_bonus']}

        # Softmax with standard temperature (1.0) for better calibration (Item 2 from note 3)
        max_q = max(bet_actions.values())
        exp_vals = {a: math.exp((v - max_q) / 1.0) for a, v in bet_actions.items()}
        total = sum(exp_vals.values())

        probs = {
            'T': exp_vals['bet_t'] / total,
            'CT': exp_vals['bet_ct'] / total,
            'Bonus': exp_vals['bet_bonus'] / total,
        }
        return probs

    def should_skip(self, recent_colors: list[str]) -> bool:
        """Check if RL agent recommends skipping this round."""
        if not self.trained:
            return False
        state = self._discretize_state(recent_colors)
        action = max(self.q_table[state], key=self.q_table[state].get)
        return action == 'skip'

    def update(self, recent_colors: list[str], action_taken: str, actual_color: str):
        """Online update after a single round."""
        state = self._discretize_state(recent_colors[:-1])
        reward = self._compute_reward(action_taken, actual_color)
        next_state = self._discretize_state(recent_colors)
        best_next = max(self.q_table[next_state].values())

        old_q = self.q_table[state][action_taken]
        self.q_table[state][action_taken] = old_q + self.alpha * (
            reward + self.gamma * best_next - old_q
        )

    def save(self, path: str = None):
        path = path or SAVE_PATH
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"[RL Agent] Q-table saved ({len(self.q_table)} states)")

    def load(self, path: str = None) -> bool:
        path = path or SAVE_PATH
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
        self.q_table.update(data)
        self.trained = True
        print(f"[RL Agent] Loaded {len(self.q_table)} states")
        return True
