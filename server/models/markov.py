"""
markov.py - Markov Chain model (Orders 1, 2, 3)
Transition matrix predict next color từ previous N colors
"""
import pickle
import os
from collections import defaultdict

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'saved', 'markov.pkl')


class MarkovChain:
    def __init__(self):
        # Transition tables for order 1, 2, 3
        self.transitions = {
            1: defaultdict(lambda: defaultdict(int)),
            2: defaultdict(lambda: defaultdict(int)),
            3: defaultdict(lambda: defaultdict(int)),
        }
        self.trained = False
        self.min_counts = {1: 10, 2: 50, 3: 30}

    def train(self, colors: list[str]):
        """Train on a sequence of colors ['T', 'CT', 'Bonus', ...]."""
        self.transitions = {
            1: defaultdict(lambda: defaultdict(int)),
            2: defaultdict(lambda: defaultdict(int)),
            3: defaultdict(lambda: defaultdict(int)),
        }

        for i in range(len(colors)):
            # Order 1
            if i >= 1:
                state = colors[i - 1]
                self.transitions[1][state][colors[i]] += 1

            # Order 2
            if i >= 2:
                state = f"{colors[i-2]}|{colors[i-1]}"
                self.transitions[2][state][colors[i]] += 1

            # Order 3
            if i >= 3:
                state = f"{colors[i-3]}|{colors[i-2]}|{colors[i-1]}"
                self.transitions[3][state][colors[i]] += 1

        self.trained = True
        total_transitions = sum(
            sum(v.values()) for d in self.transitions.values() for v in d.values()
        )
        print(f"[Markov] Trained on {len(colors)} rounds, {total_transitions} transitions")

    def predict(self, recent_colors: list[str], smoothing: float = 1.0) -> dict:
        """
        Predict next color probabilities.
        Uses highest available order with sufficient data, falls back to lower.
        Returns {T: float, CT: float, Bonus: float}
        """
        if not self.trained or len(recent_colors) == 0:
            return {'T': 7/15, 'CT': 7/15, 'Bonus': 1/15}

        all_colors = ['T', 'CT', 'Bonus']
        best_probs = None
        best_order = 0

        for order in [3, 2, 1]:
            if len(recent_colors) < order:
                continue

            if order == 1:
                state = recent_colors[-1]
            elif order == 2:
                state = f"{recent_colors[-2]}|{recent_colors[-1]}"
            else:
                state = f"{recent_colors[-3]}|{recent_colors[-2]}|{recent_colors[-1]}"

            counts = self.transitions[order].get(state)
            if counts is None:
                continue

            total = sum(counts.values())
            if total < self.min_counts[order]:
                continue

            # Laplace smoothing
            probs = {}
            for c in all_colors:
                probs[c] = (counts.get(c, 0) + smoothing) / (total + smoothing * len(all_colors))

            best_probs = probs
            best_order = order

        if best_probs is None:
            return {'T': 7/15, 'CT': 7/15, 'Bonus': 1/15}

        return best_probs

    def update(self, recent_colors: list[str], new_color: str):
        """Online update: add a single new transition."""
        if len(recent_colors) >= 1:
            self.transitions[1][recent_colors[-1]][new_color] += 1
        if len(recent_colors) >= 2:
            state = f"{recent_colors[-2]}|{recent_colors[-1]}"
            self.transitions[2][state][new_color] += 1
        if len(recent_colors) >= 3:
            state = f"{recent_colors[-3]}|{recent_colors[-2]}|{recent_colors[-1]}"
            self.transitions[3][state][new_color] += 1

    def save(self, path: str = None):
        path = path or SAVE_PATH
        data = {
            'transitions': {k: dict(v) for k, v in self.transitions.items()},
        }
        # Convert inner defaultdicts to regular dicts
        for order in data['transitions']:
            data['transitions'][order] = {
                state: dict(counts)
                for state, counts in data['transitions'][order].items()
            }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[Markov] Saved to {path}")

    def load(self, path: str = None) -> bool:
        path = path or SAVE_PATH
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.transitions = {1: defaultdict(lambda: defaultdict(int)),
                            2: defaultdict(lambda: defaultdict(int)),
                            3: defaultdict(lambda: defaultdict(int))}
        for order, states in data['transitions'].items():
            for state, counts in states.items():
                for color, count in counts.items():
                    self.transitions[int(order)][state][color] = count

        self.trained = True
        print(f"[Markov] Loaded from {path}")
        return True

    def get_stats(self) -> dict:
        stats = {}
        for order in [1, 2, 3]:
            n_states = len(self.transitions[order])
            total = sum(sum(v.values()) for v in self.transitions[order].values())
            stats[f'order_{order}'] = {'states': n_states, 'transitions': total}
        return stats
