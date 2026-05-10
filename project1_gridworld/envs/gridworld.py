import numpy as np
from typing import Tuple, Optional


class GridWorld:
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
    ACTION_NAMES = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    def __init__(self, rows: int = 5, cols: int = 5, start: Tuple[int, int] = (0, 0),
                 goal: Tuple[int, int] = (4, 4), obstacles: Optional[list] = None,
                 step_reward: float = -1.0, goal_reward: float = 0.0,
                 obstacle_reward: float = -10.0, stochastic: float = 0.0):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.obstacles = obstacles or []
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.obstacle_reward = obstacle_reward
        self.stochastic = stochastic
        self.n_states = rows * cols
        self.n_actions = 4
        self.state = start
        self._build_transition_model()

    def _build_transition_model(self):
        self.P = {}
        for r in range(self.rows):
            for c in range(self.cols):
                s = self._to_index(r, c)
                self.P[s] = {}
                for a in range(self.n_actions):
                    self.P[s][a] = self._get_transitions(r, c, a)

    def _get_transitions(self, r, c, a):
        transitions = []
        if (r, c) == self.goal:
            return [(1.0, self._to_index(r, c), 0.0, True)]

        for action in range(self.n_actions):
            if action == a:
                prob = 1.0 - self.stochastic
            else:
                prob = self.stochastic / (self.n_actions - 1)
            if prob == 0:
                continue
            nr, nc = r + self.ACTIONS[action][0], c + self.ACTIONS[action][1]
            if not self._in_bounds(nr, nc):
                nr, nc = r, c
            next_s = self._to_index(nr, nc)
            reward = self.goal_reward if (nr, nc) == self.goal else (
                self.obstacle_reward if (nr, nc) in self.obstacles else self.step_reward)
            done = (nr, nc) == self.goal
            transitions.append((prob, next_s, reward, done))
        return transitions

    def _in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _to_index(self, r, c):
        return r * self.cols + c

    def _to_coord(self, s):
        return s // self.cols, s % self.cols

    def reset(self):
        self.state = self.start
        return self._to_index(*self.state)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        transitions = self.P[self._to_index(*self.state)][action]
        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        _, next_s, reward, done = transitions[idx]
        self.state = self._to_coord(next_s)
        return next_s, reward, done, {}

    def get_state_features(self, s: int) -> np.ndarray:
        r, c = self._to_coord(s)
        return np.array([r / self.rows, c / self.cols, 
                         abs(r - self.goal[0]) / self.rows,
                         abs(c - self.goal[1]) / self.cols], dtype=np.float32)
