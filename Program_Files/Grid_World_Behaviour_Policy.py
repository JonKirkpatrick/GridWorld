import numpy as np
import random
from collections import defaultdict

class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.grid = np.zeros((self.rows, self.cols))
        self.special_states = {
            'blue': (0, 1), 
            'green': (0, 4), 
            'red': (4, 2), 
            'yellow': (4, 4),
            'terminal1': (4, 0),
            'terminal2': (2, 4)
        }
        self.actions = ['up', 'down', 'left', 'right']
        self.gamma = 0.95
        self.alpha = 0.1

    def get_next_state_and_reward(self, state, action):
        x, y = state
        if action == 'up':
            next_state = (max(0, x - 1), y)
        elif action == 'down':
            next_state = (min(self.rows - 1, x + 1), y)
        elif action == 'left':
            next_state = (x, max(0, y - 1))
        elif action == 'right':
            next_state = (x, min(self.cols - 1, y + 1))
        else:
            raise ValueError(f"Invalid action: {action}")

        if next_state in [self.special_states['terminal1'], self.special_states['terminal2']]:
            return next_state, 0  # Terminal state
        if state == self.special_states['blue']:
            return self.special_states['red'], 5
        elif state == self.special_states['green']:
            next_state_idx = np.random.choice([0, 1])
            next_state = [self.special_states['yellow'], self.special_states['red']][next_state_idx]
            return next_state, 2.5
        elif next_state == state:
            return state, -0.5
        else:
            return next_state, -0.2  # Reward for white-to-white move

    def is_terminal(self, state):
        return state in [self.special_states['terminal1'], self.special_states['terminal2']]

    def get_actions(self):
        return self.actions

    def initialize_policy(self):
        policy = defaultdict(lambda: {action: 1/len(self.actions) for action in self.actions})
        return policy

    def initialize_Q(self):
        Q = defaultdict(lambda: {action: 0 for action in self.actions})
        return Q

    def softmax(self, Q_values):
        max_q = max(Q_values.values())  # for numerical stability
        exp_q = {a: np.exp(q - max_q) for a, q in Q_values.items()}
        sum_exp_q = sum(exp_q.values())
        return {a: exp_q[a] / sum_exp_q for a in Q_values}

    def permute_squares(self):
        if random.random() < 0.1:  # 10% chance to permute
            self.special_states['blue'], self.special_states['green'] = (
                self.special_states['green'], self.special_states['blue']
            )

    def monte_carlo_importance_sampling(self, num_episodes=20000):
        actions = self.get_actions()
        Q = self.initialize_Q()
        C = defaultdict(lambda: {action: 0 for action in self.actions})
        target_policy = self.initialize_policy()

        for episode_num in range(num_episodes):
            state = (np.random.randint(self.rows), np.random.randint(self.cols))
            episode = []

            while not self.is_terminal(state):
                action = np.random.choice(actions)
                next_state, reward = self.get_next_state_and_reward(state, action)
                episode.append((state, action, reward))
                state = next_state

            # Calculate the returns G for the episode
            G = []
            G_value = 0
            for state, action, reward in reversed(episode):
                G_value = reward + self.gamma * G_value
                G.insert(0, (state, action, G_value))

            # Calculate the cumulative importance ratios
            W = []
            importance_ratio = 1.0
            for state, action, _ in reversed(episode):
                # Probability of taking the action under the target policy
                target_prob = target_policy[state][action]
                behavior_prob = 1 / len(actions)  # Equiprobable behavior policy
                importance_ratio *= target_prob / behavior_prob
                W.insert(0, importance_ratio)

            # Update Q and C using the weights
            for i, (state, action, G_value) in enumerate(G):
                C[state][action] += W[i]
                Q[state][action] += (W[i] / C[state][action]) * (G_value - Q[state][action])

            # Update the target policy using softmax of Q
            for state in Q:
                softmax_prob = self.softmax(Q[state])
                for action in self.actions:
                    target_policy[state][action] = softmax_prob[action]

        return Q, target_policy

    def extract_deterministic_policy(self, target_policy):
        deterministic_policy = np.empty((self.rows, self.cols), dtype=object)
        action_map = { 'up': '^', 'down': 'v', 'left': '<', 'right': '>' }

        for i in range(self.rows):
            for j in range(self.cols):
                state = (i, j)
                best_action = max(target_policy[state], key=target_policy[state].get)
                deterministic_policy[i, j] = action_map[best_action]

        return deterministic_policy

if __name__ == "__main__":
    gridworld = GridWorld()
    Q_function, Target_Policy = gridworld.monte_carlo_importance_sampling()

    # Extract and print the deterministic policy
    Deterministic_Policy = gridworld.extract_deterministic_policy(Target_Policy)
    print("Deterministic Policy:")
    print(Deterministic_Policy)
