# Gridworld.py

import numpy as np
import random

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
        self.action_probs = [0.25, 0.25, 0.25, 0.25]

    def get_next_state_and_reward(self, state, action):
        x, y = state
        if state in [self.special_states['terminal1'], self.special_states['terminal2']]:
            next_state = state
        elif action == 'up':
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

    def permute_squares(self):
        if random.random() < 0.1:  # 10% chance to permute
            self.special_states['blue'], self.special_states['green'] = (
                self.special_states['green'], self.special_states['blue']
            )

    def is_terminal(self, state):
        return state in [self.special_states['terminal1'], self.special_states['terminal2']]


def policy_iteration(gridworld, theta=1e-6):
    policy = np.full((gridworld.rows, gridworld.cols), 'right')
    V = np.zeros((gridworld.rows, gridworld.cols))

    def policy_evaluation(policy, V):
        iteration = 0
        while True:
            delta = 0
            for i in range(gridworld.rows):
                for j in range(gridworld.cols):
                    gridworld.permute_squares()
                    state = (i, j)
                    v = V[state]
                    new_v = 0
                    for action in gridworld.actions:
                        next_state, reward = gridworld.get_next_state_and_reward(state, policy[state])
                        new_v += 0.25 * (reward + gridworld.gamma * V[next_state])
                    V[state] = new_v
                    delta = max(delta, abs(v - new_v))
            iteration += 1
            if delta < theta:
                break
        return V

    def policy_improvement(policy, V):
        policy_stable = True
        for i in range(gridworld.rows):
            for j in range(gridworld.cols):
                state = (i, j)
                old_action = policy[state]
                action_values = []
                for action in gridworld.actions:
                    next_state, reward = gridworld.get_next_state_and_reward(state, action)
                    action_values.append(reward + gridworld.gamma * V[next_state])
                best_action = gridworld.actions[np.argmax(action_values)]
                if old_action != best_action:
                    policy_stable = False
                policy[state] = best_action
        return policy, policy_stable

    while True:
        V = policy_evaluation(policy, V)
        policy, policy_stable = policy_improvement(policy, V)
        if policy_stable:
            break

    return policy, V


if __name__ == "__main__":
    gridworld = GridWorld()

    # Part 2: Determine the optimal policy
    print("Determining Optimal Policy and Value Function by Policy Iteration...")
    policy_pi, V_pi = policy_iteration(gridworld)
    print("Done.\n")

    print("Optimal Policy and Value Function by Policy Iteration:\n")
    print(policy_pi)
    print()
    print(V_pi)
    print()
