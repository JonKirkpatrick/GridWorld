# Gridworld.py

import numpy as np

class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.grid = np.zeros((self.rows, self.cols))
        self.special_states = {
            'blue': (0, 1), 
            'green': (0, 4), 
            'red': (3, 2), 
            'yellow': (4, 4)
        }
        self.actions = ['up', 'down', 'left', 'right']
        self.action_probs = [0.25, 0.25, 0.25, 0.25]
        self.gamma = 0.95

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
    
        if state == self.special_states['blue']:
            return self.special_states['red'], 5
        elif state == self.special_states['green']:
            next_state_idx = np.random.choice([0, 1])
            next_state = [self.special_states['yellow'], self.special_states['red']][next_state_idx]
            return next_state, 2.5
        elif next_state == state:
            return state, -0.5
        else:
            return next_state, 0



def solve_bellman_equations(gridworld):
    n_states = gridworld.rows * gridworld.cols
    V = np.zeros(n_states)
    R = np.zeros(n_states)
    P = np.zeros((n_states, n_states))

    for i in range(gridworld.rows):
        for j in range(gridworld.cols):
            state = i * gridworld.cols + j
            for action in gridworld.actions:
                next_state, reward = gridworld.get_next_state_and_reward((i, j), action)
                next_state_idx = next_state[0] * gridworld.cols + next_state[1]
                R[state] += gridworld.action_probs[gridworld.actions.index(action)] * reward
                P[state, next_state_idx] += gridworld.action_probs[gridworld.actions.index(action)]

    A = np.eye(n_states) - gridworld.gamma * P
    V = np.linalg.solve(A, R)
    return V.reshape((gridworld.rows, gridworld.cols))


def iterative_policy_evaluation(gridworld, theta=1e-3):
    V = np.zeros((gridworld.rows, gridworld.cols))
    iteration = 0
    while True:
        delta = 0
        for i in range(gridworld.rows):
            for j in range(gridworld.cols):
                state = (i, j)
                v = V[state]
                new_v = 0
                for action in gridworld.actions:
                    next_state, reward = gridworld.get_next_state_and_reward(state, action)
                    new_v += 0.25 * (reward + gridworld.gamma * V[next_state])
                V[state] = new_v
                delta = max(delta, abs(v - new_v))
        iteration += 1
        if delta < theta:
            break
    return V


def value_iteration(gridworld, theta=1e-6):
    V = np.zeros((gridworld.rows, gridworld.cols))
    iteration = 0
    while True:
        delta = 0
        for i in range(gridworld.rows):
            for j in range(gridworld.cols):
                state = (i, j)
                v = V[state]
                max_value = float('-inf')
                for action in gridworld.actions:
                    next_state, reward = gridworld.get_next_state_and_reward(state, action)
                    value = reward + gridworld.gamma * V[next_state]
                    if value > max_value:
                        max_value = value
                V[state] = max_value
                delta = max(delta, abs(v - max_value))
        iteration += 1
        if delta < theta:
            break
    return V

def solve_bellman_optimality_equation(gridworld):
    n_states = gridworld.rows * gridworld.cols
    V = np.zeros(n_states)
    R = np.zeros(n_states)
    P = np.zeros((n_states, n_states, len(gridworld.actions)))
    policy = np.zeros(n_states, dtype=int)

    action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    # Fill the reward and transition matrices
    for i in range(gridworld.rows):
        for j in range(gridworld.cols):
            state = i * gridworld.cols + j
            for action in gridworld.actions:
                next_state, reward = gridworld.get_next_state_and_reward((i, j), action)
                next_state_idx = next_state[0] * gridworld.cols + next_state[1]
                R[state] += gridworld.action_probs[gridworld.actions.index(action)] * reward
                P[state, next_state_idx, gridworld.actions.index(action)] += 1

    while True:
        V_new = np.zeros(n_states)
        for state in range(n_states):
            max_value = float('-inf')
            best_action = -1
            for a in range(len(gridworld.actions)):
                value = np.dot(P[state, :, a], V)
                if value > max_value:
                    max_value = value
                    best_action = a
            V_new[state] = R[state] + gridworld.gamma * max_value
            policy[state] = best_action

        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new

    # Map the numerical policy to action strings
    policy = np.vectorize(action_map.get)(policy)

    return V.reshape((gridworld.rows, gridworld.cols)), policy.reshape((gridworld.rows, gridworld.cols))

def policy_iteration(gridworld, theta=1e-6):
    policy = np.full((gridworld.rows, gridworld.cols), 'right')
    V = np.zeros((gridworld.rows, gridworld.cols))

    def policy_evaluation(policy, V):
        iteration = 0
        while True:
            delta = 0
            for i in range(gridworld.rows):
                for j in range(gridworld.cols):
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


def value_iteration_optimal_policy(gridworld, theta=1e-6):
    V = np.zeros((gridworld.rows, gridworld.cols))
    policy = np.full((gridworld.rows, gridworld.cols), 'right')
    iteration = 0
    
    while True:
        delta = 0
        for i in range(gridworld.rows):
            for j in range(gridworld.cols):
                state = (i, j)
                v = V[state]
                action_values = []

                for action in gridworld.actions:
                    next_state, reward = gridworld.get_next_state_and_reward(state, action)
                    action_value = reward + gridworld.gamma * V[next_state]
                    action_values.append(action_value)
                
                best_value = max(action_values)
                V[state] = best_value
                policy[state] = gridworld.actions[np.argmax(action_values)]
                delta = max(delta, abs(v - best_value))

        iteration += 1
        if delta < theta:
            break

    return policy, V



if __name__ == "__main__":
    gridworld = GridWorld()

    # Part 1: Estimate the value function
    print("Estimating Value Function by Solving Bellman Equations Explicitly...")
    V_explicit = solve_bellman_equations(gridworld)
    print("Done.\n")

    print("Estimating Value Function by Iterative Policy Evaluation...")
    V_iterative = iterative_policy_evaluation(gridworld)
    print("Done.\n")

    print("Estimating Value Function by Value Iteration...")
    V_value_iter = value_iteration(gridworld)
    print("Done.\n")

    # Part 2: Determine the optimal policy
    print("Determining Optimal Value Function by Solving Bellman Optimality Equation...")
    V_optimal, optimal_policy = solve_bellman_optimality_equation(gridworld)
    print("Done.\n")

    print("Determining Optimal Policy and Value Function by Policy Iteration...")
    policy_pi, V_pi = policy_iteration(gridworld)
    print("Done.\n")

    print("Determining Optimal Policy and Value Function by Value Iteration...")
    policy_vi, V_vi = value_iteration_optimal_policy(gridworld)
    print("Done.\n")

    print("Value Function by Solving Bellman Equations Explicitly:\n")
    print(V_explicit)
    print()

    print("Value Function by Iterative Policy Evaluation:\n")
    print(V_iterative)
    print()

    print("Value Function by Value Iteration:\n")
    print(V_value_iter)
    print()

    print("--------------------------------------------------------------\n")
    print("Optimal Policy and Value Function by Solving Bellman Optimality Equation:\n")
    print(optimal_policy)
    print()
    print(V_optimal)
    print()

    print("Optimal Policy and Value Function by Policy Iteration:\n")
    print(policy_pi)
    print()
    print(V_pi)
    print()

    print("Optimal Policy and Value Function by Value Iteration:\n")
    print(policy_vi)
    print()
    print(V_vi)
