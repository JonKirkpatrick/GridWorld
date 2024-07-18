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
        self.action_probs = [0.25, 0.25, 0.25, 0.25]
        self.gamma = 0.95
        self.epsilon = 0.1

    def get_next_state_and_reward(self, state, action):
        # self.permute_squares()
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

    def random_action(self):
        return random.choice(self.actions)

    def epsilon_greedy(self, policy, state):
        if np.random.rand() < self.epsilon:
            return self.random_action()
        else:
            return policy[state]

    def epsilon_soft_action(self, policy, state):
        actions = list(policy[state].keys())
        probabilities = list(policy[state].values())
        return np.random.choice(actions, p=probabilities)

    def permute_squares(self):
        if random.random() < 0.1:  # 10% chance to permute
            self.special_states['blue'], self.special_states['green'] = (
                self.special_states['green'], self.special_states['blue']
            )

def extract_value_function(Q):
    V = {}
    for state in Q.keys():
        V[state] = max(Q[state].values())  # Take the maximum Q value for each state
    return V

def extract_policy(Q, gridworld):
    policy = {}
    for state in Q.keys():
        action_values = Q[state]
        best_action = max(action_values, key=action_values.get)
        policy[state] = best_action
    return policy

def print_policy(policy):
    # Create a grid for output
    policy_grid = np.full((gridworld.rows, gridworld.cols), '', dtype=object)
    
    for state, action in policy.items():
        policy_grid[state] = action
    
    # Print the grid
    for row in policy_grid:
        print(row)

# Monte Carlo method with exploring starts
def monte_carlo_exploring_starts(gridworld, episodes):
    # Initialize Q for state-action pairs
    Q = {(i, j): {a: 0 for a in gridworld.actions} for i in range(gridworld.rows) for j in range(gridworld.cols)}
    policy = {state: gridworld.random_action() for state in np.ndindex(gridworld.grid.shape)}

    # Initialize returns for state-action pairs
    returns = {(i, j): {a: [] for a in gridworld.actions} for i in range(gridworld.rows) for j in range(gridworld.cols)}

    for episode in range(episodes):
        # Reset visited states for each episode
        visited_states = set()

        # Start at a random non-terminal state
        state = (random.randint(0, gridworld.rows - 1), random.randint(0, gridworld.cols - 1))
        while gridworld.is_terminal(state):
            state = (random.randint(0, gridworld.rows - 1), random.randint(0, gridworld.cols - 1))

        action = policy[state]
        episode_log = []

        # Generate episode
        while not gridworld.is_terminal(state):
            next_state, reward = gridworld.get_next_state_and_reward(state, action)
            episode_log.append((state, action, reward))  # Include reward here
            state = next_state
            action = gridworld.epsilon_greedy(policy, state)

        # Calculate returns for first visits
        G = 0
        Gt = []
        for (state, action, reward) in reversed(episode_log):
            G = reward + gridworld.gamma * G
            Gt.insert(0, G)
        for i, (state, action, reward) in enumerate(episode_log):
            # Check for the first visit to the state-action pair in the reversed episode log
            if not any((state == s and action == a) for s, a, r in episode_log[:i]):
                returns[state][action].append(Gt[i])
                Q[state][action] = np.mean(returns[state][action])  # Update state-action value
                visited_states.add((state, action))  # Mark state-action pair as visited
                # Update the policy
                best_action = max(Q[state], key=Q[state].get)
                policy[state] = best_action

    return Q, policy

def monte_carlo_epsilon_soft(gridworld, episodes, start_state=(2, 2)):
    # Initialize Q for state-action pairs
    Q = {(i, j): {a: 0 for a in gridworld.actions} for i in range(gridworld.rows) for j in range(gridworld.cols)}
    returns = {(i, j): {a: [] for a in gridworld.actions} for i in range(gridworld.rows) for j in range(gridworld.cols)}

    # Start with an equiprobable policy
    policy = {state: {a: 1 / len(gridworld.actions) for a in gridworld.actions} for state in np.ndindex(gridworld.grid.shape)}

    for episode in range(episodes):
        # Reset visited states for each episode
        visited_states = set()
        
        state = start_state  # Always start from the same location
        episode_log = []

        # Generate episode
        while not gridworld.is_terminal(state):
            # Epsilon-soft action selection
            action = gridworld.epsilon_soft_action(policy, state)

            next_state, reward = gridworld.get_next_state_and_reward(state, action)
            episode_log.append((state, action, reward))  # Log the step
            state = next_state
            

        # Calculate returns for first visits
        G = 0
        for i, (state, action, reward) in enumerate(reversed(episode_log)):
            G = reward + gridworld.gamma * G
            if not any((state == s and action == a) for s, a, r in reversed(episode_log[i+1:])):
                returns[state][action].append(G)
                Q[state][action] = np.mean(returns[state][action])  # Update state-action value
                visited_states.add((state, action))  # Mark state-action pair as visited
                # Update policy using epsilon-soft
                best_action = max(Q[state], key=Q[state].get)
                for a in gridworld.actions:
                    if a == best_action:
                        policy[state][a] = 1 - gridworld.epsilon + gridworld.epsilon / len(gridworld.actions)
                    else:
                        policy[state][a] = gridworld.epsilon / len(gridworld.actions)
    return Q, policy


if __name__ == "__main__":
    gridworld = GridWorld()
    Q_MC_exploring_starts, policy_MC_exploring_starts = monte_carlo_exploring_starts(gridworld,10000)


    print("Policy for Monte Carlo with exploring starts.\n")
    print_policy(policy_MC_exploring_starts)
    V_MC_exploring_starts = extract_value_function(Q_MC_exploring_starts)
    print("Value function V_*:\n")
    print_policy(V_MC_exploring_starts)
    print()

    Q_MC_epsilon_soft, Policy_MC_epsilon_soft = monte_carlo_epsilon_soft(gridworld, 10000)
    optimal_policy = extract_policy(Q_MC_epsilon_soft, gridworld)

    print("Policy for Monte Carlo with epsilon soft.\n")
    print_policy(optimal_policy)
    V_MC_epsilon_soft = extract_value_function(Q_MC_epsilon_soft)
    print()
    print("Value function V_*:\n")
    print_policy(V_MC_epsilon_soft)    
