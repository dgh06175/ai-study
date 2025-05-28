import numpy as np

states = ['S0', 'S1', 'S2', 'S3']
actions = ['A0', 'A1']
terminal_state = 3

transition = {
    0: {0: (0, 0), 1: (1, 0)},
    1: {0: (0, 0), 1: (2, 0)},
    2: {0: (1, 0), 1: (3, 1)},
    3: {0: (3, 0), 1: (3, 0)}
}

gamma = 0.9  # 할인율
theta = 1e-4 # 수렴 조건

V = np.zeros(len(states))

def value_iteration():
    while True:
        delta = 0
        for s in range(len(states)):
            if s == terminal_state:
                continue
            v = V[s]
            V[s] = max(
                sum([p * (r + gamma * V[s_]) for (s_, r), p in [(transition[s][actions.index(a)], 1.0)]])
                for a in actions
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    policy = {}
    for s in range(len(states)):
        if s == terminal_state:
            policy[s] = '-'
            continue
        best_action = None
        best_value = float('-inf')
        for a in actions:
            next_state, reward = transition[s][actions.index(a)]
            value = reward + gamma * V[next_state]
            if value > best_value:
                best_value = value
                best_action = a
        policy[s] = best_action
    return V, policy

optimal_values, optimal_policy = value_iteration()
print("Optimal Value Function:")
for i, v in enumerate(optimal_values):
    print(f"V({states[i]}) = {v:.4f}")

print("\nOptimal Policy:")
for s in range(len(states)):
    print(f"π({states[s]}) = {optimal_policy[s]}")
