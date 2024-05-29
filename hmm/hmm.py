import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple


def update_occurance_matrix(matrix: np.ndarray, state_idx: int, action: str) -> None:
    action_idx = {'P': 0, 'K': 1, 'N': 2}[action]
    matrix[state_idx, action_idx] += 1


def simulate_game(iteration_number: int = 3000) -> Tuple[np.ndarray, List[int]]:
    choices = ['P', 'K', 'N']
    occurance_matrix = np.array([[2, 4, 1], [0, 0, 4], [4, 1, 2]])
    opponent_strategy = np.array([1 / 3, 1 / 3, 1 / 3])
    kasa = []
    stan_kasy = 0

    state = random.choice(choices)

    for _ in range(iteration_number):
        state_idx = choices.index(state)
        prediction = np.random.choice(choices, p=occurance_matrix[state_idx] / sum(occurance_matrix[state_idx]))
        opponent_action = np.random.choice(choices, p=opponent_strategy)

        update_occurance_matrix(occurance_matrix, state_idx, opponent_action)

        if opponent_action == prediction:
            stan_kasy += 1
        elif (state == 'P' and opponent_action == 'N') or (state == 'N' and opponent_action == 'K') or (
                state == 'K' and opponent_action == 'P'):
            stan_kasy -= 1

        kasa.append(stan_kasy)
        state = opponent_action

    plt.plot(kasa)
    plt.xlabel('Iteration')
    plt.ylabel('Stan kasy')
    plt.title('Stan kasy w czasie')
    plt.show()

    return occurance_matrix, kasa


if __name__ == "__main__":
    simulate_game()
