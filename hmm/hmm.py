import numpy as np
import time
import matplotlib.pyplot as plt
import random

choices = ['P', 'K', 'N']

occurance_matrix = np.array([[2, 4, 1], [0, 0, 4], [4, 1, 2]])

opponent_strategy = np.array([1/3, 1/3, 1/3])

kasa = []

stan_kasy = 0

iterarion_number = 3000
first_element_decision = random.random()
state = ''

if first_element_decision > 0.66:
    state = 'P'
elif first_element_decision < 0.33:
    state = 'N'
else:
    state = 'K'

for i in range(iterarion_number):
    print("Iteration ", i + 1)
    opponent_action = ''
    if state == 'P':
        prediction = np.random.choice(choices, p=occurance_matrix[0] / sum(occurance_matrix[0]))
        print("Prediction: \n" + prediction)
        print(occurance_matrix[0] / sum(occurance_matrix[0]))

        opponent_action = np.random.choice(choices, p=opponent_strategy)

        print("Opponent Action:", opponent_action)
        if opponent_action == 'P':
            occurance_matrix[0, 0] += 1
        elif opponent_action == 'K':
            occurance_matrix[0, 1] += 1
        else:
            occurance_matrix[0, 2] += 1

        if opponent_action == prediction:
            stan_kasy += 1
            kasa.append(stan_kasy)
        elif opponent_action == "N":
            stan_kasy -= 1
            kasa.append(stan_kasy)
        else:
            kasa.append(stan_kasy)
    elif state == 'K':
        prediction = np.random.choice(choices, p=occurance_matrix[1] / sum(occurance_matrix[1]))
        print("Prediction: \n" + prediction)
        print(occurance_matrix[1] / sum(occurance_matrix[1]))
        opponent_action = np.random.choice(choices, p=opponent_strategy)

        print("Opponent Action:", opponent_action)
        if opponent_action == 'P':
            occurance_matrix[1, 0] += 1
        elif opponent_action == 'K':
            occurance_matrix[1, 1] += 1
        else:
            occurance_matrix[1, 2] += 1

        if opponent_action == prediction:
            stan_kasy += 1
            kasa.append(stan_kasy)
        elif opponent_action == "P":
            stan_kasy -= 1
            kasa.append(stan_kasy)
        else:
            kasa.append(stan_kasy)
    elif state == 'N':
        prediction = np.random.choice(choices, p=occurance_matrix[2] / sum(occurance_matrix[2]))
        print("Prediction: \n" + prediction)
        print(occurance_matrix[2] / sum(occurance_matrix[2]))
        opponent_action = np.random.choice(choices, p=opponent_strategy)

        print("Opponent Action:", opponent_action)

        if opponent_action == 'P':
            occurance_matrix[2, 0] += 1
        elif opponent_action == 'K':
            occurance_matrix[2, 1] += 1
        else:
            occurance_matrix[2, 2] += 1

        if opponent_action == prediction:
            stan_kasy += 1
            kasa.append(stan_kasy)
        elif opponent_action == "K":
            stan_kasy -= 1
            kasa.append(stan_kasy)
        else:
            kasa.append(stan_kasy)

    state = opponent_action

    print("\n")


occurance_matrix
plt.plot(kasa)
plt.show()
print(len(kasa))
