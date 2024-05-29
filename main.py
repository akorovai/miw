from autocoder.autocoder import start_encoder
from connecting_classifiers.index import evaluate_classification_models
from dense_network.dense_network import start_network
from hmm.hmm import simulate_game
from liniear_regression.regression import compare_models
from multiclass_classifier.multiclass_classifier import classify
from multiclass_classifier.reglog import soft_max

# Simulate game and print results
print("Simulation of the game:")
occurance_matrix, kasa = simulate_game(3000)
print(f"Final occurrence matrix:\n{occurance_matrix}")
print(f"Length of kasa: {len(kasa)}")
print()

# Compare linear regression models
print("Comparison of linear regression models:")
compare_models('dane8.txt', test_size=0.3, random_state=1)
print()

# Multiclass classification
print("Multiclass Classification:")
classify()
print()

# Softmax regression
print("Softmax Regression:")
soft_max()
print()

# Dense network training
print("Training Dense Neural Network:")
start_network("dane8.txt", 1, 10, 1, 50000, 0.05)
print()

# Autoencoder training
print("Training Autoencoder:")
start_encoder(5, 64, 20, 256)

print("Evaluation Classification models: ")
evaluate_classification_models(10000, 0.4, 42, 0.2, 100, 1000)


from cnn.CNN_example import train_and_evaluate_models

print("CNN training and evaluation models:")
train_and_evaluate_models()
