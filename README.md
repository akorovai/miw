
# README

This README file provides an overview of the functionalities and usage of the modules included in this project.

## Libraries Used:

- `os`
- `argparse`
- `matplotlib`
- `numpy`
- `keras` (from TensorFlow)
- `tensorflow`
- `sklearn`

### Installation:

You can install the required libraries using pip. Run the following command in your terminal:

```
pip install keras tensorflow scikit-learn matplotlib
```

## Modules Overview:

1. **Autocoder Module**: 
    - Import: `from autocoder.autocoder import start_encoder`
    - Description: Contains functionality related to autoencoder training.

2. **Connecting Classifiers Module**:
    - Import: `from connecting_classifiers.index import evaluate_classification_models`
    - Description: Provides functions for evaluating classification models.

3. **Dense Network Module**: 
    - Import: `from dense_network.dense_network import start_network`
    - Description: Includes functionalities for training dense neural networks.

4. **HMM Module**: 
    - Import: `from hmm.hmm import simulate_game`
    - Description: Simulates a game and provides results.

5. **Linear Regression Module**: 
    - Import: `from linear_regression.regression import compare_models`
    - Description: Compares linear regression models.

6. **Multiclass Classifier Module**: 
    - Imports: 
        - `from multiclass_classifier.multiclass_classifier import classify`
        - `from multiclass_classifier.reglog import soft_max`
    - Description: Performs multiclass classification and softmax regression.

7. **CNN Module**: 
    - Import: `from cnn.CNN_example import train_and_evaluate_models`
    - Description: Includes functionalities for training and evaluating Convolutional Neural Network (CNN) models.

## Usage:

### Simulation of the Game:
```python
from hmm.hmm import simulate_game

occurance_matrix, kasa = simulate_game(3000)
print(f"Final occurrence matrix:\n{occurance_matrix}")
print(f"Length of kasa: {len(kasa)}")
```

### Comparison of Linear Regression Models:
```python
from linear_regression.regression import compare_models

compare_models('dane8.txt', test_size=0.3, random_state=1)
```

### Multiclass Classification:
```python
from multiclass_classifier.multiclass_classifier import classify

classify()
```

### Softmax Regression:
```python
from multiclass_classifier.reglog import soft_max

soft_max()
```

### Training Dense Neural Network:
```python
from dense_network.dense_network import start_network

start_network("dane8.txt", 1, 10, 1, 50000, 0.05)
```

### Training Autoencoder:
```python
from autocoder.autocoder import start_encoder

start_encoder(5, 64, 20, 256)
```

### Evaluation of Classification Models:
```python
from connecting_classifiers.index import evaluate_classification_models

evaluate_classification_models(10000, 0.4, 42, 0.2, 100, 1000)
```

### CNN Training and Evaluation Models:
```python
from cnn.CNN_example import train_and_evaluate_models

train_and_evaluate_models()
```

## Starting the Project:

To start the project, navigate to the directory containing the main.py file and execute it using the following command:

```
python main.py
```

Feel free to explore each module further for detailed usage and functionalities.
