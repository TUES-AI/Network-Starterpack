from backprop import train, evaluate
from sklearn.datasets import make_circles
import numpy as np

X, y = make_circles(n_samples=400, noise=0.1, factor=0.3, random_state=0)
for i in range(len(y)):
    print(f"X: {X[i]}, y: {y[i]}")
train_split = 0.8

def get_data(split, index):
    if split == "train":
        return X[index], y[index]
    else:
        return X[index + (len(X) * train_split)], y[index + (len(y) * train_split)]

def initialize_network_layer(layer1, layer2):
    weights = [[np.random.randint(-100, 100) for _ in range(layer2)] for _ in range(layer1)]
    biases = [0 for _ in range(layer2)]
    return {"weights": weights, "biases": biases}

network_size = [2, 20, 10, 2]
network = [initialize_network_layer(network_size[i], network_size[i + 1]) for i in range(len(network_size) - 1)]

data_size = int(len(X) * train_split)
print("train_split:", train_split, "data_size:", data_size)
network, metrics = train(
    network,
    get_data,
    learning_rate=0.01,
    number_of_epochs=100,
    size_data_set=data_size,
    data_to_train_on=data_size,
    batch_size=data_size,
)

test_acc = evaluate(network, get_data, size_data_set=int(len(X) - data_size), split="test")
print("metrics:", metrics)
print("test_acc:", test_acc)
