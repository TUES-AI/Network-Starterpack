from backprop import train, evaluate
from sklearn.datasets import make_circles
import numpy as np

# Generate data once and split it
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=0)
split_index = int(0.8 * len(X))
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

def get_data(index, split):
    if split == "train":
        return X_train[index], y_train[index]
    else:
        return X_test[index], y_test[index]

def size(split, data_type):
    if split == "train":
        return len(X_train)
    else:
        return len(X_test)

def initialize_network_layer(layer1, layer2):
    weights = [[0 for _ in range(layer2)] for _ in range(layer1)]
    biases = [0 for _ in range(layer2)]
    return {"weights": weights, "biases": biases}

network = []
network_size = [2, 256, 128, 64, 2]  # 2 input features, 2 output classes for circles data
for i in range(1, len(network_size)):
    network.append(initialize_network_layer(network_size[i-1], network_size[i]))

test_acc = evaluate(network, get_data, size_data_set=size("test", "data"), split="test")
print("test_acc:", test_acc)

train_size = size("train", "data")
print("train_size:", train_size)
network, metrics = train(
    network,
    get_data,
    learning_rate=0.1,
    number_of_epochs=10000,
    data_to_train_on=min(60_000, train_size),
    size_data_set=train_size,
    activation_function="relu",
    batch_size=min(60_000, train_size),
    split="train",
)

test_acc = evaluate(network, get_data, size_data_set=size("test", "data"), split="test")
print("metrics:", metrics)
print("test_acc:", test_acc)
