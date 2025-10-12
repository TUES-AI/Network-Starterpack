# train_demo.py
# assumes backprop.py and my_data_loader.py are in the same folder

from backprop import train, evaluate
from my_data_loader import prepare_data, get_data, size

# 1) Prepare data (pick one)
prepare_data(dataset="mnist", dir="data")   # downloads MNIST
# prepare_data(dataset="blobs", dir="data") # or synthetic fallback

# 2) Build a dumb network like your snippet
def initialize_network_layer(layer1, layer2):
    weights = [[0 for _ in range(layer2)] for _ in range(layer1)]
    biases = [0 for _ in range(layer2)]
    return {"weights": weights, "biases": biases}

network = []
network_size = [784, 256, 128, 64, 10]
for i in range(1, len(network_size)):
    network.append(initialize_network_layer(network_size[i-1], network_size[i]))

# 3) Train
train_size = size("train", "data")
network, metrics = train(
    network,
    get_data,
    learning_rate=0.15,
    number_of_epochs=10000,
    data_to_train_on=min(60_000, train_size),  # e.g. use a subset
    size_data_set=train_size,
    activation_function="relu",
    batch_size=128,
    split="train",
)

# 4) Evaluate
test_acc = evaluate(network, get_data, size_data_set=size("test", "data"), split="test")
print("metrics:", metrics)
print("test_acc:", test_acc)

