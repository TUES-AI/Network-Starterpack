from backprop import train
from sklearn.datasets import load_digits

data, labels = load_digits().images, load_digits().target

print("data shape:", data.shape, " labels shape:", labels.shape)

print(data[0].shape)

train_split = 0.8
train_data_size = int(len(data) * train_split)
test_data_size = int(len(data) - train_data_size)

def get_data(split, index):
    if split == "train":
        return data[index], labels[index]
    else:
        return data[index + train_data_size], labels[index + train_data_size]

print(get_data("train", 0))

def initialize_network_layer(layer1, layer2):
    weights = [[np.random.randint(-100, 100) for _ in range(layer2)] for _ in range(layer1)]
    biases = [0 for _ in range(layer2)]
    return {"weights": weights, "biases": biases}


