import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as data
from sklearn.neural_network import MLPClassifier
import yaml

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate blobs dataset
X, y = data.make_blobs(
    n_samples=config['training']['n_samples'],
    centers=2,
    cluster_std=1.0,
    random_state=config['training']['random_state']
)

# Create classifier with shared configuration
clf = MLPClassifier(
    hidden_layer_sizes=tuple(config['network']['hidden_layer_sizes']),
    activation=config['network']['activation'],
    solver=config['network']['solver'],
    batch_size=config['network']['batch_size'],
    max_iter=config['network']['max_iter'],
    random_state=config['network']['random_state']
)
clf.fit(X, y)

# Create grid for visualization - blobs can have wider range
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
grid_res = config['visualization']['grid_resolution']

xx = np.linspace(x_min, x_max, grid_res)
yy = np.linspace(y_min, y_max, grid_res)
XX, YY = np.meshgrid(xx, yy)
grid = np.c_[XX.ravel(), YY.ravel()]
Z = clf.predict_proba(grid)[:,1].reshape(XX.shape)

# Plot results
plt.figure(figsize=(6,5))
plt.contourf(XX, YY, Z, levels=20, cmap='RdYlBu', alpha=0.4)
plt.contour(XX, YY, Z, levels=[0.5], colors='purple', linewidths=2)
plt.scatter(X[y==0,0], X[y==0,1], marker='x', label='class 0')
plt.scatter(X[y==1,0], X[y==1,1], marker='o', label='class 1', edgecolors='k')
plt.legend()
plt.title('MLP decision boundary on Blobs Dataset')
plt.show()