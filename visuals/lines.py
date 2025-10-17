import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import yaml

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate linearly separable dataset
np.random.seed(config['training']['random_state'])
n_samples = config['training']['n_samples']

# Create two linearly separable clusters
X1 = np.random.randn(n_samples//2, 2) + [2, 2]
X2 = np.random.randn(n_samples//2, 2) + [-2, -2]
X = np.vstack([X1, X2])
y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])

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

# Create grid for visualization - lines dataset needs wider range
x_range = [-5, 5]
y_range = [-5, 5]
grid_res = config['visualization']['grid_resolution']

xx = np.linspace(x_range[0], x_range[1], grid_res)
yy = np.linspace(y_range[0], y_range[1], grid_res)
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
plt.title('MLP decision boundary on Lines Dataset')
plt.show()