import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as data
from sklearn.neural_network import MLPClassifier
import yaml

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate moons dataset
X, y = data.make_moons(
    n_samples=config['training']['n_samples'],
    noise=0.1,
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

# Create grid for visualization - moons need slightly different range
x_range = [-1.5, 2.5]  # Adjusted for moons dataset
y_range = [-1.0, 1.5]  # Adjusted for moons dataset
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
plt.title('MLP decision boundary on Moons Dataset')
plt.show()