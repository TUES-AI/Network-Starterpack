import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as data
from sklearn.neural_network import MLPClassifier

# X, y = data.make_circles(n_samples=400, noise=0.1, factor=0.3, random_state=0)
X, y = data.make_moons(n_samples=400, noise=0.1, random_state=0)

clf = MLPClassifier(hidden_layer_sizes=(5,10), activation='relu',
                   solver='adam', batch_size=400,
                    max_iter=2000, random_state=0)
clf.fit(X, y)

xx = np.linspace(-1.5, 1.5, 400)
yy = np.linspace(-1.5, 1.5, 400)
XX, YY = np.meshgrid(xx, yy)
grid = np.c_[XX.ravel(), YY.ravel()]
Z = clf.predict_proba(grid)[:,1].reshape(XX.shape)

plt.figure(figsize=(6,5))
plt.contourf(XX, YY, Z, levels=20, cmap='RdYlBu', alpha=0.4)
plt.contour(XX, YY, Z, levels=[0.5], colors='purple', linewidths=2)
plt.scatter(X[y==0,0], X[y==0,1], marker='x', label='class 0')
plt.scatter(X[y==1,0], X[y==1,1], marker='o', label='class 1', edgecolors='k')
plt.legend()
plt.title('MLP decision boundary on Circles Dataset')
plt.show()
