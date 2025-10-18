"""
Enhanced interactive MLP visualization module with click-to-add data points and performance optimizations.
Allows users to add Class 0 (left click) and Class 1 (right click) points to test model generalization.
"""
import numpy as np
# Use ipympl so clicks, TextBox, and Button work in Colab
import os, matplotlib
os.environ.setdefault("MPLBACKEND", "module://ipympl.backend_nbagg")
try:
    matplotlib.use("module://ipympl.backend_nbagg")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from sklearn.neural_network import MLPClassifier
import yaml
import time



def create_interactive_mlp_visualization(X, y, title, x_range=None, y_range=None, grid_resolution=400):
    """
    Create an interactive MLP visualization with click-to-add data points.

    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Original input features
    y : array-like, shape (n_samples,)
        Original target labels
    title : str
        Plot title (not displayed, kept for compatibility)
    x_range : tuple, optional
        X-axis range for visualization grid (min, max)
    y_range : tuple, optional
        Y-axis range for visualization grid (min, max)
    grid_resolution : int
        Resolution for the visualization grid

    Returns:
    --------
    None
    """

    # Load shared configuration
    with open('Config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Create figure with interactive controls
    fig = plt.figure(figsize=(8, 7))
    ax = plt.axes([0.1, 0.3, 0.8, 0.65])  # More space for main plot

    # Create input fields - moved to the right for better visibility
    ax_epoch = plt.axes([0.15, 0.15, 0.2, 0.04])
    ax_lr = plt.axes([0.6, 0.15, 0.2, 0.04])  # Moved further right for better centering
    ax_network = plt.axes([0.15, 0.08, 0.5, 0.04])
    ax_button = plt.axes([0.75, 0.08, 0.2, 0.04])
    ax_clear = plt.axes([0.15, 0.02, 0.3, 0.04])

    epoch_field = widgets.TextBox(ax_epoch, 'Epochs:', initial=str(config['network']['max_iter']))
    lr_field = widgets.TextBox(ax_lr, 'Learning Rate:', initial='0.001')
    network_field = widgets.TextBox(ax_network, 'Network Size:', initial=' '.join(map(str, config['network']['hidden_layer_sizes'])))
    train_button = widgets.Button(ax_button, 'TRAIN')
    clear_button = widgets.Button(ax_clear, 'CLEAR ADDED POINTS')

    # Create grid for visualization
    if x_range is None:
        x_range = [X[:, 0].min() - 1, X[:, 0].max() + 1]
    if y_range is None:
        y_range = [X[:, 1].min() - 1, X[:, 1].max() + 1]

    xx = np.linspace(x_range[0], x_range[1], grid_resolution)
    yy = np.linspace(y_range[0], y_range[1], grid_resolution)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]

    # Global variables for the classifier and plot elements
    clf = None
    contourf = None
    contour = None

    # Store original data
    X_original = X.copy()
    y_original = y.copy()

    # Store user-added points
    X_user_class0 = []
    X_user_class1 = []

    # Scatter plot elements
    scatter0_original = None
    scatter1_original = None
    scatter0_user = None
    scatter1_user = None

    # Performance optimization: debounced retraining
    last_retrain_time = 0
    retrain_pending = False

    def debounced_train_and_plot():
        """Debounced version of train_and_plot to prevent lag"""
        nonlocal last_retrain_time, retrain_pending

        current_time = time.time()
        if current_time - last_retrain_time < 0.1:  # 0.1 second delay
            retrain_pending = True
            return

        train_and_plot()
        last_retrain_time = current_time
        retrain_pending = False

    def train_and_plot(event=None):
        nonlocal clf, contourf, contour, scatter0_original, scatter1_original, scatter0_user, scatter1_user

        # Get values from input fields
        try:
            epochs = int(epoch_field.text)
            learning_rate = float(lr_field.text)
            network_sizes = [int(x) for x in network_field.text.split()]
        except ValueError:
            print("Invalid input values. Using defaults.")
            epochs = config['network']['max_iter']
            learning_rate = 0.001
            network_sizes = config['network']['hidden_layer_sizes']

        # Combine original and user-added data
        X_combined = X_original.copy()
        y_combined = y_original.copy()

        if X_user_class0:
            X_combined = np.vstack([X_combined, np.array(X_user_class0)])
            y_combined = np.hstack([y_combined, np.zeros(len(X_user_class0))])

        if X_user_class1:
            X_combined = np.vstack([X_combined, np.array(X_user_class1)])
            y_combined = np.hstack([y_combined, np.ones(len(X_user_class1))])

        # Create classifier with custom parameters
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(network_sizes),
            activation=config['network']['activation'],
            solver=config['network']['solver'],
            batch_size=config['network']['batch_size'],
            max_iter=epochs,
            learning_rate_init=learning_rate,
            random_state=config['network']['random_state']
        )

        # Train the model
        clf.fit(X_combined, y_combined)

        # Predict on grid
        Z = clf.predict_proba(grid)[:,1].reshape(XX.shape)

        # Clear previous plot elements
        if contourf:
            contourf.remove()
        if contour:
            contour.remove()
        if scatter0_original:
            scatter0_original.remove()
        if scatter1_original:
            scatter1_original.remove()
        if scatter0_user:
            scatter0_user.remove()
        if scatter1_user:
            scatter1_user.remove()

        # Create new plot elements
        contourf = ax.contourf(XX, YY, Z, levels=20, cmap='RdYlBu', alpha=0.4)
        contour = ax.contour(XX, YY, Z, levels=[0.5], colors='purple', linewidths=2)

        # Plot original data points
        scatter0_original = ax.scatter(X_original[y_original==0,0], X_original[y_original==0,1],
                                      marker='o', label='Class 0 (original)', color='red', alpha=0.7)
        scatter1_original = ax.scatter(X_original[y_original==1,0], X_original[y_original==1,1],
                                      marker='x', label='Class 1 (original)', color='blue', alpha=0.7)

        # Plot user-added points
        if X_user_class0:
            scatter0_user = ax.scatter(np.array(X_user_class0)[:,0], np.array(X_user_class0)[:,1],
                                      marker='o', label='Class 0 (added)', color='red', s=100, edgecolors='black')
        if X_user_class1:
            scatter1_user = ax.scatter(np.array(X_user_class1)[:,0], np.array(X_user_class1)[:,1],
                                      marker='x', label='Class 1 (added)', color='blue', s=100, linewidths=2)

        ax.legend()
        plt.draw()

    def on_click(event):
        if event.inaxes == ax:
            # Left click → Class 0 (unless Shift is held)
            if event.button == 1 and not (event.key and "shift" in str(event.key).lower()):
                X_user_class0.append([event.xdata, event.ydata])
                print(f'Added Class 0 point at ({event.xdata:.2f}, {event.ydata:.2f})')
            # Right click OR Shift+Left click → Class 1 (works even if browser blocks right-click)
            elif event.button == 3 or (event.button == 1 and event.key and "shift" in str(event.key).lower()):
                X_user_class1.append([event.xdata, event.ydata])
                print(f'Added Class 1 point at ({event.xdata:.2f}, {event.ydata:.2f})')

            debounced_train_and_plot()

    def clear_user_points(event):
        nonlocal X_user_class0, X_user_class1
        X_user_class0.clear()
        X_user_class1.clear()
        print("Cleared all user-added points")
        train_and_plot()

    # Connect the button to the training function
    train_button.on_clicked(train_and_plot)
    clear_button.on_clicked(clear_user_points)

    # Connect mouse click events
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Initial training and plot
    train_and_plot()

    plt.show()


def create_interactive_mlp_visualization_with_config(X, y, title, config_overrides=None):
    """
    Create an interactive MLP visualization with click-to-add using configuration from Config.yml.

    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Input features
    y : array-like, shape (n_samples,)
        Target labels
    title : str
        Plot title
    config_overrides : dict, optional
        Override specific configuration values

    Returns:
    --------
    None
    """

    # Load shared configuration
    with open('Config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if key in config['visualization']:
                config['visualization'][key] = value

    x_range = config['visualization']['x_range']
    y_range = config['visualization']['y_range']
    grid_resolution = config['visualization']['grid_resolution']

    create_interactive_mlp_visualization(
        X, y, title, x_range, y_range, grid_resolution
    )

    # Print interactive instructions
    print("\n=== INTERACTIVE MODE ACTIVATED ===")
    print("Left click: Add Class 0 point (red circles)")
    print("Right click: Add Class 1 point (blue crosses)")
    print("Model retrains automatically when points are added")
    print("Use 'CLEAR ADDED POINTS' to reset user additions")
