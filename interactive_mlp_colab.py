"""
Enhanced interactive MLP visualization module for Google Colab.
Allows users to add Class 0 (left click) and Class 1 (right click) points to test model generalization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import yaml
from IPython.display import display, clear_output
import ipywidgets as widgets
from ipywidgets import HBox, VBox


def create_interactive_mlp_visualization(X, y, title, x_range=None, y_range=None, grid_resolution=400):
    """
    Create an interactive MLP visualization with click-to-add data points for Google Colab.

    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Original input features
    y : array-like, shape (n_samples,)
        Original target labels
    title : str
        Plot title
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

    # Create grid for visualization
    if x_range is None:
        x_range = [X[:, 0].min() - 1, X[:, 0].max() + 1]
    if y_range is None:
        y_range = [X[:, 1].min() - 1, X[:, 1].max() + 1]

    xx = np.linspace(x_range[0], x_range[1], grid_resolution)
    yy = np.linspace(y_range[0], y_range[1], grid_resolution)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]

    # Store original data
    X_original = X.copy()
    y_original = y.copy()

    # Store user-added points
    X_user_class0 = []
    X_user_class1 = []

    # Global classifier
    clf = None

    # Create ipywidgets controls
    epoch_field = widgets.IntText(
        value=config['network']['max_iter'],
        description='Epochs:',
        style={'description_width': '100px'}
    )
    
    lr_field = widgets.FloatText(
        value=0.001,
        description='Learning Rate:',
        style={'description_width': '100px'},
        step=0.0001,
        format='.4f'
    )
    
    network_field = widgets.Text(
        value=' '.join(map(str, config['network']['hidden_layer_sizes'])),
        description='Network Size:',
        style={'description_width': '100px'}
    )
    
    train_button = widgets.Button(
        description='TRAIN',
        button_style='success',
        tooltip='Train the network with current settings'
    )
    
    clear_button = widgets.Button(
        description='CLEAR ADDED POINTS',
        button_style='warning',
        tooltip='Remove all user-added points'
    )

    output = widgets.Output()

    def train_and_plot(event=None):
        nonlocal clf

        with output:
            clear_output(wait=True)
            
            # Get values from input fields
            try:
                epochs = int(epoch_field.value)
                learning_rate = float(lr_field.value)
                network_sizes = [int(x) for x in network_field.value.split()]
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
            print(f"Training with {len(X_combined)} points...")
            clf.fit(X_combined, y_combined)
            print(f"Training complete! Loss: {clf.loss_:.4f}")

            # Predict on grid
            Z = clf.predict_proba(grid)[:,1].reshape(XX.shape)

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot decision boundary
            contourf = ax.contourf(XX, YY, Z, levels=20, cmap='RdYlBu', alpha=0.4)
            contour = ax.contour(XX, YY, Z, levels=[0.5], colors='purple', linewidths=2)
            plt.colorbar(contourf, ax=ax, label='Class 1 Probability')

            # Plot original data points
            ax.scatter(X_original[y_original==0,0], X_original[y_original==0,1],
                      marker='o', label='Class 0 (original)', color='red', alpha=0.7, s=50)
            ax.scatter(X_original[y_original==1,0], X_original[y_original==1,1],
                      marker='x', label='Class 1 (original)', color='blue', alpha=0.7, s=50)

            # Plot user-added points
            if X_user_class0:
                ax.scatter(np.array(X_user_class0)[:,0], np.array(X_user_class0)[:,1],
                          marker='o', label='Class 0 (added)', color='red', s=150, 
                          edgecolors='black', linewidths=2)
            if X_user_class1:
                ax.scatter(np.array(X_user_class1)[:,0], np.array(X_user_class1)[:,1],
                          marker='x', label='Class 1 (added)', color='blue', s=150, linewidths=3)

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'{title}\n{len(X_user_class0)} Class 0 and {len(X_user_class1)} Class 1 points added')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Enable click events
            def on_click(event):
                if event.inaxes == ax and event.xdata and event.ydata:
                    if event.button == 1:  # Left click - Class 0
                        X_user_class0.append([event.xdata, event.ydata])
                        print(f'✓ Added Class 0 point at ({event.xdata:.2f}, {event.ydata:.2f})')
                    elif event.button == 3:  # Right click - Class 1
                        X_user_class1.append([event.xdata, event.ydata])
                        print(f'✓ Added Class 1 point at ({event.xdata:.2f}, {event.ydata:.2f})')
                    
                    # Retrain automatically
                    train_and_plot()

            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()

    def clear_user_points(event):
        nonlocal X_user_class0, X_user_class1
        X_user_class0.clear()
        X_user_class1.clear()
        with output:
            print("🗑️ Cleared all user-added points")
        train_and_plot()

    # Connect buttons
    train_button.on_click(train_and_plot)
    clear_button.on_click(clear_user_points)

    # Layout widgets
    controls_row1 = HBox([epoch_field, lr_field])
    controls_row2 = HBox([network_field])
    buttons_row = HBox([train_button, clear_button])
    
    ui = VBox([controls_row1, controls_row2, buttons_row, output])
    
    # Display interface
    display(ui)

    # Print instructions
    print("\n" + "="*60)
    print("🎯 INTERACTIVE MODE ACTIVATED")
    print("="*60)
    print("📍 Left click: Add Class 0 point (red circles)")
    print("📍 Right click: Add Class 1 point (blue crosses)")
    print("🔄 Model retrains automatically when points are added")
    print("🗑️  Use 'CLEAR ADDED POINTS' button to reset additions")
    print("="*60 + "\n")

    # Initial training and plot
    train_and_plot()


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
