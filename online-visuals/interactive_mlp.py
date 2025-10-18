# interactive_mlp_colab.py
import numpy as np
import yaml
from pathlib import Path
from ipywidgets import (VBox, HBox, Button, ToggleButtons, Text, Dropdown,
                        FloatText, IntText, HTML, Layout, Accordion)
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def _load_config():
    # Fallbacks if Config.yml is missing or partial
    defaults = {
        "network": {
            "hidden_layer_sizes": [5, 10],
            "activation": "relu",
            "solver": "adam",
            "batch_size": 400,
            "max_iter": 2000,
            "random_state": 0,
        },
        "visualization": {
            "x_range": [-1.5, 1.5],
            "y_range": [-1.5, 1.5],
            "grid_resolution": 400,
        },
        "training": {
            "n_samples": 400,
            "random_state": 0,
        },
    }
    cfg_path = Path("Config.yml")
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            user = yaml.safe_load(f) or {}
        # recursive merge – shallow is fine for this config
        for k in defaults:
            if k in user and isinstance(user[k], dict):
                defaults[k].update(user[k])
            elif k in user:
                defaults[k] = user[k]
    return defaults

def _parse_layers(text, fallback):
    try:
        if isinstance(text, (list, tuple)):
            return tuple(int(v) for v in text)
        # allow "5,10" or "5 10"
        text = text.replace(",", " ")
        return tuple(int(t) for t in text.split() if t.strip())
    except Exception:
        return tuple(int(v) for v in fallback)

def create_interactive_mlp_visualization_with_config(X, y, title="Interactive MLP"):
    config = _load_config()
    x_min, x_max = config["visualization"]["x_range"]
    y_min, y_max = config["visualization"]["y_range"]
    grid_res = int(config["visualization"]["grid_resolution"])

    # Defaults from config
    default_layers = tuple(config["network"]["hidden_layer_sizes"])
    default_activation = config["network"]["activation"]
    default_solver = config["network"]["solver"]
    default_max_iter = int(config["network"]["max_iter"])
    default_lr = 0.001  # scikit default; YAML doesn't set learning_rate_init

    # Make grid for decision boundary
    xs = np.linspace(x_min, x_max, grid_res)
    ys = np.linspace(y_min, y_max, grid_res)
    gx, gy = np.meshgrid(xs, ys)               # shape (R, R)
    grid_points = np.c_[gx.ravel(), gy.ravel()]

    # Original data (immutable); user-added mutable lists
    X0 = X[y == 0]
    X1 = X[y == 1]
    user0, user1 = [], []

    # Widgets
    class_selector = ToggleButtons(
        options=[("Add Class 0", 0), ("Add Class 1", 1)],
        value=0,
        layout=Layout(width="auto")
    )
    layers_text = Text(
        value=",".join(str(v) for v in default_layers),
        description="Hidden:",
        layout=Layout(width="220px")
    )
    activation_dd = Dropdown(
        options=["relu", "tanh", "logistic", "identity"],
        value=default_activation, description="Act."
    )
    solver_dd = Dropdown(
        options=["adam", "lbfgs", "sgd"],
        value=default_solver, description="Solver"
    )
    lr_ft = FloatText(value=default_lr, step=0.0005, description="LR")
    maxiter_it = IntText(value=default_max_iter, description="MaxIter")
    retrain_btn = Button(description="Retrain", button_style="success")
    clear_btn = Button(description="Clear added points", button_style="warning")
    status_html = HTML(value="")

    controls_row = HBox([
        class_selector, layers_text, activation_dd, solver_dd, lr_ft, maxiter_it,
        retrain_btn, clear_btn
    ], layout=Layout(flex_flow="row wrap", gap="10px"))

    # Build model in a pipeline with scaling for stability
    def make_clf():
        layers = _parse_layers(layers_text.value, default_layers)
        clf = MLPClassifier(
            hidden_layer_sizes=layers,
            activation=activation_dd.value,
            solver=solver_dd.value,
            learning_rate_init=float(lr_ft.value),
            max_iter=int(maxiter_it.value),
            random_state=config["network"]["random_state"],
        )
        return make_pipeline(StandardScaler(), clf)

    clf = make_clf()

    # Initial training
    def current_training_set():
        if user0 or user1:
            Xu = np.array(user0 + user1) if (user0 or user1) else np.empty((0, 2))
            yu = np.array([0] * len(user0) + [1] * len(user1)) if (user0 or user1) else np.empty((0,))
            Xtrain = np.vstack([X, Xu]) if Xu.size else X
            ytrain = np.concatenate([y, yu]) if Xu.size else y
        else:
            Xtrain, ytrain = X, y
        return Xtrain, ytrain

    def fit_and_predict():
        Xtr, ytr = current_training_set()
        model = make_clf()
        model.fit(Xtr, ytr)
        # Predict proba for heatmap
        prob = model.predict_proba(grid_points)[:, 1].reshape(gx.shape)
        return model, prob

    model, prob = fit_and_predict()

    # Figure: Heatmap = decision surface; scatters = points
    # Plotly Heatmap expects z.shape == (len(y), len(x))
    heat = go.Heatmap(
        x=xs, y=ys, z=prob, zmin=0.0, zmax=1.0,
        colorscale="RdBu", reversescale=True, showscale=False,
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>P(class=1)=%{z:.3f}<extra></extra>",
    )
    s0 = go.Scatter(x=X0[:, 0], y=X0[:, 1], mode="markers",
                    name="Class 0 (orig)", marker=dict(symbol="circle", size=7))
    s1 = go.Scatter(x=X1[:, 0], y=X1[:, 1], mode="markers",
                    name="Class 1 (orig)", marker=dict(symbol="x", size=9))
    su0 = go.Scatter(x=[], y=[], mode="markers",
                     name="Class 0 (added)", marker=dict(symbol="circle-open", size=10))
    su1 = go.Scatter(x=[], y=[], mode="markers",
                     name="Class 1 (added)", marker=dict(symbol="x-thin-open", size=12))

    fig = go.FigureWidget(data=[heat, s0, s1, su0, su1])
    fig.update_layout(
        title=title,
        width=700, height=700,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[x_min, x_max], constrain="domain"),
        yaxis=dict(range=[y_min, y_max], scaleanchor="x", scaleratio=1),
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    # Click-to-add handler (click the heatmap background)
    def on_click(trace, points, state):
        if not points.points:
            return
        p = points.points[0]
        x_click = float(p.x)
        y_click = float(p.y)
        cls = class_selector.value
        if cls == 0:
            user0.append([x_click, y_click])
        else:
            user1.append([x_click, y_click])
        # update user scatter traces
        with fig.batch_update():
            if user0:
                arr0 = np.array(user0)
                fig.data[3].x = arr0[:, 0]
                fig.data[3].y = arr0[:, 1]
            if user1:
                arr1 = np.array(user1)
                fig.data[4].x = arr1[:, 0]
                fig.data[4].y = arr1[:, 1]
        retrain()  # retrain immediately

    def retrain(*_):
        nonlocal model, prob
        model, prob = fit_and_predict()
        with fig.batch_update():
            fig.data[0].z = prob
        status_html.value = "<b>Model retrained.</b>"

    def clear_added(*_):
        user0.clear()
        user1.clear()
        with fig.batch_update():
            fig.data[3].x, fig.data[3].y = [], []
            fig.data[4].x, fig.data[4].y = [], []
        retrain()

    retrain_btn.on_click(retrain)
    clear_btn.on_click(clear_added)

    # Wire click handler to the heatmap trace
    fig.data[0].on_click(on_click)

    help_box = HTML(
        value=(
            "<b>Usage</b>: Click anywhere in the plot to add a point. Use the toggle to choose the class. "
            "Model retrains automatically. Pan/zoom with the usual Plotly controls.<br>"
            "<i>Tips</i>: Increase hidden units if the boundary underfits; "
            "try tanh/logistic for different shapes; adjust LR and MaxIter for convergence."
        )
    )

    ui = VBox([controls_row, fig, status_html, help_box])
    display(ui)

