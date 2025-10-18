# interactive_mlp.py
# Works in Colab when the runner cell does: 
#   output.enable_custom_widget_manager(); %pip -q install ipympl ipywidgets; %matplotlib widget
# Do NOT call `%matplotlib inline` when using this.

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mw
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Keep widget refs alive in notebooks (prevents GC -> dead callbacks)
_WIDGET_REFS = {}

# --------------------------------------------------------------------------------------
# Public entry points (use any of these from your dataset scripts):
#   from interactive_mlp import run, render, launch, start, interactive_mlp
#   run(X, y, title="Circles")
# --------------------------------------------------------------------------------------

def run(X, y, title: str = "Interactive MLP"):
    return _launch(X, y, title)

def render(X, y, title: str = "Interactive MLP"):
    return _launch(X, y, title)

def launch(X, y, title: str = "Interactive MLP"):
    return _launch(X, y, title)

def start(X, y, title: str = "Interactive MLP"):
    return _launch(X, y, title)

def interactive_mlp(X, y, title: str = "Interactive MLP"):
    return _launch(X, y, title)

# --------------------------------------------------------------------------------------
# Implementation
# --------------------------------------------------------------------------------------

def _launch(X, y, title: str):
    """
    X: (n_samples, 2) float array
    y: (n_samples,) int array with labels {0,1}
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    assert X.ndim == 2 and X.shape[1] == 2, "X must be (n, 2)"
    assert y.ndim == 1 and len(y) == len(X), "y must be (n,) aligned with X"

    # Ranges for the plot / grid
    pad = 0.2
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    # User-added points
    X_user_class0, X_user_class1 = [], []

    # ---- Figure & main axes
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.28)  # leave space for controls
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    # Initial scatter (original data only)
    scat_train0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, label="Class 0 (original)")
    scat_train1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, marker="x", label="Class 1 (original)")
    ax.legend(loc="upper right")

    # Text instructions (bottom area)
    fig.text(
        0.02, 0.02,
        "=== INTERACTIVE MODE ===\n"
        " Left click: add Class 0 point (red dot)\n"
        " Right click OR Shift+Left click: add Class 1 point (blue cross)\n"
        " TRAIN: retrain with current params & added points\n"
        " CLEAR ADDED POINTS: reset user-added points",
        family="monospace", fontsize=9
    )

    # ---- Controls (TextBoxes + Buttons)
    # Labels
    fig.text(0.10, 0.12, "Epochs:", ha="right", va="center")
    fig.text(0.55, 0.12, "Learning Rate:", ha="right", va="center")
    fig.text(0.10, 0.07, "Network Size:", ha="right", va="center")

    # TextBoxes
    ax_epochs = plt.axes([0.12, 0.115, 0.18, 0.05])
    ax_lr     = plt.axes([0.57, 0.115, 0.18, 0.05])
    ax_hid    = plt.axes([0.12, 0.065, 0.28, 0.05])

    tb_epochs = mw.TextBox(ax_epochs, "", initial="2000")
    tb_lr     = mw.TextBox(ax_lr,     "", initial="0.001")
    tb_hidden = mw.TextBox(ax_hid,    "", initial="5 10")  # space- or comma-separated

    # Buttons
    ax_btn_clear = plt.axes([0.12, 0.005, 0.28, 0.05])
    ax_btn_train = plt.axes([0.70, 0.055, 0.18, 0.05])
    btn_clear = mw.Button(ax_btn_clear, "CLEAR ADDED POINTS")
    btn_train = mw.Button(ax_btn_train, "TRAIN")

    # Keep strong references
    _WIDGET_REFS.update(
        fig=fig, ax=ax,
        tb_epochs=tb_epochs, tb_lr=tb_lr, tb_hidden=tb_hidden,
        btn_clear=btn_clear, btn_train=btn_train,
        ax_epochs=ax_epochs, ax_lr=ax_lr, ax_hid=ax_hid,
        ax_btn_clear=ax_btn_clear, ax_btn_train=ax_btn_train
    )

    # ---- Helpers
    def _parse_hidden(s: str):
        s = s.strip().replace(",", " ")
        if not s:
            return (10,)
        vals = [int(v) for v in s.split() if v.strip().lstrip("+-").isdigit()]
        return tuple([v for v in vals if v > 0]) or (10,)

    def _current_params():
        try:
            epochs = int(float(tb_epochs.text))
        except Exception:
            epochs = 2000
        try:
            lr = float(tb_lr.text)
        except Exception:
            lr = 1e-3
        hidden = _parse_hidden(tb_hidden.text)
        return epochs, lr, hidden

    # We’ll reuse these to replot efficiently
    _STATE = {"contour": None, "scat_user0": None, "scat_user1": None, "clf": None}

    def _combine_data():
        if X_user_class0 or X_user_class1:
            X_add0 = np.array(X_user_class0, dtype=float) if X_user_class0 else np.zeros((0, 2))
            X_add1 = np.array(X_user_class1, dtype=float) if X_user_class1 else np.zeros((0, 2))
            X_all = np.vstack([X, X_add0, X_add1])
            y_all = np.concatenate([y, np.zeros(len(X_add0), dtype=int), np.ones(len(X_add1), dtype=int)])
            return X_all, y_all
        else:
            return X, y

    def _plot_decision_boundary(clf):
        # Remove old contour if present
        if _STATE["contour"] is not None:
            for c in _STATE["contour"].collections:
                c.remove()
            _STATE["contour"] = None

        # Grid
        gx = np.linspace(x_min, x_max, 300)
        gy = np.linspace(y_min, y_max, 300)
        XX, YY = np.meshgrid(gx, gy)
        grid = np.c_[XX.ravel(), YY.ravel()]

        try:
            Z = clf.predict_proba(grid)[:, 1]
        except Exception:
            # If predict_proba not available
            Z = clf.predict(grid).astype(float)
        Z = Z.reshape(XX.shape)

        # Filled contour (probability of class 1)
        _STATE["contour"] = ax.contourf(XX, YY, Z, levels=50, alpha=0.6, cmap="RdBu_r", antialiased=True)

        # Re-draw original + user points
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, marker="x")

        # User points (fresh artists each time)
        if _STATE["scat_user0"] is not None:
            _STATE["scat_user0"].remove()
            _STATE["scat_user0"] = None
        if _STATE["scat_user1"] is not None:
            _STATE["scat_user1"].remove()
            _STATE["scat_user1"] = None

        if X_user_class0:
            d0 = np.array(X_user_class0)
            _STATE["scat_user0"] = ax.scatter(d0[:, 0], d0[:, 1], s=30, edgecolor="k", linewidth=0.5)
        if X_user_class1:
            d1 = np.array(X_user_class1)
            _STATE["scat_user1"] = ax.scatter(d1[:, 0], d1[:, 1], s=40, marker="x", linewidth=1.0)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

    def train_and_plot():
        epochs, lr, hidden = _current_params()
        X_all, y_all = _combine_data()

        # Classifier: Standardize -> MLP
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                learning_rate_init=lr,
                max_iter=epochs,
                batch_size=min(512, len(X_all)),
                random_state=0,
                verbose=False,
            ),
        )
        clf.fit(X_all, y_all)
        _STATE["clf"] = clf

        # Train accuracy
        acc = float((clf.predict(X_all) == y_all).mean())
        ax.set_title(f"{title}  |  hidden={hidden}  epochs={epochs}  lr={lr:g}  acc={acc:.3f}")

        # Update decision boundary
        _plot_decision_boundary(clf)
        fig.canvas.draw_idle()

    # First render
    train_and_plot()

    # ---- Interactions
    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        # Left click → Class 0 unless Shift is held
        if event.button == 1 and not (event.key and "shift" in str(event.key).lower()):
            X_user_class0.append([event.xdata, event.ydata])
            print(f"Added Class 0 at ({event.xdata:.2f}, {event.ydata:.2f})")
        # Right click OR Shift+Left click → Class 1
        elif event.button == 3 or (event.button == 1 and event.key and "shift" in str(event.key).lower()):
            X_user_class1.append([event.xdata, event.ydata])
            print(f"Added Class 1 at ({event.xdata:.2f}, {event.ydata:.2f})")
        else:
            return
        train_and_plot()

    def _on_train_click(_):
        print("[TRAIN]")
        train_and_plot()

    def _on_clear_click(_):
        print("[CLEAR ADDED POINTS]")
        X_user_class0.clear()
        X_user_class1.clear()
        train_and_plot()

    # Connect callbacks; keep cids so they don't get GC'd
    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    _WIDGET_REFS.update(cid_click=cid_click)
    _WIDGET_REFS["cid_train"] = btn_train.on_clicked(_on_train_click)
    _WIDGET_REFS["cid_clear"] = btn_clear.on_clicked(_on_clear_click)

    plt.show()
    return {
        "figure": fig,
        "axis": ax,
        "get_params": _current_params,
        "train_and_plot": train_and_plot,
        "clf": lambda: _STATE.get("clf", None),
    }

# --------------------------------------------------------------------------------------
# Standalone demo (for local runs): python interactive_mlp.py
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_circles
    X_demo, y_demo = make_circles(n_samples=400, noise=0.1, factor=0.3, random_state=0)
    run(X_demo, y_demo, title="Circles (demo)")

