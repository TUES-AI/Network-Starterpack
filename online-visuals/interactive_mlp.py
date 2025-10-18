"""
Colab-friendly interactive MLP decision boundary visualizer.

Replaces matplotlib.widgets with ipywidgets + ipympl so it works in Google Colab.
Keeps click-to-add-points behavior and live retraining.

Original references:
- /mnt/data/interactive_mlp.py  (desktop/Jupyter version)
- /mnt/data/circles.py          (entry script)
- /mnt/data/Config.yml          (hyperparameters & viz ranges)

Usage: import from a script like circles.py and call
create_interactive_mlp_visualization_with_config(X, y, title=...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Widgets (must be installed + widget manager enabled in the calling notebook)
try:
    import ipywidgets as W
except Exception as e:  # pragma: no cover
    W = None

from sklearn.neural_network import MLPClassifier
import yaml


@dataclass
class VizConfig:
    x_range: Tuple[float, float] = (-1.5, 1.5)
    y_range: Tuple[float, float] = (-1.5, 1.5)
    grid_resolution: int = 400


@dataclass
class NetConfig:
    hidden_layer_sizes: Tuple[int, ...] = (10,)
    activation: str = "relu"
    solver: str = "adam"
    batch_size: int = 128
    max_iter: int = 500
    random_state: int = 0
    learning_rate_init: float = 1e-3


def _find_config_path() -> Optional[str]:
    """
    Look for Config.yml in common locations:
    - CWD
    - repo root (one level up from this file)
    - online-visuals/
    """
    candidates = [
        "Config.yml",
        os.path.join(os.path.dirname(__file__), "..", "Config.yml"),
        os.path.join(os.path.dirname(__file__), "Config.yml"),
        os.path.join("online-visuals", "Config.yml"),
    ]
    for p in candidates:
        p2 = os.path.abspath(p)
        if os.path.exists(p2):
            return p2
    # As a last resort, if user uploaded to /mnt/data/Config.yml in Colab:
    p3 = "/mnt/data/Config.yml"
    if os.path.exists(p3):
        return p3
    return None


def _load_config_yaml(path: Optional[str] = None) -> tuple[NetConfig, VizConfig]:
    net = NetConfig()
    viz = VizConfig()
    cfg_path = path or _find_config_path()
    if not cfg_path or not os.path.exists(cfg_path):
        return net, viz

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    n = cfg.get("network", {}) or {}
    v = cfg.get("visualization", {}) or {}

    # Network
    hls = n.get("hidden_layer_sizes", list(net.hidden_layer_sizes))
    net.hidden_layer_sizes = tuple(int(x) for x in hls)
    net.activation = n.get("activation", net.activation)
    net.solver = n.get("solver", net.solver)
    net.batch_size = int(n.get("batch_size", net.batch_size))
    net.max_iter = int(n.get("max_iter", net.max_iter))
    net.random_state = int(n.get("random_state", net.random_state))
    if "learning_rate_init" in n:
        net.learning_rate_init = float(n.get("learning_rate_init"))

    # Viz
    viz.x_range = tuple(v.get("x_range", list(viz.x_range)))
    viz.y_range = tuple(v.get("y_range", list(viz.y_range)))
    viz.grid_resolution = int(v.get("grid_resolution", viz.grid_resolution))
    return net, viz


class MLPInteractive:
    """
    Interactive classifier + decision surface plot using ipympl in Colab.
    - Left-click on the figure to add a point for the selected class.
    - Change layers/epochs/LR in the widgets; click "Retrain" to re-fit.
    - "Clear user pts" removes only the user-added points.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        title: str = "MLP decision boundary",
        net_cfg: Optional[NetConfig] = None,
        viz_cfg: Optional[VizConfig] = None,
    ):
        if W is None:
            raise RuntimeError(
                "ipywidgets is not available. In Colab, ensure you've run:\n"
                "from google.colab import output; output.enable_custom_widget_manager()\n"
                "%pip install ipympl ipywidgets\n"
                "%matplotlib widget\n"
                "…BEFORE running this script."
            )

        self.title = title
        self.X0 = np.asarray(X, float)
        self.y0 = np.asarray(y, int)
        self.X_user: List[Tuple[float, float]] = []
        self.y_user: List[int] = []

        self.net_cfg = net_cfg or NetConfig()
        self.viz_cfg = viz_cfg or VizConfig()

        # Precompute grid
        self.xx, self.yy = np.meshgrid(
            np.linspace(self.viz_cfg.x_range[0], self.viz_cfg.x_range[1], self.viz_cfg.grid_resolution),
            np.linspace(self.viz_cfg.y_range[0], self.viz_cfg.y_range[1], self.viz_cfg.grid_resolution),
        )
        self.grid = np.c_[self.xx.ravel(), self.yy.ravel()]

        # Widgets
        self.epochs = W.IntText(value=self.net_cfg.max_iter, description="Epochs:", layout=W.Layout(width="120px"))
        self.lr = W.FloatText(value=self.net_cfg.learning_rate_init, description="LR:", layout=W.Layout(width="140px"))
        self.layers = W.Text(
            value=" ".join(map(str, self.net_cfg.hidden_layer_sizes)),
            description="Layers:",
            layout=W.Layout(width="220px"),
            placeholder="e.g. 10 10 5",
        )
        self.class_selector = W.Dropdown(
            options=[("Class 0", 0), ("Class 1", 1)],
            value=0,
            description="New point:",
            layout=W.Layout(width="160px"),
        )
        self.retrain_btn = W.Button(description="Retrain", button_style="")
        self.clear_user_btn = W.Button(description="Clear user pts", button_style="warning")
        self.status = W.HTML(value="Ready.")

        self.retrain_btn.on_click(self._on_retrain_click)
        self.clear_user_btn.on_click(self._on_clear_user_click)
        self.layers.observe(self._on_layers_change, names="value")
        self.epochs.observe(self._on_hyper_change, names="value")
        self.lr.observe(self._on_hyper_change, names="value")

        # Figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title(self.title)
        self.ax.set_xlim(*self.viz_cfg.x_range)
        self.ax.set_ylim(*self.viz_cfg.y_range)
        self.ax.set_aspect("equal")

        self.bg = None
        self.scatter0 = None
        self.scatter1 = None
        self.scatter0u = None
        self.scatter1u = None

        self._fit_and_draw(initial=True)

        # Mouse events
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ---------- UI callbacks ----------
    def _on_retrain_click(self, _):
        self._fit_and_draw()

    def _on_clear_user_click(self, _):
        self.X_user.clear()
        self.y_user.clear()
        self._fit_and_draw(skip_fit=True)

    def _on_layers_change(self, change):
        try:
            _ = tuple(int(x) for x in str(change["new"]).split())
            self.status.value = "Layers OK."
        except Exception:
            self.status.value = "<b>Invalid Layers</b> (use space-separated ints, e.g. '5 10 5')"

    def _on_hyper_change(self, _):
        self.status.value = "Hyperparams edited. Click <b>Retrain</b> or add a point."

    # ---------- Model helpers ----------
    def _build_clf(self) -> MLPClassifier:
        try:
            hidden = tuple(int(x) for x in self.layers.value.split()) or (10,)
        except Exception:
            hidden = self.net_cfg.hidden_layer_sizes
        max_iter = int(self.epochs.value) if (self.epochs.value and self.epochs.value > 0) else self.net_cfg.max_iter
        lr = float(self.lr.value) if (self.lr.value and self.lr.value > 0) else self.net_cfg.learning_rate_init

        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=self.net_cfg.activation,
            solver=self.net_cfg.solver,
            batch_size=self.net_cfg.batch_size,
            max_iter=max_iter,
            random_state=self.net_cfg.random_state,
            learning_rate_init=lr,
        )

    def _concat_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self.X_user:
            Xu = np.array(self.X_user, dtype=float)
            yu = np.array(self.y_user, dtype=int)
            X = np.vstack([self.X0, Xu])
            y = np.concatenate([self.y0, yu])
            return X, y
        return self.X0, self.y0

    def _fit_and_draw(self, initial: bool = False, skip_fit: bool = False):
        import time

        t0 = time.time()
        X, y = self._concat_data()

        if not skip_fit:
            self.clf = self._build_clf()
            self.clf.fit(X, y)

        proba = self.clf.predict_proba(self.grid)[:, 1].reshape(self.xx.shape)

        if self.bg is None:
            self.bg = self.ax.imshow(
                proba,
                origin="lower",
                extent=(self.viz_cfg.x_range[0], self.viz_cfg.x_range[1],
                        self.viz_cfg.y_range[0], self.viz_cfg.y_range[1]),
                vmin=0.0,
                vmax=1.0,
                alpha=0.6,
                interpolation="bilinear",
            )
            self.cbar = self.fig.colorbar(self.bg, ax=self.ax, fraction=0.046, pad=0.04)
            self.cbar.set_label("P(class=1)")
        else:
            self.bg.set_data(proba)

        m0 = y == 0
        m1 = y == 1
        if self.scatter0 is None:
            self.scatter0 = self.ax.scatter(X[m0, 0], X[m0, 1], marker="o", s=24, label="class 0", alpha=0.9)
            self.scatter1 = self.ax.scatter(X[m1, 0], X[m1, 1], marker="x", s=32, label="class 1", alpha=0.9)
            self.ax.legend(loc="upper right")
        else:
            self.scatter0.set_offsets(np.c_[X[m0, 0], X[m0, 1]])
            self.scatter1.set_offsets(np.c_[X[m1, 0], X[m1, 1]])

        if self.X_user:
            XU = np.array(self.X_user, float)
            yU = np.array(self.y_user, int)
            m0u = yU == 0
            m1u = yU == 1
            if self.scatter0u is None:
                self.scatter0u = self.ax.scatter(
                    XU[m0u, 0], XU[m0u, 1], marker="o", s=80, facecolors="none", edgecolors="black", linewidths=1.2,
                    label="user 0"
                )
            else:
                self.scatter0u.set_offsets(np.c_[XU[m0u, 0], XU[m0u, 1]])
            if self.scatter1u is None:
                self.scatter1u = self.ax.scatter(
                    XU[m1u, 0], XU[m1u, 1], marker="+", s=120, linewidths=1.5, color="black", label="user 1"
                )
            else:
                self.scatter1u.set_offsets(np.c_[XU[m1u, 0], XU[m1u, 1]])
        else:
            if self.scatter0u is not None:
                self.scatter0u.remove(); self.scatter0u = None
            if self.scatter1u is not None:
                self.scatter1u.remove(); self.scatter1u = None

        self.ax.set_title(self.title + f"  (n={len(X)})")
        self.fig.canvas.draw_idle()

        ms = (time.time() - t0) * 1000
        self.status.value = (
            f"Fitted in {ms:.1f} ms | layers={self.layers.value} | epochs={self.epochs.value} | lr={self.lr.value}"
        )

    # ---------- Mouse ----------
    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:  # left clicks only
            return
        if event.xdata is None or event.ydata is None:
            return
        self.X_user.append((float(event.xdata), float(event.ydata)))
        self.y_user.append(int(self.class_selector.value))
        self._fit_and_draw()

    # ---------- Public ----------
    def widget(self):
        controls = W.HBox([
            self.layers, self.epochs, self.lr, self.class_selector, self.retrain_btn, self.clear_user_btn
        ])
        return W.VBox([controls, self.status])


def create_interactive_mlp_visualization_with_config(X, y, title="MLP decision boundary"):
    net_cfg, viz_cfg = _load_config_yaml()
    viz = MLPInteractive(X, y, title, net_cfg=net_cfg, viz_cfg=viz_cfg)
    display(viz.widget())
    display(viz.fig)
    return viz

