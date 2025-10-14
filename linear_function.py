
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.grid": False,
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "xtick.color": "black",
    "ytick.color": "black",
    "toolbar": "none",
})

fig = plt.figure(figsize=(6.6, 6.8))
gs = fig.add_gridspec(nrows=8, ncols=1, height_ratios=[6, 0.15, 0.75, 0.1, 0.75, 0.15, 0.1, 0.1])

ax = fig.add_subplot(gs[0])
ax.set_navigate(False)

XMAX = 10
ax.set_xlim(-XMAX, XMAX)
ax.set_ylim(-XMAX, XMAX)
ax.set_aspect("equal", adjustable="box")
ax.set_xticks([]); ax.set_yticks([])

ax.axhline(0, color="black", lw=1.0, ls=(0, (2, 3)), alpha=0.8)
ax.axvline(0, color="black", lw=1.0, ls=(0, (2, 3)), alpha=0.8)


a0, b0 = 1.0, 0.0
x = np.linspace(-XMAX, XMAX, 800)
(line,) = ax.plot(x, a0 * x + b0, lw=2.5, color="crimson")
p_intercept = ax.plot(0, b0, "o", ms=6, color="crimson", alpha=0.9)[0]
p_sample    = ax.plot(-3, a0 * (-3) + b0, "o", ms=6, color="crimson", alpha=0.7)[0]

A_ANCHORS = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
U_ANCHORS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

def ui_to_a(u):
    i = np.searchsorted(U_ANCHORS, u, side="right") - 1
    i = int(np.clip(i, 0, len(U_ANCHORS) - 2))
    u0, u1 = U_ANCHORS[i], U_ANCHORS[i + 1]
    v0, v1 = A_ANCHORS[i], A_ANCHORS[i + 1]
    t = 0.0 if u1 == u0 else (u - u0) / (u1 - u0)
    return v0 + t * (v1 - v0)

def a_to_ui(a):
    j = np.searchsorted(A_ANCHORS, a, side="right") - 1
    j = int(np.clip(j, 0, len(A_ANCHORS) - 2))
    v0, v1 = A_ANCHORS[j], A_ANCHORS[j + 1]
    u0, u1 = U_ANCHORS[j], U_ANCHORS[j + 1]
    t = 0.0 if v1 == v0 else (a - v0) / (v1 - v0)
    return u0 + t * (u1 - u0)

ax_a = fig.add_subplot(gs[2])
ax_b = fig.add_subplot(gs[4])

for ax_s in (ax_a, ax_b):
    ax_s.set_xticks([])
    ax_s.set_yticks([])

slope_slider = Slider(ax=ax_a, label="", valmin=0.0, valmax=1.0, valinit=a_to_ui(a0), valstep=0.001)
interc_slider = Slider(ax=ax_b, label="", valmin=-5, valmax=5, valinit=b0, valstep=0.01)

slope_slider.label.set_visible(False)
interc_slider.label.set_visible(False)
slope_slider.valtext.set_visible(False)
interc_slider.valtext.set_visible(False)

ax_formula = fig.add_subplot(gs[1])
ax_formula.set_navigate(False)
ax_formula.axis("off")
formula_text = ax_formula.text(
    0.5, 0.5, "", ha="center", va="center", fontsize=14, color="black", transform=ax_formula.transAxes
)

def update(_):
    a = ui_to_a(slope_slider.val)
    b = interc_slider.val

    line.set_ydata(a * x + b)
    p_intercept.set_data([0], [b])
    p_sample.set_data([-3], [a * (-3) + b])

    formula_text.set_text(f"y = {a:.3g} * x + {b:.3g}")

    fig.canvas.draw_idle()

update(None)

slope_slider.on_changed(update)
interc_slider.on_changed(update)

fig.subplots_adjust(left=0.08, right=0.97, top=0.98, bottom=0.05)
try:
    fig.canvas.toolbar_visible = False
except Exception:
    pass

plt.show()
