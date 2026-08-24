import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

n_petals = 8
n_frames = 100
petal_length = 1.6
petal_width = 0.55

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor("#12131a")
fig.patch.set_facecolor("#12131a")
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect("equal")
ax.axis("off")

def petal_shape(scale, angle, color):
    t = np.linspace(0, 1, 40)
    x = petal_width * scale * np.sin(np.pi * t)
    y = petal_length * scale * t
    x = np.concatenate([x, -x[::-1]])
    y = np.concatenate([y, y[::-1]])
    pts = np.column_stack([x, y])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = pts @ rot.T
    return Polygon(pts, closed=True, facecolor=color, edgecolor="none", alpha=0.95)

petal_colors = plt.cm.spring(np.linspace(0.1, 0.9, n_petals))
ax.plot([0, 0], [-2.2, -0.3], color="#3f8f4a", linewidth=4, zorder=0)

leaf = Polygon(np.array([[0, -1.2], [0.5, -0.9], [0, -0.6], [-0.15, -0.9]]),
               closed=True, facecolor="#3f8f4a", edgecolor="none", zorder=1)
ax.add_patch(leaf)

center = plt.Circle((0, 0), 0.001, facecolor="#f2c14e", edgecolor="none", zorder=5)
ax.add_patch(center)

petal_patches = []

def ease(t):
    return t * t * (3 - 2 * t)

def update(frame):
    global petal_patches
    for p in petal_patches:
        p.remove()
    petal_patches = []

    t = ease(min(frame / (n_frames * 0.7), 1.0))
    wobble = 0.03 * np.sin(frame * 0.15)

    for i in range(n_petals):
        angle = 2 * np.pi * i / n_petals + wobble
        patch = petal_shape(t, angle, petal_colors[i])
        ax.add_patch(patch)
        petal_patches.append(patch)

    center.set_radius(0.25 * t)
    return petal_patches + [center]

anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
anim.save("flower_bloom.gif", writer="pillow", fps=25)
plt.show()