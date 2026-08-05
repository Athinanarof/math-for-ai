"""
==========================================================================
COMPLETE GUIDE: Linear Regression from scratch - the bowl, gradient
descent, the regression line with error squares, and the tangent line
==========================================================================

Script structure:
  PART 1: Data and the cost function J(m,b)
  PART 2: Building the "bowl" (sweep over all possible m,b) in 3D
  PART 3: Gradient Descent -- "rolling" toward the minimum step by step
  PART 4: The regression line + error squares, over the real data
  PART 5: 2D slice of the bowl + the tangent line at a point (the derivative, visually)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)

# ==========================================================================
# PART 1: Data and cost function
# ==========================================================================
x = np.array([50, 70, 100])   # square meters
y = np.array([100, 140, 200]) # price (thousands $)
n = len(x)

def cost(m, b, x, y):
    """J(m,b) = (1/n) * sum( (y_i - (m*x_i + b))^2 )"""
    y_pred = m * x + b
    errors = y - y_pred
    return np.mean(errors ** 2)

def gradients(m, b, x, y):
    """dJ/dm and dJ/db -- the same formulas derived by hand."""
    y_pred = m * x + b
    error = y - y_pred
    dm = (-2 / n) * np.sum(x * error)
    db = (-2 / n) * np.sum(error)
    return dm, db

print("Data:")
for xi, yi in zip(x, y):
    print(f"  x={xi}, y={yi}")

# Instead of assuming m=2 is the answer, sweep candidates and find the
# lowest J ourselves -- this is how you'd actually discover it.
print("\nSweeping candidate values of m (with b=0 fixed):")
print(f"{'m':>6} | {'J(m, b=0)':>12}")
print("-" * 22)
best_m, best_j = None, np.inf
for m_candidate in np.arange(0, 4.5, 0.5):
    j = cost(m_candidate, 0, x, y)
    print(f"{m_candidate:>6.1f} | {j:>12.2f}")
    if j < best_j:
        best_m, best_j = m_candidate, j

print(f"\nLowest J found in this sweep: m={best_m}, J={best_j:.4f}")
print("(Coarse sweep only checks m in steps of 0.5 -- the real minimum")
print(" between two candidates is found exactly via calculus or gradient descent below.)")

# Plot the sweep so you can SEE where the minimum falls, not just read numbers
m_candidates = np.arange(0, 4.5, 0.5)
j_candidates = [cost(m, 0, x, y) for m in m_candidates]

fig0, ax0 = plt.subplots(figsize=(8, 5))
ax0.plot(m_candidates, j_candidates, marker='o', color="#2b6cb0", linewidth=2,
          label="J(m) for each candidate tried")
ax0.scatter([best_m], [best_j], color='red', s=150, zorder=5,
            label=f"Lowest found: m={best_m}, J={best_j:.1f}")
ax0.set_xlabel("m (candidate slope)")
ax0.set_ylabel("J(m, b=0)")
ax0.set_title("Sweeping m by hand: where does J bottom out?")
ax0.grid(alpha=0.3)
ax0.legend()
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/0_sweep_m.png', dpi=130)
print("Saved: 0_sweep_m.png")

# ==========================================================================
# PART 2: Building the bowl -- sweep over all combinations of m,b
# ==========================================================================
m_vals = np.linspace(-2, 6, 200)
b_vals = np.linspace(-100, 100, 200)
M, B = np.meshgrid(m_vals, b_vals)

J = np.zeros_like(M)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        J[i, j] = cost(M[i, j], B[i, j], x, y)

m_opt, b_opt = 2, 0
j_opt = cost(m_opt, b_opt, x, y)

fig1 = plt.figure(figsize=(14, 6))

ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(M, B, J, cmap='viridis', alpha=0.85, edgecolor='none')
ax1.scatter([m_opt], [b_opt], [j_opt], color='red', s=100,
            label=f'Minimum (m={m_opt}, b={b_opt}, J={j_opt:.1f})')
ax1.set_xlabel('m (slope)')
ax1.set_ylabel('b (intercept)')
ax1.set_zlabel('J(m,b)')
ax1.set_title('The "bowl": J(m,b) for every possible line')
ax1.legend()

ax2 = fig1.add_subplot(1, 2, 2)
contour = ax2.contourf(M, B, J, levels=40, cmap='viridis')
plt.colorbar(contour, ax=ax2, label='J(m,b)')
ax2.scatter([m_opt], [b_opt], color='red', s=100, marker='*', label='Minimum')
ax2.set_xlabel('m (slope)')
ax2.set_ylabel('b (intercept)')
ax2.set_title('Top-down view (contour lines)')
ax2.legend()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/1_bowl_3d.png', dpi=130)
print("\nSaved: 1_bowl_3d.png")

# ==========================================================================
# PART 3: Gradient Descent -- rolling toward the minimum
# ==========================================================================
m_current, b_current = -1.0, -80.0
lr_m = 0.000015   # different learning rates for m and b due to scale difference
lr_b = 0.9
history = [(m_current, b_current, cost(m_current, b_current, x, y))]

for step in range(3000):
    dm, db = gradients(m_current, b_current, x, y)
    m_current -= lr_m * dm
    b_current -= lr_b * db
    history.append((m_current, b_current, cost(m_current, b_current, x, y)))

print(f"\nAfter 3000 gradient descent steps:")
print(f"  m reached: {m_current:.4f}  (target: 2.0)")
print(f"  b reached: {b_current:.4f}  (target: 0.0)")
print(f"  Final J:   {history[-1][2]:.4f}  (target: 0.0)")

hist_m = [h[0] for h in history]
hist_b = [h[1] for h in history]
hist_j = [h[2] for h in history]

fig2 = plt.figure(figsize=(14, 6))

ax3 = fig2.add_subplot(1, 2, 1, projection='3d')
ax3.plot_surface(M, B, J, cmap='viridis', alpha=0.5, edgecolor='none')
ax3.plot(hist_m, hist_b, hist_j, color='red', linewidth=2, marker='o', markersize=2,
          label='Gradient descent path')
ax3.scatter([m_opt], [b_opt], [j_opt], color='yellow', s=150, marker='*', label='Minimum')
ax3.set_xlabel('m'); ax3.set_ylabel('b'); ax3.set_zlabel('J(m,b)')
ax3.set_title('Rolling toward the minimum (Gradient Descent)')
ax3.legend()

ax4 = fig2.add_subplot(1, 2, 2)
ax4.contourf(M, B, J, levels=40, cmap='viridis')
ax4.plot(hist_m, hist_b, color='red', linewidth=2, marker='o', markersize=2, label='Path')
ax4.scatter([m_opt], [b_opt], color='yellow', s=150, marker='*', label='Minimum')
ax4.set_xlabel('m'); ax4.set_ylabel('b')
ax4.set_title('Top-down view of the path')
ax4.legend()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/2_gradient_descent.png', dpi=130)
print("Saved: 2_gradient_descent.png")

# ==========================================================================
# PART 4: The regression line + error squares, over the real data
# ==========================================================================
fig3, axes = plt.subplots(1, 2, figsize=(14, 6))

lines_to_test = [
    {"m": 1.4, "b": 10, "title": "'Bad' line (m=1.4, b=10)", "ax": axes[0]},
    {"m": 2.0, "b": 0,  "title": "Optimal line (m=2, b=0)",  "ax": axes[1]},
]

for config in lines_to_test:
    m, b, ax = config["m"], config["b"], config["ax"]

    ax.scatter(x, y, color="#2b6cb0", s=80, zorder=5, label="Real data (y)")
    x_line = np.linspace(30, 110, 100)
    ax.plot(x_line, m * x_line + b, color="#b5533c", linewidth=2, label=f"ŷ = {m}x + {b}")

    j_total = 0
    for xi, yi in zip(x, y):
        yi_pred = m * xi + b
        error = yi - yi_pred
        area = error ** 2
        j_total += area

        ax.scatter(xi, yi_pred, color="#b5533c", s=50, zorder=5)
        ax.plot([xi, xi], [yi, yi_pred], color="gray", linestyle="--", linewidth=1)

        side = abs(error)
        corner_x = xi + 1
        corner_y = min(yi, yi_pred)
        rect = patches.Rectangle((corner_x, corner_y), side * 0.3, side,
                                   linewidth=1, edgecolor="#b5533c",
                                   facecolor="#b5533c", alpha=0.25)
        ax.add_patch(rect)
        ax.text(corner_x + side * 0.35, corner_y + side / 2,
                f"error={error:.0f}\narea={area:.0f}", fontsize=8, va="center")

    ax.set_title(f"{config['title']}\nJ = {j_total / n:.1f}")
    ax.set_xlabel("x (square meters)"); ax.set_ylabel("y (price)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/3_line_and_squares.png', dpi=130)
print("Saved: 3_line_and_squares.png")

# ==========================================================================
# PART 5: 2D slice of the bowl (b=0 fixed) + the tangent line at a point
# ==========================================================================
def J_of_m(m):
    return cost(m, 0, x, y)

def derivative_J_of_m(m):
    y_pred = m * x
    error = y - y_pred
    return (-2 / n) * np.sum(x * error)

fig4, ax = plt.subplots(figsize=(9, 6))

m_range = np.linspace(0, 4, 200)
J_vals = [J_of_m(m) for m in m_range]
ax.plot(m_range, J_vals, color="#2b6cb0", linewidth=2.5, label="J(m), with b=0 fixed")

m_point = 1.0
J_point = J_of_m(m_point)
tangent_slope = derivative_J_of_m(m_point)

m_tangent = np.linspace(m_point - 0.5, m_point + 0.5, 50)
J_tangent = J_point + tangent_slope * (m_tangent - m_point)

ax.plot(m_tangent, J_tangent, color="#d99a5b", linewidth=2.5, linestyle="--",
        label=f"Tangent at m={m_point} (slope = dJ/dm = {tangent_slope:.0f})")
ax.scatter([m_point], [J_point], color="#d99a5b", s=100, zorder=5,
           label=f"Point (m={m_point}, J={J_point:.0f})")
ax.scatter([2], [0], color="red", s=120, marker="*", zorder=5, label="Minimum (m=2, J=0)")

ax.set_xlabel("m (slope)"); ax.set_ylabel("J(m)  [with b=0 fixed]")
ax.set_title("2D slice of the bowl: the tangent line = the derivative at that point")
ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-2000, 26000)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/4_tangent_derivative.png', dpi=130)
print("Saved: 4_tangent_derivative.png")

print(f"\nAt m={m_point}: tangent slope = {tangent_slope:.0f} (negative)")
print("-> the curve is still going downhill there, so gradient descent")
print("   increases m until it reaches the real minimum at m=2.")