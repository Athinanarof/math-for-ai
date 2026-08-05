"""
=======================
Linear Regression Math
=======================

Part 1: Data and Cost function J(m,b)
Didactico, como el rango 0 - 4 para darme una idea de como se ve la funcion de costo

Simple linear regression          | ML in general (neural nets, multiple regression)
-----------------------------------------------------------------------------------
m (a single number)               | w -- weights, one per input feature
b (a single number)               | b -- bias, same name, same role
ŷ = m*x + b                       | ŷ = w1*x1 + w2*x2 + ... + wn*xn + b
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================================
# PART 1: Data and cost function J(m,b)
# ==========================================================================

x = np.array([50, 70, 100])     # square meters
y = np.array([100, 140, 200])   # price (thousands $)
n = len(x)


def cost(m, b, x, y):
    """J(m,b) = (1/n) * sum( (y_i - (m*x_i + b))^2 )"""
    y_pred = m * x + b
    error = y - y_pred
    return np.mean(error ** 2)

print("Data:")
for xi, yi in zip(x, y):
    print(f"  x={xi}, y={yi}")

# ==========================================================================
#  PART 1: 2D, Find the lowest J
# 2D: fijamos b=0 (asumido) → solo buscamos m → resultado: una curva (parábola)
# 3D: dejamos que b también varíe → buscamos m y b juntos → resultado: una superficie (tazón)
# ==========================================================================

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
plt.savefig('0_sweep_m.png', dpi=130)
print("Saved: 0_sweep_m.png")

# ==========================================================================
#  PART 2: 3D, Building the bowl -- sweep over all combinations of m,b
# 3D: dejamos que b también varíe → buscamos m y b juntos → resultado: una superficie (tazón)
# ==========================================================================
m_vals = np.linspace(-2, 6, 200)
b_vals = np.linspace(-100, 100, 200)
M, B = np.meshgrid(m_vals, b_vals)

J = np.zeros_like(M)
for i in range(M.shape[0]): #row
    for j in range(M.shape[1]): #column
        J[i, j] = cost(M[i,j], B[i,j], x, y)

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
plt.savefig('1_bowl_3d.png', dpi=130)
plt.show()
print("\nSaved: 1_bowl_3d.png")


# ==========================================================================
# PART 3: Gradient Descent -- rolling toward the minimum
# ==========================================================================

# ∂J/∂m = (−2/n) · Σ xᵢ(yᵢ − m·xᵢ − b)
# ∂J/∂b = (−2/n) · Σ (yᵢ − m·xᵢ − b)
def gradients(m, b, x, y):
    """dJ/dm and dJ/db -- the same formulas derived by hand"""
    y_pred = m * x + b
    error = y - y_pred
    dm = (-2 / n) * np.sum(x * error)
    db = (-2 / n) * np.sum(error)
    return dm, db

m_current, b_current = -1.0, -80.0
lr_m = 0.000015   # different learning rates for m and b due to scale difference
lr_b = 0.9 # 🐻‍❄️ PENDING: esto se optiene a traves de normalizar -> optimizacion

history = [(m_current, b_current, cost(m_current, b_current, x, y))]

for step in range(3000): # 🐻‍❄️ PENDING: valores reales -> epochs + validation
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
plt.savefig('2_gradient_descent.png', dpi=130)
plt.show()
print("Saved: 2_gradient_descent.png")
