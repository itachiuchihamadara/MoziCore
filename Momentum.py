import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Define the function and its partial derivatives
import numpy as np

def z(x, y):
    """
    Convex quadratic function: f(x, y) = x^2 + y^2 + x*y
    """
    return x**2 + y**2 + x * y

def dz_dx(x, y):
    """
    Partial derivative of f(x, y) with respect to x.
    For f(x, y) = x^2 + y^2 + x*y, this is 2*x + y
    """
    return 2 * x + y

def dz_dy(x, y):
    """
    Partial derivative of f(x, y) with respect to y.
    For f(x, y) = x^2 + y^2 + x*y, this is 2*y + x
    """
    return 2 * y + x

# Create grid
x = np.linspace(-15, 15, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = z(X, Y)

# Choose a starting point for descent (not at a minimum)
start_x, start_y = -14, -2
start_z = z(start_x, start_y)

# Create plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.7)


# Add labels and title
ax.set_title('Function z(x,y) with Starting Point for Descent')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()

beta = 0.1
momentum_x = 0
momentum_y = 0

left_m = False
ax.scatter(start_x, start_y, z(start_x, start_y), color='red', s=100, label='Starting point')
print(z(start_x, start_y))
for i in range(100):
    grad_x = dz_dx(start_x, start_y)
    grad_y = dz_dy(start_x, start_y)

    momentum_x = (1-beta) * momentum_x + beta * (grad_x)
    start_x = start_x - 0.1 * momentum_x

    momentum_y = (1-beta) * momentum_y + beta * (grad_y)
    start_y = start_y - 0.1 * momentum_y

    if momentum_x > 0 and not left_m:
        ax.scatter(start_x, start_y, z(start_x, start_y), color='blue', s=100, label='Starting point')
        left_m = True
        # break

    if momentum_x < 0 and left_m:
        ax.scatter(start_x, start_y, z(start_x, start_y), color='green', s=100, label='Starting point')
        left_m = False

    # if i%9 == 0:
    ax.scatter(start_x, start_y, z(start_x, start_y), color='red', s=100, label='Starting point')

# for i in range(20):
#     grad_x = dz_dx(start_x, start_y)
#     grad_y = dz_dy(start_x, start_y)
#
#     start_x = start_x - 0.1 * grad_x
#     start_y = start_y - 0.1 * grad_y
#
#     # if i%7 == 0:
#     ax.scatter(start_x, start_y, z(start_x, start_y), color='red', s=100, label='Starting point')

print(start_x, start_y)
ax.scatter(start_x, start_y, z(start_x, start_y), color='black', s=1000, label='Starting point')
print(z(start_x, start_y))
plt.show()
plt.tight_layout()
