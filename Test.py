import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Define the function and its partial derivatives
def z(x, y):
    return (x**2 + y**2)/10 + np.sin(x) + 2*np.cos(y) + 0.5*np.sin(x)*np.cos(y)

def dz_dx(x, y):
    return x/5 + np.cos(x) + 0.5*np.cos(x)*np.cos(y)

def dz_dy(x, y):
    return y/5 - 2*np.sin(y) - 0.5*np.sin(x)*np.sin(y)

# Create grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = z(X, Y)

# Choose a starting point for descent (not at a minimum)
start_x, start_y = -5, 0
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

# Show the gradient vector at this point
grad_x = dz_dx(start_x, start_y)
grad_y = dz_dy(start_x, start_y)
ax.quiver(start_x, start_y, start_z, -grad_x, -grad_y, 0, 
          color='green', length=1, arrow_length_ratio=0.1, label='Negative gradient (descent direction)')



beta = 0.1
momentum_x = 0
momentum_y = 0
ax.scatter(start_x, start_y, z(start_x, start_y), color='red', s=100, label='Starting point')
for i in range(20):
    grad_x = dz_dx(start_x, start_y)
    grad_y = dz_dy(start_x, start_y)

    momentum_x = (1-beta) * momentum_x + beta * (grad_x)
    print(start_x, start_y)

    start_x = start_x - 0.2 * momentum_x;

    momentum_y = (1-beta) * momentum_y + beta * (grad_y)
    start_y = start_y - 0.2 * momentum_y;

    ax.scatter(start_x, start_y, z(start_x, start_y), color='red', s=100, label='Starting point')

for i in range(100):
    grad_x = dz_dx(start_x, start_y)
    grad_y = dz_dy(start_x, start_y)

    start_x = start_x - 0.2 * grad_x;
    start_y = start_y - 0.2 * grad_y;

    ax.scatter(start_x, start_y, z(start_x, start_y), color='red', s=100, label='Starting point')

print(start_x, start_y)
plt.show()
plt.tight_layout()
