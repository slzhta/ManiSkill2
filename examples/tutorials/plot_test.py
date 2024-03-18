import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data: Replace this with your actual data
points = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
lines = [(0, 1), (1, 2), (2, 0)]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
xs, ys, zs = zip(*points)
ax.scatter(xs, ys, zs, c='b', marker='o', label='Points')

# Plot lines
for line in lines:
    x_vals, y_vals, z_vals = zip(*[points[i] for i in line])
    print(x_vals)
    ax.plot(x_vals, y_vals, z_vals, c='r', label='Lines')

# Customize plot (optional)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show plot
plt.savefig("./logs/test.png")