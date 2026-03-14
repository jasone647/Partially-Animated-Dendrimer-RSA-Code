import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

boxlength = 400.0
radius = 8.0
innerthresh = 4.0
outsidethresh = 10.0

surfcov = (np.pi*radius**2)/(boxlength*boxlength)
placed = []
placed.append([np.random.rand()*boxlength, np.random.rand()*boxlength])
failedattempts = 0

while surfcov < 0.35:
    attemptcoords = [np.random.rand()*boxlength, np.random.rand()*boxlength]

    # Check if this position is valid against ALL previously placed circles
    valid_placement = True

    for j in placed:
        dist = np.sqrt((attemptcoords[0] - j[0])**2 + (attemptcoords[1] - j[1])**2)

        # Too close - reject immediately
        if dist <= (innerthresh + radius):
            failedattempts += 1
            valid_placement = False
            break

        # In the forbidden zone - reject immediately
        elif dist > (radius + radius) and dist <= (radius + outsidethresh):
            failedattempts += 1
            valid_placement = False
            break

    # Only place the circle if it passed all distance checks
    if valid_placement:
        placed.append(attemptcoords)
        surfcov += (np.pi*radius**2)/(boxlength*boxlength)
        print(f"Placed circle {len(placed)}, surface coverage: {surfcov:.4f}")

print(f"Final: {len(placed)} circles placed, coverage: {surfcov:.4f}")
print(f"Failed attempts: {failedattempts}")

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Set up the plot area
ax.set_xlim(0, boxlength)
ax.set_ylim(0, boxlength)
ax.set_aspect('equal')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title(f'Circle Placement Visualization\n{len(placed)} circles, Coverage: {surfcov:.4f}')

# Plot each circle
for i, (x, y) in enumerate(placed):
    circle = plt.Circle((x, y), radius, fill=False, edgecolor='blue', linewidth=1.5)
    ax.add_patch(circle)
    # Optionally add center points
    ax.plot(x, y, 'ro', markersize=3, alpha=0.7)

# Add grid for better visualization
ax.grid(True, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()
