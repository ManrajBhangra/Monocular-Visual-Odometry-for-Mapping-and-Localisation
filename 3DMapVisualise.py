import numpy as np
import matplotlib.pyplot as plt


features = np.load("robotino_xyz.npy") 
print(f"Loaded {len(features)} 3D points")

z = features[:, 2]
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9) 

fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    features[:, 0], features[:, 1], features[:, 2],
    c=z_norm, cmap='plasma',
    s=2, alpha=0.7
)

plt.colorbar(sc, ax=ax, label='Normalised Height', shrink=0.6)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Robotino 3D Feature Map')

ax.scatter([0], [0], [0], c='lime', s=80, marker='^', label='Robotino')
ax.legend()

plt.tight_layout()
plt.savefig("3d Visualised Map")
