from sheet_current import SheetCurrent
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import plot
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



nfp = 2
surf = SurfaceRZFourier(nfp, stellsym=True, mpol=3, ntor=3, 
                        quadpoints_phi=np.linspace(0, 1/nfp, 31, endpoint=False),
                        quadpoints_theta=np.linspace(0, 1, 31, endpoint=False))
surf.set('rc(0,1)', 0.1)
surf.set('rc(0,2)', 0.1)
surf.set('rc(0,3)', 0.1)
surf.set('zs(0,1)', 0.2)
surf.set('zs(0,2)', 0.1)



surf_outer = surf.copy()
mr = surf.minor_radius()
surf_outer.extend_via_normal(mr*2)
G = 0.8
M = 6
N = 6
current = SheetCurrent(surf_outer, G, M, N)
current.set('s(0,0)', 1.0)
current.set('s(0,1)', 1.0)
current.set('s(1,0)', 1.0)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = surf.gamma()
sc = ax.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], label='surface')
X = surf_outer.gamma()
sc = ax.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], alpha=0.5, label='winding surface')
plt.tight_layout()
plt.legend()
plt.show()


# check the squared flux
print('initial squared flux:')
print(current.squared_flux(surf))

# fit the sheet current
current.fit(surf)

# check the squared flux
print('final squared flux:')
print(current.squared_flux(surf))

# compute B on axis
X = surf.gamma() # (nphi, ntheta, 3)
X_axis = np.mean(X, axis=1) # (nphi, 3)
B_axis = current.B(X_axis)
print('Average |B| on axis:')
print(np.mean(np.linalg.norm(B_axis, axis=-1)))


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# copy the surface with more points for better visualization
surf = surf.copy(nphi=128, ntheta=64, range='field period')
X = surf.gamma()
Bnormal = current.B_normal(surf)
norm = plt.Normalize(vmin=Bnormal.min(), vmax=Bnormal.max())
colors = plt.cm.viridis(norm(Bnormal))
sc = ax.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], facecolors=colors, rstride=1, cstride=1, antialiased=True, shade=False)
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
mappable.set_array(Bnormal)
plt.colorbar(mappable, ax=ax, shrink=0.8)
mappable.colorbar.set_label('|B*n|')
plt.tight_layout()
plt.legend()
plt.show()

