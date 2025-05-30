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
I_P = 1e5
M = 10
N = 10
current = SheetCurrent(surf_outer, I_P, M, N, jit=1e-12)
current.set('s(0,0)', 1.e5)
current.set('s(0,1)', 1.e5)
current.set('s(1,0)', 1.e5)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = surf.gamma()
sc = ax.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2])
X = surf_outer.gamma()
sc = ax.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], alpha=0.5)
plt.tight_layout()
plt.show()



# check the squared flux
print('initial squared flux:')
print(current.squared_flux(surf))

current.fit(surf)

# check the squared flux
print('final squared flux:')
print(current.squared_flux(surf))

K = current.current()
print('final mean squared current:')
print(np.mean(np.linalg.norm(K, axis=-1)**2))


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# copy the surface with more points for better visualization
surf = surf.copy(nphi=128, ntheta=128)
X = surf.gamma()
Bnormal = current.B_normal(surf)
norm = plt.Normalize(vmin=Bnormal.min(), vmax=Bnormal.max())
colors = plt.cm.viridis(norm(Bnormal))
sc = ax.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], facecolors=colors, rstride=1, cstride=1, antialiased=True, shade=False)
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
mappable.set_array(Bnormal)
plt.colorbar(mappable, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

