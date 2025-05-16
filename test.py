from sheet_current import SheetCurrent
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import plot
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



nfp = 1
surf = SurfaceRZFourier(nfp, stellsym=True, mpol=5, ntor=5, 
                        quadpoints_phi=np.linspace(0, 1/nfp, 65, endpoint=False),
                        quadpoints_theta=np.linspace(0, 1, 65, endpoint=False))
surf_outer = surf.copy()
surf_outer.extend_via_normal(surf.get('rc(0,0)') / 8)
I_P = 1e5
M = 7
N = 8
current = SheetCurrent(surf_outer, I_P, M, N)

# current that varies toroidally
current.set('c(0,3)', 1e5)

# compute the current
K = current.current()
K_norm = np.linalg.norm(K, axis=-1)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = surf.gamma()
sc = ax.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], facecolors=plt.cm.viridis(K_norm / K_norm.max()), rstride=1, cstride=1, antialiased=True, shade=False)
plt.colorbar(sc, label='|K|')
plt.show()



# compute the magnetic field
B = current.B(surf.gamma().reshape(-1, 3) )
B_norm = np.linalg.norm(B, axis=1)
print(np.max(B_norm))

# Create a 3D plot
X = surf.gamma()
B_norm = np.linalg.norm(B, axis=1).reshape(X.shape[0], X.shape[1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], facecolors=plt.cm.viridis(B_norm), rstride=1, cstride=1, antialiased=True, shade=False)
plt.colorbar(sc, label='|B|')
plt.show()

# compute squared flux
print(current.squared_flux(surf))