from sheet_current import SheetCurrent
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import plot
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from finite_difference import finite_difference
import numpy as np

def test_names():
    """ Test the names of the current dofs. """

    """ stellsym = True  tests """
    surf = SurfaceRZFourier(1, stellsym=True, mpol=5, ntor=5, 
                            quadpoints_phi=np.linspace(0, 1, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    # M=0, N=0 test
    M = 0
    N = 0
    current = SheetCurrent(surf, 1.0, M, N)
    assert current.local_dof_names == ['c(0,0)'], "Incorrect names"
    assert current.names == ['c(0,0)'], "Incorrect names"

    # M=1, N=0 test
    M = 1
    N = 0
    current = SheetCurrent(surf, 1.0, M, N)
    assert current.local_dof_names == ['c(0,0)', 'c(1,0)'], "Incorrect names"
    assert current.names == ['c(0,0)', 'c(1,0)'], "Incorrect names"

    # M=1, N=0 test
    M = 1
    N = 1
    current = SheetCurrent(surf, 1.0, M, N)
    assert current.local_dof_names == ['c(0,0)', 'c(0,1)', 'c(1,0)', 'c(1,1)'], "Incorrect names"
    assert current.names == ['c(0,0)', 'c(0,1)', 'c(1,0)', 'c(1,1)'], "Incorrect names"

    """ stellsym = False  tests """

    surf = SurfaceRZFourier(1, stellsym=False, mpol=5, ntor=5, 
                            quadpoints_phi=np.linspace(0, 1, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    # M=0, N=0 test
    M = 0
    N = 0
    current = SheetCurrent(surf, 1.0, M, N)
    assert current.local_dof_names == ['c(0,0)'], "Incorrect names"
    assert current.names == ['c(0,0)'], "Incorrect names"

    # M=1, N=0 test
    M = 1
    N = 0
    current = SheetCurrent(surf, 1.0, M, N)
    assert current.local_dof_names == ['c(0,0)', 'c(1,0)', 's(1,0)'], "Incorrect names"
    assert current.names == ['c(0,0)', 'c(1,0)',  's(1,0)'], "Incorrect names"

    # M=1, N=0 test
    M = 1
    N = 1
    current = SheetCurrent(surf, 1.0, M, N)
    sol = ['c(0,0)', 'c(0,1)', 'c(1,0)', 'c(1,1)', 's(0,1)', 's(1,0)', 's(1,1)']
    assert current.local_dof_names == sol, "Incorrect names"
    assert current.names == sol, "Incorrect names"


def test_current():
    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=5, ntor=5, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    I_P = 1.0
    M = 4
    N = 4
    current = SheetCurrent(surf, I_P, M, N)
    current.set('c(0,0)', 1.23)
    current.set('c(1,0)', 2)
    current.set('c(0,1)', 0.01)
    current.set('c(0,3)', 0.01)

    # test the current is tangent
    normal = current.surface.unitnormal() 
    K = current.current()
    K = K / np.linalg.norm(K, axis=-1, keepdims=True)  # normalize K
    err = np.max(np.abs(np.sum(K * normal, axis=-1)))
    print(err)
    assert err < 1e-14, "Current is not tangent to the surface: max error = {}".format(err)


def test_div_curl():
    """ Test the divergence and curl using finite differences. """

    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=False, mpol=3, ntor=3, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    I_P = 1e5
    M = 1
    N = 1
    current = SheetCurrent(surf, I_P, M, N)
    # current.set('c(0,0)', 1.23e5)
    # current.set('c(1,0)', 2e5)
    current.set('c(0,1)', 2e5)
    # current.set('c(0,3)', 0.01 * I_P)

    def magfield(x):
        """ compute the magnetic field at x """
        return current.B(x.reshape(-1, 3)).flatten()

    x0 = np.array([0.5, 0.03, 0.07])

    # compute B
    print("B at x0:")
    print(current.B(x0.reshape(-1, 3)).flatten())

    # check gradB accuracy with finite difference
    gradB_fd = finite_difference(magfield, x0, eps=1e-4)
    print("gradB finite difference:")
    print(gradB_fd)

    print("gradB symmetry error:")
    print(gradB_fd - gradB_fd.T)

    print('divergence of B:')
    print(np.trace(gradB_fd))

    print('curl of B:')
    curlB_fd = np.zeros(3)
    curlB_fd[0] = gradB_fd[2, 1] - gradB_fd[1, 2]
    curlB_fd[1] = gradB_fd[0, 2] - gradB_fd[2, 0]
    curlB_fd[2] = gradB_fd[1, 0] - gradB_fd[0, 1]
    print(curlB_fd)

    # # Create a 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X = surf.gamma()
    # sc = ax.plot_surface(X[:, :, 0], X[:, :, 1], X[:, :, 2], alpha=0.5)
    # ax.scatter(x0[0], x0[1], x0[2], color='r', s=100, label='x0', zorder=100)
    # plt.colorbar(sc, label='|K|')
    # plt.show()

def test_gradB():
    """ Test the gradient of the magnetic field using finite differences. """

    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=5, ntor=5, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    I_P = 1e5
    M = 1
    N = 1
    current = SheetCurrent(surf, I_P, M, N)
    current.set('c(0,0)', 1.23e5)
    current.set('c(1,0)', 2e5)
    current.set('c(0,1)', 2e5)
    # current.set('c(0,3)', 0.01 * I_P)

    def magfield(x):
        """ compute the magnetic field at x """
        return current.B(x.reshape(-1, 3)).flatten()

    x0 = np.array([0.5, 0.03, 0.07])

    # compute gradB
    gradB = current.gradB(x0.reshape(-1, 3))[0]
    print("gradB at x0:")
    print(gradB)

    # check gradB accuracy with finite difference
    gradB_fd = finite_difference(magfield, x0, eps=1e-4)
    print("gradB finite difference:")
    print(gradB_fd)

    print("gradB finite difference error:")
    print(gradB - gradB_fd)


def test_div_curl_simsopt():

    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.geo import create_equally_spaced_curves

    # Create the initial coils:
    ncoils = 4
    nfp = 2
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=1.0, R1=0.3, order=1, numquadpoints=128)
    base_currents = [Current(1e5) for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    bs = BiotSavart(coils)

    x0 = np.array([0.25, 0.2, 0.07])
    bs.set_points(x0.reshape((-1, 3)))

    print("Magnetic field at x0:")
    print(bs.B())

    # check gradB symmetry error
    gradB = bs.dB_by_dX()[0]
    print("gradB at x0:")
    print(gradB)
    print("gradB symmetry error:")
    print(gradB - gradB.T)


if __name__ == "__main__":
    test_names()
    # test_current()
    # test_div_curl()
    # test_gradB()
    # test_div_curl_simsopt()
