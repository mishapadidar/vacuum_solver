import numpy as np
from simsopt.geo import SurfaceRZFourier
from sheet_current import SheetCurrent
from util import finite_difference

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
    """ Test the current is tangent to the surface. """
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
    assert err < 1e-14, "Current is not tangent to the surface: max error = {}".format(err)


def test_div_curl():
    """ Test the divergence and curl using finite differences. """

    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=False, mpol=3, ntor=3, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    I_P = 1e5
    M = 3
    N = 3
    current = SheetCurrent(surf, I_P, M, N)
    current.set('c(0,0)', 1.23e5)
    current.set('c(1,0)', 2e5)
    current.set('c(0,1)', 2e5)
    current.set('c(0,3)', 0.01 * I_P)

    def magfield(x):
        """ compute the magnetic field at x """
        return current.B(x.reshape(-1, 3)).flatten()

    x0 = np.array([0.5, 0.03, 0.07])

    # check gradB accuracy with finite difference
    gradB_fd = finite_difference(magfield, x0, eps=1e-5)

    # test divergence is zero
    err = np.trace(gradB_fd)
    assert np.abs(err) < 1e-8, "Divergence of B is not zero: error = {}".format(err)

    # test curl is zero
    curlB_fd = np.zeros(3)
    curlB_fd[0] = gradB_fd[2, 1] - gradB_fd[1, 2]
    curlB_fd[1] = gradB_fd[0, 2] - gradB_fd[2, 0]
    curlB_fd[2] = gradB_fd[1, 0] - gradB_fd[0, 1]
    err = np.max(np.abs(curlB_fd))
    assert err < 1e-8, "Curl of B is not zero: max error = {}".format(err)


def test_gradB():
    """ Test the gradient of the magnetic field using finite differences. """

    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=5, ntor=5, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    I_P = 1e5
    M = 3
    N = 3
    current = SheetCurrent(surf, I_P, M, N)
    current.set('c(0,0)', 1.23e5)
    current.set('c(1,0)', 2e5)
    current.set('c(0,1)', 2e5)
    current.set('c(0,3)', 0.01 * I_P)

    def magfield(x):
        """ compute the magnetic field at x """
        return current.B(x.reshape(-1, 3)).flatten()

    x0 = np.array([0.5, 0.03, 0.07])

    # compute gradB
    gradB = current.gradB(x0.reshape(-1, 3))[0]

    # check gradB accuracy with finite difference
    gradB_fd = finite_difference(magfield, x0, eps=1e-5)

    # check the gradient of B is accurate
    err = np.max(np.abs(gradB - gradB_fd))
    assert err < 1e-8, "Gradient of B is not accurate: max error = {}".format(err)


if __name__ == "__main__":
    test_names()
    test_current()
    test_div_curl()
    test_gradB()
