import numpy as np
from simsopt.geo import SurfaceRZFourier
from sheet_current import SheetCurrent
from util import finite_difference
from unittest.mock import patch

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
    assert current.local_dof_names == ['s(0,0)'], "Incorrect names"
    assert current.names == ['s(0,0)'], "Incorrect names"

    # M=1, N=0 test
    M = 1
    N = 0
    current = SheetCurrent(surf, 1.0, M, N)
    dofs = ['s(0,0)', 's(1,0)']
    assert current.local_dof_names == dofs, "Incorrect names"
    assert current.names == dofs, "Incorrect names"

    # M=1, N=0 test
    M = 1
    N = 1
    current = SheetCurrent(surf, 1.0, M, N)
    dofs = ['s(0,0)', 's(0,1)', 's(1,0)', 's(1,1)']
    assert current.local_dof_names == dofs, "Incorrect names"
    assert current.names == dofs, "Incorrect names"

    """ stellsym = False  tests """

    surf = SurfaceRZFourier(1, stellsym=False, mpol=5, ntor=5, 
                            quadpoints_phi=np.linspace(0, 1, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    # M=0, N=0 test
    M = 0
    N = 0
    current = SheetCurrent(surf, 1.0, M, N)
    dofs = ['s(0,0)']
    assert current.local_dof_names == dofs, "Incorrect names"
    assert current.names == dofs, "Incorrect names"

    # M=1, N=0 test
    M = 1
    N = 0
    current = SheetCurrent(surf, 1.0, M, N)
    dofs = ['s(0,0)', 's(1,0)', 'c(1,0)']
    assert current.local_dof_names == dofs, "Incorrect names"
    assert current.names == dofs, "Incorrect names"

    # M=1, N=0 test
    M = 1
    N = 1
    current = SheetCurrent(surf, 1.0, M, N)
    sol = ['s(0,0)', 's(0,1)', 's(1,0)', 's(1,1)', 'c(0,1)', 'c(1,0)', 'c(1,1)']
    assert current.local_dof_names == sol, "Incorrect names"
    assert current.names == sol, "Incorrect names"


def test_current():
    """ Test the current is tangent to the surface. """
    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=5, ntor=5, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    G = 1.0
    M = 4
    N = 4
    current = SheetCurrent(surf, G, M, N)
    current.set('s(0,0)', 1.23)
    current.set('s(1,0)', 2)
    current.set('s(0,1)', 0.01)
    current.set('s(0,3)', 0.01)

    # test the current is tangent
    normal = current.surface.unitnormal() 
    K = current.current()
    K = K / np.linalg.norm(K, axis=-1, keepdims=True)  # normalize K
    err = np.max(np.abs(np.sum(K * normal, axis=-1)))
    assert err < 1e-14, "Current is not tangent to the surface: max error = {}".format(err)

    """ Test the current is stellarator symmetric"""
    nfp = 2
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=2, ntor=2, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 64, endpoint=True), # endpoint=True to include 1/nfp
                            quadpoints_theta=np.linspace(0, 1, 64, endpoint=True))
    surf.set('rc(0,1)', 1.0)
    surf.set('rc(1,0)', 0.1)
    surf.set('zs(0,1)', 0.1)
    G = 1e-5
    M = 4
    N = 4
    current = SheetCurrent(surf, G, M, N)
    current.set('s(0,0)', 1.23e-7)
    current.set('s(1,0)', 2e-7) # make sure entries are O(1) for floating point error
    current.set('s(0,1)', 1e-7)
    current.set('s(0,3)', 1.23e-7)
    K = current.current() # (nphi, ntheta, 3)

    # check for stellarator symmetry
    modK = np.linalg.norm(K, axis=-1)
    err = modK - modK[::-1,::-1]
    err = np.max(np.abs(err))
    # error is sensitive to scale of G and modes etc
    assert err < 3e-14, "Current is not stellarator symmetric: max error = {}".format(err)



def test_B():
    """ Test the divergence and curl of B using finite differences. """

    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=False, mpol=3, ntor=3, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    G = 1e0
    M = 3
    N = 3
    current = SheetCurrent(surf, G, M, N)
    current.set('s(0,0)', 1.23)
    current.set('s(1,0)', 2)
    current.set('s(0,1)', 2)
    current.set('c(0,1)', 2)
    current.set('s(0,3)', 0.01 * G)

    def magfield(x):
        """ compute the magnetic field at x """
        return current.B(x.reshape(-1, 3)).flatten()

    x0 = np.array([0.5, 0.03, 0.07])

    # check gradB accuracy with finite difference
    gradB_fd = finite_difference(magfield, x0, eps=1e-6)

    # test divergence is zero
    err = np.trace(gradB_fd)
    print('divergence err', err)
    assert np.abs(err) < 1e-8, "Divergence of B is not zero: error = {}".format(err)

    # test curl is zero
    curlB_fd = np.zeros(3)
    curlB_fd[0] = gradB_fd[2, 1] - gradB_fd[1, 2]
    curlB_fd[1] = gradB_fd[0, 2] - gradB_fd[2, 0]
    curlB_fd[2] = gradB_fd[1, 0] - gradB_fd[0, 1]
    err = np.max(np.abs(curlB_fd))
    print('curlB_fd err', err)
    assert err < 1e-8, "Curl of B is not zero: max error = {}".format(err)

    """ Now test with multiple nfp"""
    nfp = 2
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=3, ntor=3, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    G = 1e0
    M = 3
    N = 3
    current = SheetCurrent(surf, G, M, N)
    current.set('s(0,0)', 1.23)
    current.set('s(1,0)', 2)
    current.set('s(0,1)', 2)
    current.set('s(0,3)', 0.01 * G)

    def magfield(x):
        """ compute the magnetic field at x """
        return current.B(x.reshape(-1, 3)).flatten()

    x0 = np.array([0.5, 0.03, 0.07])

    # check gradB accuracy with finite difference
    gradB_fd = finite_difference(magfield, x0, eps=1e-5)

    # test divergence is zero
    err = np.trace(gradB_fd)
    print('divergence err', err)
    assert np.abs(err) < 1e-8, "Divergence of B is not zero: error = {}".format(err)

    # test curl is zero
    curlB_fd = np.zeros(3)
    curlB_fd[0] = gradB_fd[2, 1] - gradB_fd[1, 2]
    curlB_fd[1] = gradB_fd[0, 2] - gradB_fd[2, 0]
    curlB_fd[2] = gradB_fd[1, 0] - gradB_fd[0, 1]
    err = np.max(np.abs(curlB_fd))
    print('curlB_fd err', err)
    assert err < 1e-8, "Curl of B is not zero: max error = {}".format(err)


    """ Test the field is stellarator symmetric"""
    nfp = 2
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=2, ntor=2, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 64, endpoint=True), # endpoint=True to include 1/nfp
                            quadpoints_theta=np.linspace(0, 1, 64, endpoint=True))
    surf.set('rc(0,1)', 1.0)
    surf.set('rc(1,0)', 0.1)
    surf.set('zs(0,1)', 0.1)
    G = 1e-5
    M = 4
    N = 4
    surf_winding = surf.copy(range='field period')
    surf_winding.extend_via_normal(surf_winding.minor_radius())
    current = SheetCurrent(surf_winding, G, M, N)
    current.set('s(0,0)', 1.23e-7)
    current.set('s(1,0)', 2e-7) # make sure entries are O(1) for floating point error
    current.set('s(0,1)', 1e-7)
    current.set('s(0,3)', 1.23e-7)

    # check for stellarator symmetry
    X = surf.gamma()
    B = current.B(X.reshape((-1, 3))) # (nphi, ntheta, 3)
    modB = np.linalg.norm(B, axis=-1).reshape(X.shape[:-1])
    err = modB - modB[::-1,::-1]
    err = np.max(np.abs(err))
    print('stellsym error', err)
    # error is sensitive to scale of G and modes etc
    assert err < 3e-14, "Current is not stellarator symmetric: max error = {}".format(err)


def test_gradB():
    """ Test the gradient of the magnetic field using finite differences. """

    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=False, mpol=5, ntor=5, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    G = 1e0
    M = 3
    N = 3
    current = SheetCurrent(surf, G, M, N)
    current.set('s(0,0)', 1.23)
    current.set('s(1,0)', 2)
    current.set('s(0,1)', 2)
    current.set('c(0,1)', 2)
    current.set('s(0,3)', 0.01 * G)

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
    print('gradB_fd err', err)
    assert err < 1e-8, "Gradient of B is not accurate: max error = {}".format(err)

    """ Now test with multiple nfp"""

    nfp = 2
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=5, ntor=5, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    G = 1e0
    M = 3
    N = 3
    current = SheetCurrent(surf, G, M, N)
    current.set('s(0,0)', 1.23)
    current.set('s(1,0)', 2)
    current.set('s(0,1)', 2)
    current.set('s(0,3)', 0.01 * G)

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
    print('gradB_fd err', err)
    assert err < 1e-8, "Gradient of B is not accurate: max error = {}".format(err)



def test_fit():
    """ test the .fit method of SheetCurrent """

    # Create a surface and SheetCurrent instance
    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=5, ntor=5, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 128, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    G = 1.0
    M = 2
    N = 2
    current = SheetCurrent(surf, G, M, N, jit = 0.0)

    """ Test a linear system with a unique solution"""

    # Mock the build_linear_system method
    np.random.seed(0)
    H = np.random.randn(current.n_dofs, current.n_dofs)
    y = np.random.randn(current.n_dofs)

    with patch.object(SheetCurrent, 'build_linear_system', return_value=(H, y)) as mock_method:

        # Call the fit method
        w = current.fit(surf)

        # Verify the mocked build_linear_system was called
        SheetCurrent.build_linear_system.assert_called_once()

        # Check the result
        w_actual = np.linalg.solve(H, y)

        assert np.allclose(w, w_actual, atol=1e-13), "The fit method did not return the expected result."

    """ Test a least squares problem"""

    jit = 1e-6
    current = SheetCurrent(surf, G, M, N, jit = jit)

    # Mock the build_linear_system method
    np.random.seed(0)
    H = np.random.randn(30,current.n_dofs)
    y = np.random.randn(30)
    with patch.object(SheetCurrent, 'build_linear_system', return_value=(H, y)) as mock_method:

        # Call the fit method
        w = current.fit(surf)

        # Verify the mocked build_linear_system was called
        SheetCurrent.build_linear_system.assert_called_once()

        # Check the result by solving normal equations
        w_actual = np.linalg.solve((H.T @ H + jit * np.eye(current.n_dofs)), H.T @ y)
        assert np.allclose(w, w_actual, atol=1e-14), "The fit method did not return the expected result."


def test_secular_field():
    """ Test the secular magnetic field in the SheetCurrent class. """
    # Create a surface
    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=2, ntor=2, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 17, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 17, endpoint=False))
    
    # we need toroidal shaping to test secular term
    surf.set('rc(0,0)', 1.0)
    surf.set('rc(0,1)', 0.1)
    surf.set('zs(0,2)', 0.1)

    # initialize the SheetCurrent
    surf_outer = surf.copy()
    surf_outer.extend_via_normal(surf.get('rc(0,0)') / 8)
    G = 1.2
    M = 2
    N = 2
    current = SheetCurrent(surf_outer, G, M, N, jit = 0.0)

    # biot savart returns secular term when fourier coeffs are zero
    X = surf.gamma().reshape(-1, 3) # (nphi * ntheta, 3)
    nhat = surf.unitnormal().reshape(-1, 3) # (nphi * ntheta, 3)
    h_secular = current.B_normal(surf).reshape(-1)  # (nphi * ntheta,)

    # compute secular term
    current.biot_savart_precomputation()
    for ii, x_target in enumerate(X):
        # alternative computation of secular term
        h_secular_alt = current.compute_h_secular(x_target, nhat[ii])
        err = h_secular[ii] - h_secular_alt
        # print("Secular term error at point {}: {}".format(ii, err))
        assert err < 1e-14, "Secular term is not correct: error = {}".format(err)

def test_fourier_field():
    """ Test the fourier magnetic field in the SheetCurrent class. """
    # Create a surface
    nfp = 3
    surf = SurfaceRZFourier(nfp, stellsym=False, mpol=2, ntor=2, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 7, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 7, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    surf.set('rc(0,1)', 0.1)
    surf.set('zs(0,2)', 0.1)

    # initialize the SheetCurrent
    surf_outer = surf.copy()
    surf_outer.extend_via_normal(surf.get('rc(0,0)') / 8)
    
    # set G to zero to remove secular term
    G = 0.0

    current = SheetCurrent(surf_outer, G, 2, 2, jit = 0.0)

    # only turn on one fourier term
    current.set('s(1,2)', 1.1)
    current.set('s(1,0)', 1.2)
    current.set('s(0,1)', 1.3)
    current.set('s(1,1)', 0.932)
    current.set('c(1,1)', 0.932)

    # compute B*n
    B_dot_n = current.B_normal(surf).reshape(-1)  # (nphi * ntheta,)

    # alternative computation of B*n
    X = surf.gamma().reshape(-1, 3) # (nphi * ntheta, 3)
    nhat = surf.unitnormal().reshape(-1, 3) # (nphi * ntheta, 3)
    current.biot_savart_precomputation()
    for ii, x_target in enumerate(X):
        h_fourier_alt = current.compute_h_fourier(x_target, nhat[ii])
        B_dot_n_alt = np.sum(h_fourier_alt * current.local_full_x) # since we only have one fourier term
        err = np.abs(B_dot_n[ii] - B_dot_n_alt)
        assert err < 1e-14, "Fourier term is not correct: error = {}".format(err)


def test_linear_system():
    """ test the linear system method """

    # Create a surface
    nfp = 1
    surf = SurfaceRZFourier(nfp, stellsym=True, mpol=2, ntor=2, 
                            quadpoints_phi=np.linspace(0, 1/nfp, 17, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 17, endpoint=False))
    surf.set('rc(0,0)', 1.0)
    surf.set('rc(0,1)', 0.1)
    surf.set('zs(0,2)', 0.1)

    # initialize the SheetCurrent
    surf_outer = surf.copy()
    surf_outer.extend_via_normal(surf.get('rc(0,0)') / 8)
    G = 1.2
    M = 2
    N = 2
    current = SheetCurrent(surf_outer, G, M, N, jit = 0.0)
    current.set('s(0,0)', 1.0)  # set the current at the M=0, N=0 mode
    current.set('s(0,1)', 1.1)  # set the current at the M=1, N=0 mode to zero

    H, y = current.build_linear_system(surf)

    """ check that B*n is consistent"""
    B_dot_n = current.B_normal(surf).reshape(-1)  # (nphi * ntheta,)

    # alternative computation of B*n
    dphi = np.diff(surf.quadpoints_phi)[0]
    dtheta = np.diff(surf.quadpoints_theta)[0]
    normal = surf.normal() # (nphi, ntheta, 3)
    dS = dphi * dtheta * np.linalg.norm(normal, axis=-1).reshape(-1) # (nphi, ntheta)
    sqrt_dS = np.sqrt(dS) # (nphi * ntheta)
    B_dot_n_alt = (H @ current.local_full_x - y)/sqrt_dS
    err = np.max(np.abs(B_dot_n - B_dot_n_alt))
    assert err < 1e-14, "B*n from linear system does not match actual B*n: max error = {}".format(err)

    """ check that the linear least squares objective is consistent with the squared flux """
    squared_flux = current.squared_flux(surf)
    squared_flux_ls = np.linalg.norm(H @ current.local_full_x - y)**2
    err = np.abs(squared_flux - squared_flux_ls)
    assert err < 1e-14, "squared flux from linear system does not match actual squared flux"


if __name__ == "__main__":
    test_names()
    test_current()
    test_B()
    test_gradB()
    test_fit()
    test_secular_field()
    test_fourier_field()
    test_linear_system()
