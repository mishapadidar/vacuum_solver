import numpy as np
from util import get_fourier_coeffs_2d, get_helical_fourier_coeffs

def test_get_fourier_coeffs_2d():
    """ Test the 2d Fourier transform
    """

    nfp = 3
    theta_period = 1.0
    phi_period = 1.0 / nfp
    nphi = 5
    ntheta = 5
    theta = np.linspace(0, theta_period, ntheta, endpoint=False)
    phi = np.linspace(0, phi_period, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
    fx1d = (0.3 + np.cos(2 * np.pi * nfp * phi) + 0.7 * np.sin(2 * np.pi * nfp * phi) 
            + 0.478 * np.cos(2 * 2 * np.pi * nfp * phi) + 0.22 * np.sin(2 * 2 * np.pi * nfp * phi))
    fx = (0.387
          + 1.03242*np.cos(2 * np.pi * theta_grid)
          + 0.632*np.cos(2 * np.pi * nfp*phi_grid)
          + 1.7*np.sin(2 * np.pi * theta_grid)
        #   + 0.23*np.sin(2 * np.pi * nfp*phi_grid)
          + 6.2*np.cos(2 * np.pi * nfp * phi_grid) * np.cos(2 * np.pi * theta_grid)
        #   + .96*np.cos(2 * 2 * np.pi *  nfp * phi_grid) * np.cos(2 * np.pi * theta_grid)
          + 0.5 * np.sin(2 * np.pi * nfp * phi_grid) *  np.cos(2 * np.pi * (theta_grid))
          + 0.767 * np.sin(2 * 2 * np.pi * nfp * phi_grid) *  np.cos(2 * np.pi * (theta_grid))
          + 0.124 * np.cos(2 * np.pi * nfp * phi_grid) * np.sin(2 * np.pi * (theta_grid))
          + 0.88 * np.sin(2 * np.pi * nfp * phi_grid) * np.sin(2 * np.pi * (theta_grid))
          - 0.74 * np.sin(2 * 2 * np.pi * nfp * phi_grid) * np.sin(2 * np.pi * (theta_grid))
          )

    
    nphi_mode = (nphi-1)//2 + 1
    ntheta_mode =  (ntheta-1)//2 + 1
    ACC = np.random.randn(nphi_mode, ntheta_mode)
    ACS = np.random.randn(nphi_mode, ntheta_mode)
    ACS[:,0] = 0.0 # sin(0) modes
    ASC = np.random.randn(nphi_mode, ntheta_mode)
    ASC[0,:] = 0.0 # sin(0) modes
    ASS = np.random.randn(nphi_mode, ntheta_mode)
    ASS[:,0] = 0.0 # sin(0) modes
    ASS[0,:] = 0.0 # sin(0) modes

    fx = 0.0
    for n in range(nphi_mode):
        for m in range(ntheta_mode):
            fx += ACC[n, m] *  np.cos(2 * np.pi * nfp * n * phi_grid) *  np.cos(2 * np.pi * m * theta_grid)
            fx += ACS[n, m] *  np.cos(2 * np.pi * nfp * n * phi_grid) *  np.sin(2 * np.pi * m * theta_grid)
            fx += ASC[n, m] *  np.sin(2 * np.pi * nfp * n * phi_grid) *  np.cos(2 * np.pi * m * theta_grid)
            fx += ASS[n, m] *  np.sin(2 * np.pi * nfp * n * phi_grid) *  np.sin(2 * np.pi * m * theta_grid)

    A_coscos, A_cossin, A_sincos, A_sinsin = get_fourier_coeffs_2d(fx, phi_period=phi_period, theta_period=theta_period)
    err = np.max(np.abs(ACC - A_coscos))
    # print('A_coscos err', err)
    assert err < 1e-14, "Error in A_coscos is too large"

    err = np.max(np.abs(ACS - A_cossin))
    # print('A_cossin err', err)
    assert err < 1e-14, "Error in A_cossin is too large"

    err = np.max(np.abs(ASC - A_sincos))
    # print('A_sincos err', err)
    assert err < 1e-14, "Error in A_sincos is too large"

    err = np.max(np.abs(ASS - A_sinsin))
    # print('A_sinsin err', err)
    assert err < 1e-14, "Error in A_sinsin is too large"

def test_get_helical_fourier_coeffs():
    """ Test the helical fourier representation
    """

    nfp = 3
    theta_period = 1.0
    phi_period = 1.0 / nfp
    nphi = 5
    ntheta = 5
    theta = np.linspace(0, theta_period, ntheta, endpoint=False)
    phi = np.linspace(0, phi_period, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
    
    nphi_mode = (nphi-1)//2 + 1
    ntheta_mode =  (ntheta-1)//2 + 1
    AC = np.random.randn(nphi_mode, ntheta_mode)
    AS = np.random.randn(nphi_mode, ntheta_mode)
    AS[0,0] = 0.0 # sin(0) modes

    fx = 0.0
    for n in range(nphi_mode):
        for m in range(ntheta_mode):
            fx += AC[n, m] *  np.cos(2 * np.pi * (m * theta_grid - nfp * n * phi_grid))
            fx += AS[n, m] *  np.sin(2 * np.pi * (m * theta_grid - nfp * n * phi_grid))

    A_cos, A_sin = get_helical_fourier_coeffs(fx, phi_period=phi_period, theta_period=theta_period)
    err = np.max(np.abs(AC - A_cos))
    # print('A_cos err', err)
    assert err < 1e-14, "Error in A_cos is too large"
    err = np.max(np.abs(AS - A_sin))
    # print('A_sin err', err)
    assert err < 1e-14, "Error in A_sin is too large"

if __name__ == "__main__":
    test_get_fourier_coeffs_2d()
    test_get_helical_fourier_coeffs()