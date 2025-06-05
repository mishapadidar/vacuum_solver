import numpy as np
from scipy.fft import fft

def rotate_nfp(X, nfp):
    """Rotate a vector field X on a flux surface by 1 field period. Given say the normal 
    vectors X to a flux surface on one field period, rotate_nfp(X, nfp) will yield the normal vectors
    on the next field period.

    Args:
        X (array): 3D array, shape (nphi, ntheta, m) of values on a flux surface. m = 3 for vector
            quantites such as normal vectors.
        nfp (int): number of field periods

    Returns:
        X: 3D array, shape (nphi, ntheta, m) of rotated values.
    """
    
    angle = 2 * np.pi / nfp
    Q = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]]
                    )
    return np.einsum('ij,klj->kli', Q, X)


def finite_difference(f, x, eps=1e-6, *args, **kwargs):
    """Approximate jacobian with central difference.

    Args:
        f (function): function to differentiate, can be scalar valued or 1d-array
            valued.
        x (1d-array): input to f(x) at which to take the gradient.
        eps (float, optional): finite difference step size. Defaults to 1e-6.

    Returns:
        array: Jacobian of f at x. If f returns a scalar, the output will be a 1d-array. Otherwise,
            the output will be a 2d-array with shape (len(f(x)), len(x)).
    """
    jac_est = []
    for i in range(len(x)):
        x[i] += eps
        fx = f(x, *args, **kwargs)
        x[i] -= 2*eps
        fy = f(x, *args, **kwargs)
        x[i] += eps
        jac_est.append((fx-fy)/(2*eps))
    return np.array(jac_est).T

# def get_fourier_coeffs_2d(fx, phi_period=1.0, theta_period=1.0):
#     """
#     Compute 2D Fourier coefficients of a real-valued periodic function f(θ, φ),
#     sampled on a uniform grid over [0, theta_period) x [0, phi_period).

#     Args:
#         fx (2D array): f(phi, theta) evaluated on a grid with shape (N_phi, N_theta).
#         phi_period (float): Period in φ direction.
#         theta_period (float): Period in θ direction.

#     Returns:
#         A (2D array): Cosine coefficients (N_phi, N_theta).
#         B (2D array): Sine coefficients (N_phi, N_theta)
#     """
#     assert (fx.shape[0] % 2 != 0) and (fx.shape[1] % 2 != 0), "Input array must have odd number of points"

#     N_phi, N_theta = fx.shape

#     # fft over theta
#     coeffs = fft(fx, axis=-1) / N_theta
#     M = N_theta // 2 + 1
#     an = coeffs[:, :M].real
#     bn = coeffs[:, :M].imag

#     # fft the real part over phi
#     coeffs = fft(an, axis=0) / N_phi
#     M = N_phi // 2 + 1
#     A_coscos =  2 * coeffs[:M, :].real
#     A_coscos[0,0] = coeffs[0, 0].real
#     A_coscos[1:,1:] *= 2  # double the non-constant modes
#     A_sincos = -2 * coeffs[:M, :].imag
#     A_sincos[1:,1:] *= 2  # double the non-constant modes

#     # fft the real part over phi
#     coeffs = fft(bn, axis=0) / N_phi
#     M = N_phi // 2 + 1
#     A_sinsin =  2 * coeffs[:M, :].imag
#     A_sinsin[1:,1:] *= 2  # double the non-constant modes
#     A_cossin = -2 * coeffs[:M, :].real
#     A_cossin[1:,1:] *= 2  # double the non-constant modes

#     return A_coscos, A_cossin, A_sincos, A_sinsin


def get_fourier_coeffs_1d(fx, period=1.0):
    """Compute the Fourier coefficients of a periodic function,
        f(x) = a0 + sum(a_n * cos(2 * pi * n * x / period) + b_n * sin(2 * pi * n * x / period))

    Args:
        fx (array): f(x) sampled at N uniform points in [0, period),
                    i.e., x = np.linspace(0, period, N, endpoint=False)
        period (float): Period of the function.

    Returns:
        a0 (float): DC component.
        an (ndarray): Cosine coefficients a_n for n=1 to M (M=N//2 or (N-1)//2).
        bn (ndarray): Sine coefficients b_n for n=1 to M.
                      (No sine term for Nyquist mode if N is even.)
    """
    # division by len(fx) usually happens in ifft
    N = len(fx)
    coeffs = fft(fx) / N
    a0 = coeffs[0].real

    if N % 2 != 0:
        # odd number of points
        M = N // 2 + 1
        # multiply by 2 because we only take the positive modes
        an = 2 * coeffs[1:M].real
    else:
        # for an even number of points, there is an additional cosine mode.
        M = N // 2
        an = 2 * coeffs[1:M].real
        a_nyquist = coeffs[M].real
        an = np.concatenate((an, [a_nyquist]))

    bn = -2 * coeffs[1:M].imag # -2 b/c of conjugate
    return a0, an, bn

def get_fourier_coeffs_2d(fx, phi_period=1.0, theta_period=1.0):
    """
    Compute 2D Fourier coefficients of a real-valued periodic function f(θ, φ),
    sampled on a uniform grid over [0, theta_period) x [0, phi_period).

    Args:
        fx (2D array): f(phi, theta) evaluated on a grid with shape (N_phi, N_theta).
        phi_period (float): Period in φ direction.
        theta_period (float): Period in θ direction.

    Returns:
        A_coscos (2D array): coefficients for cos(φ) * cos(θ).
        A_cossin (2D array): coefficients for cos(φ) * sin(θ).
        A_sincos (2D array): coefficients for sin(φ) * cos(θ).
        A_sinsin (2D array): coefficients for sin(φ) * sin(θ).
    """
    assert (fx.shape[0] % 2 != 0) and (fx.shape[1] % 2 != 0), "Input array must have odd number of points"

    N_phi, N_theta = fx.shape

    # fft over theta
    coeffs = fft(fx, axis=-1) / N_theta
    M = N_theta // 2 + 1
    an = coeffs[:, :M].real
    bn = coeffs[:, :M].imag

    # fft the real part over phi
    coeffs = fft(an, axis=0) / N_phi
    M = N_phi // 2 + 1
    A_coscos =  2 * coeffs[:M, :].real
    A_coscos[0,0] = coeffs[0, 0].real
    A_coscos[1:,1:] *= 2  # double the non-constant modes
    A_sincos = -2 * coeffs[:M, :].imag
    A_sincos[1:,1:] *= 2  # double the non-constant modes

    # fft the real part over phi
    coeffs = fft(bn, axis=0) / N_phi
    M = N_phi // 2 + 1
    A_sinsin =  2 * coeffs[:M, :].imag
    A_sinsin[1:,1:] *= 2  # double the non-constant modes
    A_cossin = -2 * coeffs[:M, :].real
    A_cossin[1:,1:] *= 2  # double the non-constant modes

    return A_coscos, A_cossin, A_sincos, A_sinsin


def get_helical_fourier_coeffs(fx, phi_period=1.0, theta_period=1.0):
    """Compute the Fourier coefficients for the helical fourier series approximation to f(phi, theta),
        f(phi, theta) ~ sum_{n,m} A_nm * cos(2 * pi * (m * theta / theta_period - n * phi / phi_period))
                        + B_nm * sin(2 * pi * (m * theta / theta_period - n * phi / phi_period)).

    Args:
        fx (2D array): f(phi, theta) evaluated on a grid with shape (nphi,  ntheta).
        phi_period (float): Period in φ direction.
        theta_period (float): Period in θ direction.

    Returns:
        A_cosdiff (2D array): (nphi, ntheta) array of coefficients for cos(2pi(m * theta - n * phi)).
        A_sindiff (2D array): (nphi, ntheta) array of coefficients for sin(2pi(m * theta - n * phi)).
    """
    assert (fx.shape[0] % 2 != 0) and (fx.shape[1] % 2 != 0), "Input array must have odd number of points"

    A_coscos, A_cossin, A_sincos, A_sinsin = get_fourier_coeffs_2d(fx, phi_period=1.0, theta_period=1.0)

    shape = A_coscos.shape
    # norm squared of the cos(2pi * n * phi) * cos(2 pi * m * theta) modes
    coscos_norm_squared = np.ones(shape)
    coscos_norm_squared[0, 1:] = theta_period / 2
    coscos_norm_squared[1:, 0] = phi_period / 2 
    coscos_norm_squared[1:, 1:] = theta_period * phi_period / 4

    # norm squared of the cos(2pi * n * phi) * sin(2 pi * m * theta) modes
    cossin_norm_squared = np.zeros(shape)
    cossin_norm_squared[0, 1:] = theta_period / 2
    cossin_norm_squared[1:, 1:] = theta_period * phi_period / 4

    # norm squared of the sin(2pi * n * phi) * cos(2 pi * m * theta) modes
    sincos_norm_squared = np.zeros(shape)
    sincos_norm_squared[1:, 0] = phi_period / 2 
    sincos_norm_squared[1:, 1:] = theta_period * phi_period / 4

    # norm squared of the sin(2pi * n * phi) * sin(2 pi * m * theta) modes
    sinsin_norm_squared = np.zeros(shape)
    sinsin_norm_squared[1:, 1:] = theta_period * phi_period / 4 

    # norm squared of the helical modes
    cos_helical_norm_squared = np.ones(shape)
    cos_helical_norm_squared[0, 1:] = theta_period / 2
    cos_helical_norm_squared[1:, 0] = phi_period / 2 
    cos_helical_norm_squared[1:, 1:] = theta_period * phi_period / 2 
    sin_helical_norm_squared = cos_helical_norm_squared

    # coefficient = <f, cos(theta - phi)> / ||cos(theta - phi)||^2
    A_cos_helical = (A_coscos * coscos_norm_squared + A_sinsin * sinsin_norm_squared) / cos_helical_norm_squared
    # coefficient = <f, sin(theta - phi)> / ||cos(theta - phi)||^2
    A_sin_helical = (A_cossin * cossin_norm_squared - A_sincos * sincos_norm_squared) / sin_helical_norm_squared

    return A_cos_helical, A_sin_helical
