import numpy as np

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