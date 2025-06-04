import numpy as np
from simsopt._core import Optimizable
from util import rotate_nfp


class SheetCurrent(Optimizable):
    def __init__(self, surface, G, M=8, N=8, jit=1e-13):
        """
        Initialize the sheet current with the given parameters.

        The sheet current is defined as:
            K = n x grad(Phi)
        where 
            Phi = tilde{Phi} + G * phi
        and tilde{Phi} is doubly periodic on (theta, phi) in [0, 1] x [0, 1/nfp].
        Boozer's G is defined as 
            G = mu0 * I_P / (2 * pi),
        where I_P is the sum of the currents through the hole in the torus.

        NOTE:
            In Simsopt, angles run from 0 to 1, so by Amperes law, G = mu0 * I_P.

        This class is often used to solve to find vacuum equilibrium for a given boundary.

        Example usage:
            coils = ...
            surf = ...
            I_P = np.abs(sum([np.abs(coil.current.get_value()) for coil in coils]))
            mu0 =  4 * np.pi * 10**(-7)
            G = I_P * mu0 
            surf_winding = surf.to_RZFourier().copy(ntheta=2 * M + 1, nphi=2 * N + 1, range='field period')
            # NOTE: make sure the normal is outward facing!!!
            surf_winding.extend_via_normal(2 * surf.minor_radius())
            current = SheetCurrent(surf_winding, G, M, N)
            current.fit(surf)
            print(current.squared_flux(surf))
        
        Parameters:
            surface (Surface): A Simsopt surface object.
            G (float): Normalized poloidal current.
            phi (float): The potential function.
            M,N (int): Highest fourier mode number (inclusive) in the poloidal and toroidal directions.
            jit (float): Tikhonov regularization parameter for the least squares problem.
        """
        self.surface = surface
        self.nfp = surface.nfp
        self.stellsym = surface.stellsym
        self.G = G
        self.M = M # poloidal
        self.N = N # toroidal
        self.jit = jit # regularization parameter for the least squares problem

        # constants
        self.mu0 =  1.256637061e-6 # N / A^2
        self.mu0_over_4pi = self.mu0 / (4 * np.pi) 

        self._set_names()
        
        Optimizable.__init__(self, x0=np.zeros(self.n_dofs),
                             names=self.names, depends_on=[surface])

    def _set_names(self):
        """
        Set the list of names for each degree of freedom.

        The degrees of freedom are the coefficients of the Fourier series
        for the layer potential,
            x = [s(m,n), ... , c(m,n)],
        where s(m,n) and c(m,n) are the coefficients of the sine and cosine modes, respectively.
        In stellarator symmetry, only the sine modes are used, so that the current (not potential)
        is stell symmetric. We always omit the c(0,0) mode, since it has no effect on the
        current.
        """
        names = []

        for m in range(self.M+1):
            for n in range(self.N+1):
                names += ['s({},{})'.format(m, n)]
        if not self.stellsym:
            for m in range(self.M+1):
                for n in range(self.N+1):
                    if m == 0 and n == 0:
                        continue
                    names += ['c({},{})'.format(m, n)]
        self.names = names
        self.n_dofs = len(self.names)

    def recompute_bell(self, parent=None):
        """
        This function will get called any time any of the DOFs of the
        parent class change.
        """
        self.need_to_run_code = True
        return super().recompute_bell(parent)
    
    def potential(self):
        """ Compute the potential at the surface quadrature points
            Phi = tilde{Phi} + G * phi
        and tilde{Phi} is doubly periodic on (theta, phi) in [0, 1] x [0, 1/nfp].
            tilde{Phi} = sum_{m=0}^{M} sum_{n=0}^{N} c(m,n) * cos(2 * pi (m * theta - nfp * n * phi )) 
                        + s(m,n) * sin(2 * pi (m * theta - nfp * n * phi ))

        Returns:
            np.ndarray: (nphi, ntheta) array of the potential function at the surface quadrature points.
        """
        # Get the surface quadrature points
        theta1d = self.surface.quadpoints_theta
        phi1d = self.surface.quadpoints_phi

        phis, thetas = np.meshgrid(phi1d, theta1d, indexing='ij')

        dofs = self.local_full_x
        
        # storage
        pot = np.zeros(np.shape(phis)) # (nphi, ntheta)

        # Calculate the potential function
        idx = 0
        # sine modes
        for m in range(self.M+1):
            for n in range(self.N+1):
                pot += dofs[idx] * np.sin(2 * np.pi * (m * thetas - self.nfp * n * phis))
                idx += 1

        # cos modes
        if not self.stellsym:
            for m in range(self.M+1):
                for n in range(self.N+1):
                    if m==0 and n==0:
                        continue
                    pot += dofs[idx] * np.cos(2 * np.pi * (m * thetas - self.nfp * n * phis))
                    idx += 1
        
        # secular term
        pot += (self.G) * phis
        return pot
    
    def current(self):
        """ Compute the sheet current at the quadrature points on S'. This function evaluates
            K * dS' at the surface quadrature points, where K is the sheet current and S' is the
            winding surface.

        Simsopt surfaces obey the ordering (phi, theta), so,
            n = dr/dphi x dr/dtheta.

        We use the fact that
            n x grad(theta) = -dr/dphi
            n x grad(phi) = dr/dtheta

        The potential is given by,
            Phi = tilde{Phi} + G * phi
        and
            tilde{Phi} = sum_{m,n} c(m,n) * cos(alpha_mn) + s(m,n) * sin(alpha_mn),
        where
            alpha_mn = 2 * pi (m * theta - nfp * n * phi ).
        The sheet current is,
            mu0 * K * dS' = n x [grad(tilde{Phi}) + G * grad(phi)]
              = G * dr/dtheta + 
                + sum_{m,n} [-c(m,n) * sin(alpha_mn) + s(m,n) * cos(alpha_mn)] * (n x grad(alpha_mn))
        where
            n x grad(alpha_mn) = 2 * pi (m * n x grad(theta) - nfp * n * n x grad(phi) )
                               = 2 * pi (-m * dr/dphi - nfp * n * dr/dtheta ).

        Returns:
            np.ndarray: (ntheta, nphi, 3) array of the current at the surface quadrature points.
        """
        # Get the surface quadrature points
        phi1d = self.surface.quadpoints_phi
        theta1d = self.surface.quadpoints_theta
        phis, thetas = np.meshgrid(phi1d, theta1d, indexing='ij')

        dofs = self.local_full_x
        
        # compute grad(theta) and grad(phi)
        dr_by_dphi = self.surface.gammadash1() # (nphi, ntheta, 3)
        dr_by_dtheta = self.surface.gammadash2() # (nphi, ntheta, 3)
        n_cross_grad_phi = dr_by_dtheta
        n_cross_grad_theta = - dr_by_dphi

        
        # storage
        K = np.zeros((*np.shape(thetas), 3)) # (nphi, ntheta, 3)

        # Calculate the current
        idx = 0
        # sine modes
        for m in range(self.M+1):
            for n in range(self.N+1):
                alpha = 2 * np.pi * (m * thetas - self.nfp * n * phis) # (nphi, ntheta)
                n_cross_grad_alpha = 2 * np.pi * (m * n_cross_grad_theta  - self.nfp * n * n_cross_grad_phi) # (nphi, ntheta, 3)
                K += dofs[idx] * np.cos(alpha)[:,:,None] * n_cross_grad_alpha # (nphi, ntheta, 3)
                idx += 1
        # cos modes
        if not self.stellsym:
            for m in range(self.M+1):
                for n in range(self.N+1):
                    if m == 0 and n == 0:
                        continue
                    alpha = 2 * np.pi * (m * thetas - self.nfp * n * phis) # (nphi, ntheta)
                    n_cross_grad_alpha = 2 * np.pi * (m * n_cross_grad_theta  - self.nfp * n * n_cross_grad_phi) # (nphi, ntheta, 3)
                    K += - dofs[idx] * np.sin(alpha)[:,:,None] * n_cross_grad_alpha # (nphi, ntheta, 3)
                    idx += 1
            
        # secular term
        K += self.G * n_cross_grad_phi

        # scale by mu0
        K = K / self.mu0 

        return K
    
    def B(self, X):
        """Compute the magnetic field at a set of points X using the Biot-Savart law.

        Parameters:
            X (np.ndarray): (n, 3) array of points where the magnetic field is computed.

        Returns:
            np.ndarray: (n, 3) array of the magnetic field at the points X.
        """

        # compute the sheet current
        K_1fp = self.current() # (nphi, ntheta, 3)

        # get the quadrature points
        quadpoints_1fp = self.surface.gamma() # (nphi, ntheta, 3)
        nphi, ntheta, _ = quadpoints_1fp.shape

        # rotate to get full torus
        quadpoints = np.zeros((self.nfp * nphi, ntheta, 3))
        K = np.zeros((self.nfp * nphi, ntheta, 3))
        for ii in range(self.nfp):
            quadpoints_1fp = rotate_nfp(quadpoints_1fp, self.nfp)
            quadpoints[ii * nphi:(ii + 1) * nphi, :, :] = quadpoints_1fp
            K_1fp = rotate_nfp(K_1fp, self.nfp)
            K[ii * nphi:(ii + 1) * nphi, :, :] = K_1fp

        dphi = np.diff(self.surface.quadpoints_phi)[0]
        dtheta = np.diff(self.surface.quadpoints_theta)[0]
        dA = dphi * dtheta

        # compute the magnetic field using the Biot-Savart law
        B = np.zeros(np.shape(X))
        for i in range(X.shape[0]):
            diff = X[i] - quadpoints # (nphi, ntheta, 3)
            dist = np.linalg.norm(diff, axis=-1, keepdims=True) # (nphi, ntheta, 1)
            kernel = diff / (dist**3) # (nphi, ntheta, 3)
            cross = np.cross(K, kernel, axis=-1) # (nphi, ntheta, 3)
            B[i] = self.mu0_over_4pi * np.sum(cross * dA, axis=(0, 1))

        return B

    def gradB(self, X):
        """Compute the gradient of the magnetic field at a set of points X
        using the Biot-Savart law.

        X should not be placed on the flux surface, as the Biot-Savart law will be singular.

        Parameters:
            X (np.ndarray): (n, 3) array of points where the magnetic field is computed.

        Returns:
            np.ndarray: (n, 3, 3) array of the gradient of the magnetic field at the points X.
        """

        # compute the sheet current
        K_1fp = self.current() # (nphi, ntheta, 3)

        # get the quadrature points
        quadpoints_1fp = self.surface.gamma() # (nphi, ntheta, 3)
        nphi, ntheta, _ = quadpoints_1fp.shape

        # rotate to get full torus
        quadpoints = np.zeros((self.nfp * nphi, ntheta, 3))
        K = np.zeros((self.nfp * nphi, ntheta, 3))
        for ii in range(self.nfp):
            quadpoints_1fp = rotate_nfp(quadpoints_1fp, self.nfp)
            quadpoints[ii * nphi:(ii + 1) * nphi, :, :] = quadpoints_1fp
            K_1fp = rotate_nfp(K_1fp, self.nfp)
            K[ii * nphi:(ii + 1) * nphi, :, :] = K_1fp

        dphi = np.diff(self.surface.quadpoints_phi)[0]
        dtheta = np.diff(self.surface.quadpoints_theta)[0]
        dA = dphi * dtheta

        eye = np.eye(3) # (3, 3)

        # compute the magnetic field using the Biot-Savart law
        gradB = np.zeros((*np.shape(X), 3))
        for i in range(X.shape[0]):

            # gradient kernel
            diff = X[i] - quadpoints # (nphi, ntheta, 3)
            dist = np.sqrt(np.sum(diff**2, axis=-1, keepdims=True)) # (nphi, ntheta, 1)
            dist_cubed = dist**3 # (nphi, ntheta, 1)
            dist_fifth = dist**5 # (nphi, ntheta, 1)
            second_term = 3 * diff / dist_fifth # (nphi, ntheta, 1)

            for j in range(3):
                # TODO: speed up by skipping operations with 0s in eye[j]
                first_term = eye[j][None, None, :] / dist_cubed # (nphi, ntheta, 3)
                dkernel_by_dj = first_term - diff[:,:,j][:,:,None] * second_term
                cross = np.cross(K, dkernel_by_dj, axis=-1) # (nphi, ntheta, 3)
                gradB[i, :, j] = self.mu0_over_4pi * np.sum(cross * dA, axis=(0, 1))

        return gradB

    def B_normal(self, surf):
        """
        Compute the normal field error on a surface.

        Parameters:
            surf (Surface): A Simsopt surface object.
        
        Returns:
            np.ndarray: (nphi, ntheta) array of the normal field error at the surface quadrature points.
        """
        X = surf.gamma() # (nphi, ntheta, 3)
        B = self.B(X.reshape(-1, 3)).reshape(X.shape) # (nphi, ntheta, 3)
        n = surf.unitnormal()
        Bn = np.sum(B * n, axis=-1) # (nphi, ntheta)
        return Bn

    def squared_flux(self, surf):
        """
        Compute the total squared flux error on a surface,

            J_B = int (B * unit_normal)^2 dS.

        Parameters:
            surf (Surface): A Simsopt surface object.
        
        Returns:
            float: The squared flux error on the surface.
        """
        Bn = self.B_normal(surf)

        # TODO: use simsopt surf.darea here
        normal = surf.normal() # (nphi, ntheta, 3)
        dtheta = np.diff(surf.quadpoints_theta)[0]
        dphi = np.diff(surf.quadpoints_phi)[0]
        dA = dphi * dtheta * np.linalg.norm(normal, axis=-1)
        squaredflux = np.sum(Bn**2 * dA)
        return squaredflux

    def biot_savart_precomputation(self):
        """ Precompute the necessary quantities for the Biot-Savart law.
        This function computes the sheet current, the quadrature points, and the area element on the surface
        and stores them as attributes of the class.
        """
        # TODO: this function should be called after the recompute_bell

        # compute the sheet current
        K_1fp = self.current() # (nphi, ntheta, 3)

        # get the quadrature points
        quadpoints_1fp = self.surface.gamma() # (nphi, ntheta, 3)

        # normal
        normal_1fp = self.surface.normal() # (nphi, ntheta, 3)

        # compute grad(theta) and grad(phi)
        dr_by_dphi = self.surface.gammadash1() # (nphi, ntheta, 3)
        dr_by_dtheta = self.surface.gammadash2() # (nphi, ntheta, 3)
        n_cross_grad_phi_1fp = dr_by_dtheta
        n_cross_grad_theta_1fp = - dr_by_dphi

        nphi, ntheta, _ = quadpoints_1fp.shape

        # storage
        quadpoints = np.zeros((self.nfp * nphi, ntheta, 3))
        K = np.zeros((self.nfp * nphi, ntheta, 3))
        normal = np.zeros((self.nfp * nphi, ntheta, 3))
        n_cross_grad_phi = np.zeros((self.nfp * nphi, ntheta, 3))
        n_cross_grad_theta = np.zeros((self.nfp * nphi, ntheta, 3))

        # rotate to get full torus
        for ii in range(self.nfp):
            quadpoints_1fp = rotate_nfp(quadpoints_1fp, self.nfp)
            quadpoints[ii * nphi:(ii + 1) * nphi, :, :] = quadpoints_1fp
            K_1fp = rotate_nfp(K_1fp, self.nfp)
            K[ii * nphi:(ii + 1) * nphi, :, :] = K_1fp
            normal_1fp = rotate_nfp(normal_1fp, self.nfp)
            normal[ii * nphi:(ii + 1) * nphi, :, :] = normal_1fp
            n_cross_grad_phi_1fp = rotate_nfp(n_cross_grad_phi_1fp, self.nfp)
            n_cross_grad_phi[ii * nphi:(ii + 1) * nphi, :, :] = n_cross_grad_phi_1fp
            n_cross_grad_theta_1fp = rotate_nfp(n_cross_grad_theta_1fp, self.nfp)
            n_cross_grad_theta[ii * nphi:(ii + 1) * nphi, :, :] = n_cross_grad_theta_1fp

        self.K = K
        self.quadpoints = quadpoints
        self.normal = normal # (nphi, ntheta, 3)
        self.n_cross_grad_phi = n_cross_grad_phi
        self.n_cross_grad_theta = n_cross_grad_theta

        dphi = np.diff(self.surface.quadpoints_phi)[0]
        dtheta = np.diff(self.surface.quadpoints_theta)[0]
        self.dA = dphi * dtheta 


    def compute_h_fourier(self, x, nhat):
        """
        Compute the normal component of the projection of the field onto each fourier mode,

        h_mn^C(r) = - B_mn^C(r) * nhat(r) = - mu0_over_4pi * int sin(alpha_mn') * [(n' x grad(alpha_mn')) x kernel(r,r')] * nhat(r) dA'
        h_mn^S(r) = B_mn^S(r) * nhat(r) = mu0_over_4pi * int cos(alpha_mn') * [(n' x grad(alpha_mn')) x kernel(r,r')] * nhat(r) dA'

        Args:
            x (np.ndarray): (3,) array with a point in space where the field is evaluated.
            nhat (np.ndarray): (3,) array with the unit normal vector at the point x.

        Returns:
            np.ndarray: (ndofs,) array of the normal component of the projection of the field onto each fourier mode. The array
                is organized in the same way as the names array.
        """
        dA = self.dA # (nphi, ntheta, 1)
        n_cross_grad_theta = self.n_cross_grad_theta # (nphi, ntheta, 3)
        n_cross_grad_phi = self.n_cross_grad_phi # (nphi, ntheta, 3)

        # kernel x nhat
        diff = x - self.quadpoints # (nphi, ntheta, 3)
        dist = np.linalg.norm(diff, axis=-1, keepdims=True) # (nphi, ntheta, 1)
        kernel = diff / (dist**3) # (nphi, ntheta, 3)
        kernel_cross_nhat = np.cross(kernel, nhat, axis=-1) # (nphi, ntheta, 3)

        # TODO: double check we can just copy quadpoints_phi, nfp times
        phi1d = np.concatenate([self.surface.quadpoints_phi for ii in range(self.nfp)])
        theta1d = self.surface.quadpoints_theta
        phis, thetas = np.meshgrid(phi1d, theta1d, indexing='ij')

        # storage
        h_array = np.zeros(self.n_dofs) # (ndofs,)

        const = 1 / 4 / np.pi

        # TODO: we can do this with a fourier transform!
        idx = 0
        # compute h^S
        for m in range(self.M+1):
            for n in range(self.N+1):
                fourier = np.cos(2 * np.pi * (m * thetas - self.nfp * n * phis)) # (nphi, ntheta)
                n_cross_grad_alpha = 2 * np.pi * (m * n_cross_grad_theta  - self.nfp * n * n_cross_grad_phi) # (nphi, ntheta, 3)
                dot = np.sum(n_cross_grad_alpha * kernel_cross_nhat, axis=-1) # (nphi, ntheta)
                h_array[idx] = const * np.sum(fourier * dot * dA, axis=(-2, -1)) # (ndofs,)     
                idx += 1

        # compute h^C
        if not self.stellsym:
            for m in range(self.M+1):
                for n in range(self.N+1):
                    if m == 0 and n == 0:
                        continue
                    fourier = - np.sin(2 * np.pi * (m * thetas - self.nfp * n * phis)) # (nphi, ntheta)
                    n_cross_grad_alpha = 2 * np.pi * (m * n_cross_grad_theta  - self.nfp * n * n_cross_grad_phi) # (nphi, ntheta, 3)
                    dot = np.sum(n_cross_grad_alpha * kernel_cross_nhat, axis=-1) # (nphi, ntheta)
                    h_array[idx] = const * np.sum(fourier * dot * dA, axis=(-2, -1)) # (ndofs,)     
                    idx += 1

        return h_array
    

    def compute_h_secular(self, x, nhat):
        """
        Compute the normal field generated by the secular term

            h^P(r) = B^P(r) * nhat(r) = c * int [(n' x grad(phi')) x kernel(r,r')] * nhat(r) dA'
            c = mu0_over_4pi * G

        Args:
            x (np.ndarray): (3,) array with a point in space where the secular term is evaluated.
            nhat (np.ndarray): (3,) array with the unit normal vector at the point x.

        Returns:
            np.ndarray: The secular term for the Biot-Savart integral.
        """
        const = self.G / 4 / np.pi

        # kernel
        diff = x - self.quadpoints # (nphi, ntheta, 3)
        dist = np.linalg.norm(diff, axis=-1, keepdims=True) # (nphi, ntheta, 1)
        kernel = diff / (dist**3) # (nphi, ntheta, 3)

        # integrand
        cross = np.cross(self.n_cross_grad_phi, kernel, axis=-1) # (nphi, ntheta, 3)
        dot = np.sum(cross * nhat, axis=-1) # (nphi, ntheta)

        B = const * np.sum(dot * self.dA)
        return B

    
    def build_linear_system(self, surf):
        """ 
        Build the matrix H and vector y for the least squares problem. The rows of H
        capture the projection of the normal field onto the Fourier modes, and y captures the secular term.
        H is a (nphi * ntheta, ndofs) matrix, where nphi and ntheta are the number of quadrature points on
        surf.

        Args:
            surf (Surface): Simsopt Surface object.
        
        Returns:
            H (np.ndarray): (nphi * ntheta, ndofs) matrix representing the linear system.
            y (np.ndarray): (nphi * ntheta,) array representing the right-hand side of the linear system.
        """
    
        # points at which to evaluate loss
        X = surf.gamma().reshape(-1, 3) # (nphi * ntheta, 3)
        nhat = surf.unitnormal().reshape(-1, 3) # (nphi * ntheta, 3)

        # area element of S
        dphi = np.diff(surf.quadpoints_phi)[0]
        dtheta = np.diff(surf.quadpoints_theta)[0]
        normal = surf.normal() # (nphi, ntheta, 3)
        dS = dphi * dtheta * np.linalg.norm(normal, axis=-1) # (nphi, ntheta)
        sqrt_dS = np.sqrt(dS).reshape(-1) # (nphi * ntheta)

        # precompute stuff for the Biot-Savart law
        self.biot_savart_precomputation()

        # storage
        H = np.zeros((X.shape[0], self.n_dofs)) # (nphi * ntheta, n_dofs)
        y = np.zeros(X.shape[0]) # (nphi * ntheta,)

        # build the linear system
        for ii, x_target in enumerate(X):

            # biot-savart integral for each Fourier mode
            H[ii] = self.compute_h_fourier(x_target, nhat[ii]) * sqrt_dS[ii]
            
            # build rhs: biot-savart of secular term
            y[ii] = - self.compute_h_secular(x_target, nhat[ii]) * sqrt_dS[ii]
        
        return H, y


    def fit(self, surf):
        """"Fit" the sheet to a given surface by minimizing the squared flux error,
            min_w int |B * n|^2 dS,
        over the fourier coefficients, w, of the periodic potential function.
        The problem is equivalent to linear least squares problem,
            min_w |H @ w - y |^2 + lambda |w|^2.
        By writing the padded system,
            min_w |A @ w - b |^2, 
            A = [H; sqrt(lambda) * I]^T, b = [y; 0]^T,
        The normal equations can be solved using QR factorization of A.

        Args:
            surf (Surface): Simsopt Surface object.

        Returns:
            np.ndarray: The solution w, representing the Fourier coefficients of the sheet current potential.
        """
        H, y = self.build_linear_system(surf)
        
        # form padded system
        if self.jit > 0.0:
            pad = np.sqrt(self.jit) * np.eye(self.n_dofs) # (ndofs, ndofs)
            A = np.vstack((H, pad))
            b = np.hstack((y, np.zeros(self.n_dofs)))
        else:
            A = H
            b = y

        # solve normal equations
        Q, R = np.linalg.qr(A, mode='reduced')
        # TODO: can also solve w =  np.linalg.solve(R, Q.T @ b) when R.T is invertible
        w =  np.linalg.solve(R.T @ R, R.T @ Q.T @ b)

        self.local_full_x = w

        return w

