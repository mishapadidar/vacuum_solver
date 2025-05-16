import numpy as np
from simsopt._core import Optimizable


class SheetCurrent(Optimizable):
    def __init__(self, surface, I_P, M=8, N=8):
        """
        Initialize the sheet current with the given parameters.

        The sheet current is defined as:
            K = n x grad(Phi)
        where 
            Phi = tilde{Phi} + (I_P / 2 * pi) * phi
        and tilde{Phi} is doubly periodic on (theta, phi) in [0, 1] x [0, 1/nfp].
        
        Parameters:
        surface (Surface): A Simsopt surface object.
        I_P (float): Poloidal current
        phi (float): The potential function.
        M,N (int): Highest fourier mode number (inclusive) in the poloidal and toroidal directions.
        """
        self.surface = surface
        self.nfp = surface.nfp
        self.stellsym = surface.stellsym
        self.I_P = I_P
        self.M = M # poloidal
        self.N = N # toroidal

        self._set_names()
        
        Optimizable.__init__(self, x0=np.zeros(self.n_dofs),
                             names=self.names, depends_on=[surface])

    def _set_names(self):
        """
        Set the list of names for each degree of freedom.

        The degrees of freedom are the coefficients of the Fourier series
        for the layer potential,
            x = [c(m,n), ... , s(m,n)]
        We always omit the s(0,0) term and in stellarator symmetry we omit all s(m,n) terms. 
        """
        names = []
        for m in range(self.M+1):
            for n in range(self.N+1):
                names += ['c({},{})'.format(m, n)]
        if not self.stellsym:
            for m in range(self.M):
                for n in range(self.N):
                    if m==0 and n==0:
                        continue
                    names += ['s({},{})'.format(m, n)]
        self.names = names
        self.n_dofs = len(self.names)

    def recompute_bell(self, parent=None):
        """
        This function will get called any time any of the DOFs of the
        parent class change.
        """
        self.need_to_run_code = True
        return super().recompute_bell(parent)
    
    # def build_fourier_array(self):
    #     """
    #     Build an array of the Fourier harmonics evaluated at the surface quadrature points.

    #     """
    #     # Get the surface quadrature points
    #     theta1d = self.surface.quadpoints_theta
    #     phi1d = self.surface.quadpoints_phi
    #     ntheta = len(theta1d)
    #     nphi = len(phi1d)

    #     thetas, phis = np.meshgrid(theta1d, phi1d, indexing='ij')

    #     # Calculate the potential function
    #     pot = np.zeros_like((ntheta, nphi))
    #     for m in range(self.M+1):
    #         for n in range(self.N+1):
    #             phi += np.cos(2 * np.pi * (m * thetas - self.nfp * n * phis))
    #             if not self.stellsym:
    #                 phi += np.sin(2 * np.pi * (m * theta - self.nfp * n * phis))
        
    
    def potential(self):
        """ Compute the potential at the surface quadrature points
            Phi = tilde{Phi} + (I_P / 2 * pi) * phi
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
        # cosine modes
        for m in range(self.M+1):
            for n in range(self.N+1):
                pot += dofs[idx] * np.cos(2 * np.pi * (m * thetas - self.nfp * n * phis))
                idx += 1
        # sine modes
        if not self.stellsym:
            for m in range(self.M):
                for n in range(self.N):
                    if m==0 and n==0:
                        continue
                    pot += dofs[idx] * np.sin(2 * np.pi * (m * thetas - self.nfp * n * phis))
                    idx += 1
        
        # secular term
        pot += (self.I_P / 2 / np.pi) * phis
        return pot
    
    def current(self):
        """ Compute the sheet current at the quadrature points on S'.

        Simsopt surfaces obey the ordering (phi, theta), so,
            n = dr/dphi x dr/dtheta.

        We use the fact that
            n x grad(theta) = -dr/dphi
            n x grad(phi) = dr/dtheta

        The potential is given by,
            Phi = tilde{Phi} + (I_P / 2 * pi) * phi
        and
            tilde{Phi} = sum_{m,n} c(m,n) * cos(alpha_mn) + s(m,n) * sin(alpha_mn),
        where
            alpha_mn = 2 * pi (m * theta - nfp * n * phi ).
        The sheet current is,
            K = n x [grad(tilde{Phi}) + (I_P / 2 * pi) * grad(phi)]
              = (I_P / 2 * pi) * dr/dtheta + 
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
        # cosine modes
        for m in range(self.M+1):
            for n in range(self.N+1):
                alpha = 2 * np.pi * (m * thetas - self.nfp * n * phis) # (nphi, ntheta)
                n_cross_grad_alpha = 2 * np.pi * (m * n_cross_grad_theta  - self.nfp * n * n_cross_grad_phi) # (nphi, ntheta, 3)
                K += - dofs[idx] * np.sin(alpha)[:,:,None] * n_cross_grad_alpha # (nphi, ntheta, 3)
                idx += 1
        # sine modes
        if not self.stellsym:
            for m in range(self.M):
                for n in range(self.N):
                    if m==0 and n==0:
                        continue
                    alpha = 2 * np.pi * (m * thetas - self.nfp * n * phis) # (nphi, ntheta)
                    n_cross_grad_alpha = 2 * np.pi * (m * n_cross_grad_theta  - self.nfp * n * n_cross_grad_phi) # (nphi, ntheta, 3)
                    K += dofs[idx] * np.cos(alpha)[:,:,None] * n_cross_grad_alpha # (nphi, ntheta, 3)
                    idx += 1
            
        # secular term
        K += (self.I_P / 2 / np.pi) * n_cross_grad_phi
        return K
    
    def B(self, X):
        """Compute the magnetic field at a set of points X
        using the Biot-Savart law.

        X should not be placed on the flux surface, as the Biot-Savart law will be singular.

        Parameters:
            X (np.ndarray): (n, 3) array of points where the magnetic field is computed.

        Returns:
            np.ndarray: (n, 3) array of the magnetic field at the points X.
        """

        # compute the sheet current
        K = self.current() # (nphi, ntheta, 3)

        # get the quadrature points
        quadpoints = self.surface.gamma() # (nphi, ntheta, 3)
        dphi = np.diff(self.surface.quadpoints_phi)[0]
        dtheta = np.diff(self.surface.quadpoints_theta)[0]
        normal = self.surface.normal() # (nphi, ntheta, 3)
        dA = dphi * dtheta * np.linalg.norm(normal, axis=-1, keepdims=True)

        mu0 =  1.256637061e-6 # N / A^2
        mu0_over_4pi = mu0 / (4 * np.pi) 

        # compute the magnetic field using the Biot-Savart law
        B = np.zeros(np.shape(X))
        for i in range(X.shape[0]):
            diff = X[i] - quadpoints # (nphi, ntheta, 3)
            dist = np.linalg.norm(diff, axis=-1, keepdims=True) # (nphi, ntheta, 1)
            kernel = diff / (dist**3) # (nphi, ntheta, 3)
            cross = np.cross(K, kernel, axis=-1) # (nphi, ntheta, 3)
            B[i] = mu0_over_4pi * np.sum(cross * dA, axis=(0, 1))
        
        return B
    
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
        normal = self.surface.normal() # (nphi, ntheta, 3)
        dtheta = np.diff(self.surface.quadpoints_theta)[0]
        dphi = np.diff(self.surface.quadpoints_phi)[0]
        dA = dphi * dtheta * np.linalg.norm(normal, axis=-1, keepdims=True)
        squaredflux = np.sum(Bn**2 * dA)
        return squaredflux

