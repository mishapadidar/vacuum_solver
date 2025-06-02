from sheet_current import SheetCurrent
from simsopt._core import load
import numpy as np

""" 
Starlite has 2 designs, A and B, each with 3 different iota optimizations.
In this example, we fit a sheet current to the Boozer surfaces of each design and iota optimization.
We display the error in the fit.
"""

designs = ["A", "B"]

for des in designs:

    # load the boozer surfaces (1 per Current configuration, so 3 total.)
    data = load(f"./configurations/design{des}_after_scaled.json")
    bsurfs = data[0] # BoozerSurfaces
    iota_Gs = data[1] # (iota, G) pairs
    magnetic_axis_curves = data[2] # magnetic axis CurveRZFouriers

    print("")
    print("-"*50)
    print("Design", des)
    # 3 different iota optimizations
    for iota_group_idx in range(3):
        print("")
        print("iota group:", iota_group_idx)
        # print('iota = %.4f, G = %.4f'%(iota_Gs[iota_group_idx][0], iota_Gs[iota_group_idx][1]))
        biotsavart = bsurfs[iota_group_idx].biotsavart
        surf = bsurfs[iota_group_idx].surface
        surf = surf.to_RZFourier().copy(nphi = 31, ntheta=31, range='half period')

        # winding surface cannot exploit half-period symmetry
        surf_winding = surf.copy(nphi = 51, ntheta=51, range='field period')
        minor_radius = surf.minor_radius()
        surf_winding.extend_via_normal(-minor_radius * 2)

        coils = biotsavart.coils
        currents = [np.abs(coil.current.get_value()) for coil in coils]

        # use theoretical value of G (not the fitted value)
        I_P = np.abs(sum([np.abs(coil.current.get_value()) for coil in coils]))
        mu0 =  4 * np.pi * 10**(-7)
        G = I_P * mu0
        print(f"G = {G :.2f}")

        # create the sheet current object
        solver = SheetCurrent(surf_winding, G, M=10, N=10, jit=1e-14)
        solver.fit(surf)
        print(f'Squared Flux: {solver.squared_flux(surf) :.2e}')

        # compute the actual field
        points = surf.gamma().reshape(-1, 3)
        points = np.ascontiguousarray(points, dtype=np.float64)
        B = solver.B(points)
        biotsavart.set_points(points)
        B_bs = biotsavart.B()

        # compute the error between the fields
        err = np.linalg.norm((B - B_bs), axis=-1) 
        rel_err = err / np.linalg.norm(B_bs, axis=-1)
        print(f"Max Rel. Error in B: {100 * np.max(rel_err) :.2f}%", )
