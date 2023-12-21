import torch
import torch.nn as nn
import dolfin as df
import numpy as np

from pathlib import Path
here = Path(__file__).parent.resolve()

# import FSIsolver.extension_operator.extension as extension
# from torch_extension.clement import clement_interpolate

def CG1_vector_plus_grad_to_array(u: df.Function, du: df.Function) -> np.ndarray:
    """ 
        Layout: Columns ``(u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)``

        `u` is a CG1 function. `du` is a CG1 function over same 
        mesh as `u`, and a clement interpolant as given by 
            `clement_interpolate(df.grad(qh))`,
        where `qh` is taken as
            `df.interpolate(u, df.VectorFunctionSpace(u.function_space().mesh(), "DG", 1, 2))`.
    """

    new_array = np.zeros((u.function_space().mesh().num_vertices(), 6))
    
    raw_array_base = u.vector().get_local()
    new_array[:,0] = raw_array_base[::2]
    new_array[:,1] = raw_array_base[1::2]

    raw_array_grad = du.vector().get_local()
    new_array[:,2+0] = raw_array_grad[0::4]
    new_array[:,2+1] = raw_array_grad[1::4]
    new_array[:,2+2] = raw_array_grad[2::4]
    new_array[:,2+3] = raw_array_grad[3::4]

    return new_array

def CG1_vector_plus_grad_to_array_w_coords(u: df.Function, du: df.Function) -> np.ndarray:
    """ 
        Layout: Columns ``(x, y, u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)``

        `u` is a CG1 function. `du` is a CG1 function over same 
        mesh as `u`, and a clement interpolant as given by 
            `clement_interpolate(df.grad(qh))`,
        where `qh` is taken as
            `df.interpolate(u, df.VectorFunctionSpace(u.function_space().mesh(), "DG", 1, 2))`.
    """

    new_array = np.zeros((u.function_space().mesh().num_vertices(), 8))
    
    coords = u.function_space().tabulate_dof_coordinates()[::2]
    new_array[:,0+0] = coords[:,0]
    new_array[:,0+1] = coords[:,1]
    
    raw_array_base = u.vector().get_local()
    new_array[:,2+0] = raw_array_base[::2]
    new_array[:,2+1] = raw_array_base[1::2]

    raw_array_grad = du.vector().get_local()
    new_array[:,4+0] = raw_array_grad[0::4]
    new_array[:,4+1] = raw_array_grad[1::4]
    new_array[:,4+2] = raw_array_grad[2::4]
    new_array[:,4+3] = raw_array_grad[3::4]

    return new_array

def poisson_mask_custom(V: df.FunctionSpace, f_str: str, normalize: bool = False) -> df.Function:
    """
        -Delta u = f in Omega
               u = 0 on dOmega
    """

    def boundary(x, on_boundary):
        return on_boundary
    u0 = df.Constant(0.0)
    
    bc = df.DirichletBC(V, u0, boundary)

    f = df.Expression(f_str, degree=5)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    a = df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx
    l = f * v * df.dx

    uh = df.Function(V)
    df.solve(a == l, uh, bc)
    
    if normalize:
        uh.vector()[:] /= np.max(uh.vector()[:]) # Normalize mask to have sup-norm 1.

    return uh
