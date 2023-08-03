import torch
import torch.nn as nn
import dolfin as df
import numpy as np

from pathlib import Path
here = Path(__file__).parent.resolve()

import FSIsolver.extension_operator.extension as extension
from torch_extension.clement import clement_interpolate

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
        Layout: Columns ``(u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)``

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

class TorchExtension(extension.ExtensionOperator):


    def __init__(self, mesh, model: nn.Module):
        super().__init__(mesh)

        T = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = df.FunctionSpace(self.mesh, df.MixedElement(T, T))
        self.F = df.FunctionSpace(self.mesh, T)
        self.iter = -1

        # Create time series
        self.xdmf_input = df.XDMFFile(str(here.parent) + "/TorchOutput/Extension/Data/input_.xdmf")
        self.xdmf_output = df.XDMFFile(str(here.parent) + "/TorchOutput/Extension/Data/output_.xdmf")

        # harmonic extension 
        uh = df.TrialFunction(self.F)
        v = df.TestFunction(self.F)

        a = df.inner(df.grad(uh), df.grad(v))*df.dx
        L = df.Constant(0.0) * v[0] *df.dx
        A = df.assemble(a)
        
        bc = df.DirichletBC(self.F, df.Constant((0.,0.)), 'on_boundary')
        bc.apply(A)

        self.solver_harmonic = df.LUSolver(A)
        self.rhs_harmonic = df.assemble(L)

        # For clement interpolation
        T_cg1 = df.VectorElement("CG", self.mesh.ufl_cell(), 1)
        self.F_cg1 = df.FunctionSpace(self.mesh, T_cg1)
        self.Q = df.VectorFunctionSpace(self.mesh, "DG", 1, 2)

        # Pytorch model
        self.model = model
        model.double()
        model.eval()

        # mask for adjusting pytorch correction
        V_scal = df.FunctionSpace(self.mesh, "CG", 1)
        poisson_mask_f = "2.0 * (x[0]+1.0) * (1-x[0]) * exp( -3.5*pow(x[0], 7) ) + 0.1"
        poisson_mask = poisson_mask_custom(V_scal, poisson_mask_f, normalize=True)
        self.mask_np = poisson_mask.vector().get_local().reshape(-1,1)

        # Time to switch from harmonic to pytorch extension
        self.T_switch = 0.01
        # Time to record
        self.T_record = 0.0


        return

    def extend(self, boundary_conditions, params):
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        t = params["t"]
        dx = df.Measure('dx', domain=self.mesh)

        # harmonic extension
        bc = df.DirichletBC(self.F, boundary_conditions, 'on_boundary')
        bc.apply(self.rhs_harmonic)
        uh = df.Function(self.F)
        self.solver_harmonic.solve(uh.vector(), self.rhs_harmonic)

        if t < self.T_switch:
            u_ = uh
        
        else:
            print("Torch-corrected extension")
            harmonic_cg1 = df.interpolate(uh, self.F_cg1)

            qh_harm = df.interpolate(harmonic_cg1, self.Q)
            gh_harm = clement_interpolate(df.grad(qh_harm))

            harmonic_plus_grad_w_coords_np = CG1_vector_plus_grad_to_array_w_coords(harmonic_cg1, gh_harm)
            harmonic_plus_grad_w_coords_torch = torch.tensor(harmonic_plus_grad_w_coords_np, dtype=torch.float64).reshape(1,-1,8)

            with torch.no_grad():
                corr_np = self.model(harmonic_plus_grad_w_coords_torch).detach().numpy().reshape(-1,2)
                corr_np = corr_np * self.mask_np

            corr_cg1 = df.Function(self.F_cg1)
            new_dofs = np.zeros_like(corr_cg1.vector().get_local())
            new_dofs[0::2] = corr_np[:,0]
            new_dofs[1::2] = corr_np[:,1]
            corr_cg1.vector().set_local(new_dofs)

            u_ = df.interpolate(corr_cg1, self.F)
            new_dofs = np.copy(u_.vector().get_local())
            new_dofs += uh.vector().get_local()
            u_.vector().set_local(new_dofs)

        if t > self.T_record:
            self.iter +=1
            self.xdmf_input.write_checkpoint(uh, "input_harmonic_ext", self.iter, df.XDMFFile.Encoding.HDF5, append=True)
            self.xdmf_output.write_checkpoint(u_, "output_pytorch_ext", self.iter, df.XDMFFile.Encoding.HDF5, append=True)

        return u_