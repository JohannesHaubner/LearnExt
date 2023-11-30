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


class TorchExtension(extension.ExtensionOperator):

    def __init__(self, mesh, model: nn.Module, T_switch: float = 0.0, mask_rhs: str | None = None, silent: bool = False):
        super().__init__(mesh)

        T = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.F = df.FunctionSpace(self.mesh, T)
        self.iter = -1

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

        # u_cg1 is referred to in the clement_interpolate internals, so it has to 
        # be carried through in the simulation and the dofs updated with the 
        # harmonic extension at each time step.
        self.u_cg1 = df.Function(self.F_cg1)
        _, self.clement_interpolater = clement_interpolate(df.grad(self.u_cg1), with_CI=True)

        # Pytorch model
        self.model = model
        model.eval()

        # mask for adjusting pytorch correction
        V_scal = df.FunctionSpace(self.mesh, "CG", 1)
        if mask_rhs is None:
            # Masking function custom made for specific domain.
            mask_rhs = "2.0 * (x[0]+1.0) * (1-x[0]) * exp( -3.5*pow(x[0], 7) ) + 0.1"
        poisson_mask = poisson_mask_custom(V_scal, mask_rhs, normalize=True)
        self.mask_np = poisson_mask.vector().get_local().reshape(-1,1)
        # Mask needs to have shape (num_vertices, 1) to broadcast correctly in
        # multiplication with correction of shape (num_vertices, 2).

        # # Time to switch from harmonic to torch-corrected extension
        self.T_switch = T_switch

        self.silent = silent

        return

    def extend(self, boundary_conditions, params):
        """ Torch-corrected extension of boundary_conditions (Function on self.mesh) to the interior """

        t = params["t"]

        # harmonic extension
        bc = df.DirichletBC(self.F, boundary_conditions, 'on_boundary')
        bc.apply(self.rhs_harmonic)
        uh = df.Function(self.F)
        self.solver_harmonic.solve(uh.vector(), self.rhs_harmonic)

        if t < self.T_switch:
            u_ = uh
        
        else:
            if not self.silent:
                print("Torch-corrected extension")

            harmonic_cg1 = df.interpolate(uh, self.F_cg1)
            self.u_cg1.vector().set_local(harmonic_cg1.vector().get_local())

            gh_harm = self.clement_interpolater()

            harmonic_plus_grad_w_coords_np = CG1_vector_plus_grad_to_array_w_coords(harmonic_cg1, gh_harm)
            harmonic_plus_grad_w_coords_torch = torch.tensor(harmonic_plus_grad_w_coords_np, dtype=torch.float32)

            with torch.no_grad():
                corr_np = self.model(harmonic_plus_grad_w_coords_torch).detach().double().numpy()
                corr_np = corr_np * self.mask_np

            new_dofs = harmonic_cg1.vector().get_local()
            new_dofs[0::2] += corr_np[:,0]
            new_dofs[1::2] += corr_np[:,1]
            harmonic_cg1.vector().set_local(new_dofs)

            u_ = df.interpolate(harmonic_cg1, self.F)
            # Apply the harmonic extension boundary condition to ensure fluid-solid extension matches at all
            # dof locations, not just vertices.
            bc.apply(u_.vector())

        # Store extensions for optional post-step processing by subclass.
        self.uh = uh
        self.u_ = u_

class TorchExtensionInplace(extension.ExtensionOperator):

    def __init__(self, mesh, model: nn.Module, T_switch: float = 0.0, mask_rhs: str | None = None, silent: bool = False):
        super().__init__(mesh)

        T = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.F = df.FunctionSpace(self.mesh, T)
        self.iter = -1

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

        # u_cg1 is referred to in the clement_interpolate internals, so it has to 
        # be carried through in the simulation and the dofs updated with the 
        # harmonic extension at each time step.
        self.u_cg1 = df.Function(self.F_cg1)
        _, self.clement_interpolater = clement_interpolate(df.grad(self.u_cg1), with_CI=True)

        # Pytorch model
        self.model = model
        model.eval()

        # mask for adjusting pytorch correction
        V_scal = df.FunctionSpace(self.mesh, "CG", 1)
        if mask_rhs is None:
            # Masking function custom made for specific domain.
            mask_rhs = "2.0 * (x[0]+1.0) * (1-x[0]) * exp( -3.5*pow(x[0], 7) ) + 0.1"
        poisson_mask = poisson_mask_custom(V_scal, mask_rhs, normalize=True)
        self.mask_np = poisson_mask.vector().get_local().reshape(-1,1)
        # Mask needs to have shape (num_vertices, 1) to broadcast correctly in
        # multiplication with correction of shape (num_vertices, 2).

        # # Time to switch from harmonic to torch-corrected extension
        self.T_switch = T_switch

        self.silent = silent

        self.harm_cg1 = df.Function(self.F_cg1)
        self.uh = df.Function(self.F)
        self.u_ = df.Function(self.F)

        return

    def extend(self, boundary_conditions, params):
        """ Torch-corrected extension of boundary_conditions (Function on self.mesh) to the interior """

        t = params["t"]

        # harmonic extension
        bc = df.DirichletBC(self.F, boundary_conditions, 'on_boundary')
        bc.apply(self.rhs_harmonic)
        self.solver_harmonic.solve(self.uh.vector(), self.rhs_harmonic)

        if t < self.T_switch:
            self.u_ = self.uh
        
        else:
            if not self.silent:
                print("Torch-corrected extension")

            self.harm_cg1.interpolate(self.uh)
            self.u_cg1.vector().set_local(self.harm_cg1.vector().get_local())

            gh_harm = self.clement_interpolater()

            harmonic_plus_grad_w_coords_np = CG1_vector_plus_grad_to_array_w_coords(self.harm_cg1, gh_harm)
            harmonic_plus_grad_w_coords_torch = torch.tensor(harmonic_plus_grad_w_coords_np, dtype=torch.float32)

            with torch.no_grad():
                corr_np = self.model(harmonic_plus_grad_w_coords_torch).detach().double().numpy()
                corr_np = corr_np * self.mask_np

            new_dofs = self.harm_cg1.vector().get_local()
            new_dofs[0::2] += corr_np[:,0]
            new_dofs[1::2] += corr_np[:,1]
            self.harm_cg1.vector().set_local(new_dofs)

            self.u_.interpolate(self.harm_cg1)
            # Apply the harmonic extension boundary condition to ensure fluid-solid extension matches at all
            # dof locations, not just vertices.
            bc.apply(self.u_.vector())

        return self.u_

class TorchExtensionInplaceMat(extension.ExtensionOperator):

    def __init__(self, mesh, model: nn.Module, T_switch: float = 0.0, mask_rhs: str | None = None, silent: bool = False):
        super().__init__(mesh)

        T = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.F = df.FunctionSpace(self.mesh, T)
        self.iter = -1

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

        self.harm_cg1 = df.Function(self.F_cg1)
        self.uh = df.Function(self.F)
        self.u_ = df.Function(self.F)

        self.interp_mat_2_1 = df.PETScDMCollection.create_transfer_matrix(self.F, self.F_cg1)
        self.interp_mat_1_2 = df.PETScDMCollection.create_transfer_matrix(self.F_cg1, self.F)

        # self.harm_cg1 is referred to in the clement_interpolate internals, so it has to 
        # be carried through in the simulation and the dofs updated with the 
        # harmonic extension at each time step.
        _, self.clement_interpolater = clement_interpolate(df.grad(self.harm_cg1), with_CI=True)

        # Pytorch model
        self.model = model
        model.eval()

        # mask for adjusting pytorch correction
        V_scal = df.FunctionSpace(self.mesh, "CG", 1)
        if mask_rhs is None:
            # Masking function custom made for specific domain.
            mask_rhs = "2.0 * (x[0]+1.0) * (1-x[0]) * exp( -3.5*pow(x[0], 7) ) + 0.1"
        poisson_mask = poisson_mask_custom(V_scal, mask_rhs, normalize=True)
        self.mask_np = poisson_mask.vector().get_local().reshape(-1,1)
        # Mask needs to have shape (num_vertices, 1) to broadcast correctly in
        # multiplication with correction of shape (num_vertices, 2).

        # # Time to switch from harmonic to torch-corrected extension
        self.T_switch = T_switch

        self.silent = silent

        return

    def extend(self, boundary_conditions, params):
        """ Torch-corrected extension of boundary_conditions (Function on self.mesh) to the interior """

        t = params["t"]

        # harmonic extension
        bc = df.DirichletBC(self.F, boundary_conditions, 'on_boundary')
        bc.apply(self.rhs_harmonic)
        self.solver_harmonic.solve(self.uh.vector(), self.rhs_harmonic)

        if t < self.T_switch:
            self.u_ = self.uh
        
        else:
            if not self.silent:
                print("Torch-corrected extension")

            self.interp_mat_2_1.mult(self.uh.vector(), self.harm_cg1.vector())

            gh_harm = self.clement_interpolater()

            harmonic_plus_grad_w_coords_np = CG1_vector_plus_grad_to_array_w_coords(self.harm_cg1, gh_harm)
            harmonic_plus_grad_w_coords_torch = torch.tensor(harmonic_plus_grad_w_coords_np, dtype=torch.float32)

            with torch.no_grad():
                corr_np = self.model(harmonic_plus_grad_w_coords_torch).numpy()
                corr_np = corr_np * self.mask_np

            new_dofs = self.harm_cg1.vector().get_local()
            new_dofs[0::2] += corr_np[:,0]
            new_dofs[1::2] += corr_np[:,1]
            self.harm_cg1.vector().set_local(new_dofs)

            self.interp_mat_1_2.mult(self.harm_cg1.vector(), self.u_.vector())

            # Apply the harmonic extension boundary condition to ensure fluid-solid extension matches at all
            # dof locations, not just vertices.
            bc.apply(self.u_.vector())

        return self.u_

class TorchExtensionRecord(TorchExtension):
    def __init__(self, mesh, model, T_switch=0.0, mask_rhs = None, T_record=0.0, run_name="Data0"):
        super().__init__(mesh, model, T_switch=T_switch, mask_rhs=mask_rhs)

        # Time to start recording
        self.T_record = T_record

        # Create time series
        self.xdmf_input = df.XDMFFile(str(here.parent) + f"/TorchOutput/Extension/{run_name}/harm.xdmf")
        self.xdmf_output = df.XDMFFile(str(here.parent) + f"/TorchOutput/Extension/{run_name}/torch.xdmf")

        return

    def extend(self, boundary_conditions, params):
        u_ = super().extend(boundary_conditions, params)

        if params["t"] > self.T_record:
            self.iter +=1
            self.xdmf_input.write_checkpoint(self.uh, "input_harmonic_ext", self.iter, df.XDMFFile.Encoding.HDF5, append=True)
            self.xdmf_output.write_checkpoint(self.u_, "output_pytorch_ext", self.iter, df.XDMFFile.Encoding.HDF5, append=True)

        return u_
