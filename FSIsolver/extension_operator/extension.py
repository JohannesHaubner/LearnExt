from dolfin import *
import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path
here = Path(__file__).parent.parent.parent.resolve()
import sys, os
sys.path.insert(0, str(here))
from learnExt.NeuralNet.neural_network_custom import ANN, generate_weights
from learnExt.learnext_hybridPDENN import Custom_Reduced_Functional as crf

class ExtensionOperator(object):
    def __init__(self, mesh, marker, ids):
        self.mesh = mesh
        self.marker = marker
        self.ids = ids

    def extend(self, boundary_conditions, params=None):
        """extend the boundary_conditions to the interior of the mesh"""
        raise NotImplementedError

    def custom(self, FSI):
        """custom function for extension operator"""
        return False
    
    @staticmethod
    def timings_extension(func):
        """decorator to get timings for extension"""
        def wrapper(self, *arg, **kw):
            dump_timings_to_xml('test_timings.xml', TimingClear.clear)

            with Timer("do extension"):
                res = func(self, *arg, **kw)

            dump_timings_to_xml('test_timings.xml', TimingClear.keep)

            tree = ET.parse('test_timings.xml')
            root = tree.getroot()

            mpi_kind = 'MPI_MAX'

            data = {}
            for table in root:
                if mpi_kind not in table.get('name'):
                    continue
                
                for row in table:
                    row_data = {}
                    for col in row:
                        row_data[col.get('key')] = col.get('value')
                    data[row.get('key')] = row_data

            col = 'wall tot'
            for process in data:
                print(process, data[process]['reps'], data[process][col])

            return res
        return wrapper

class Biharmonic(ExtensionOperator):
    def __init__(self, mesh, marker=None, ids=None, save_extension=False, save_filename=None):
        super().__init__(mesh, marker, ids)

        # options
        self.save_ext = save_extension

        if self.save_ext:
            self.iter = -1
            if save_filename == None:
                raise Exception('save_filename (str) not specified')
            self.xdmf_output = XDMFFile(str(save_filename))
            self.xdmf_output.write(self.mesh)

        T = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, MixedElement(T, T))

        uz = TrialFunction(self.FS)
        puz = TestFunction(self.FS)
        (u, z) = split(uz)
        (psiu, psiz) = split(puz)

        dx = Measure('dx', domain=self.mesh)

        a = inner(grad(z), grad(psiu)) * dx + inner(z, psiz) * dx - inner(grad(u), grad(psiz)) * dx
        L = Constant(0.0) * psiu[0] * dx


        self.A = assemble(a)

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS.sub(0), Constant((0.,0.)), 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS.sub(0), Constant((0., 0.)), self.marker, i))
        self.bc = bc

        for bci in self.bc:
            bci.apply(self.A)

        self.solver = LUSolver(self.A)

        self.L = assemble(L)


    @ExtensionOperator.timings_extension
    def extend(self, boundary_conditions, params=None):
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS.sub(0), boundary_conditions, 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS.sub(0), boundary_conditions, self.marker, i))

        for bci in bc:
            bci.apply(self.L)

        uz = Function(self.FS)

        self.solver.solve(uz.vector(), self.L)

        u_, z_ = uz.split(deepcopy=True)

        if self.save_ext:
            self.iter +=1
            self.xdmf_output.write_checkpoint(u_, "output_biharmonic_ext", self.iter, XDMFFile.Encoding.HDF5, append=True)

        return u_
    
class Harmonic(ExtensionOperator):
    def __init__(self, mesh, marker=None, ids=None, save_extension=False, save_filename=None, incremental=False):
        super().__init__(mesh, marker, ids)

        # options
        self.save_ext = save_extension

        if self.save_ext:
            self.iter = -1
            if save_filename == None:
                raise Exception('save_filename (str) not specified')
            self.xdmf_output = XDMFFile(str(save_filename))
            self.xdmf_output.write(self.mesh)

        T = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, T)

        self.incremental = incremental
        if self.incremental:
            self.bc_old = Function(self.FS)

        u = TrialFunction(self.FS)
        v = TestFunction(self.FS)

        dx = Measure('dx', domain=self.mesh)

        a = inner(grad(u), grad(v)) * dx
        L = Constant(0.0) * v[0] * dx


        self.A = assemble(a)

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS, Constant((0.,0.)), 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS, Constant((0., 0.)), self.marker, i))
        self.bc = bc

        for bci in self.bc:
            bci.apply(self.A)

        self.solver = LUSolver(self.A)

        self.L = assemble(L)


    @ExtensionOperator.timings_extension
    def extend(self, boundary_conditions, params=None):
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS, boundary_conditions, 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS, boundary_conditions, self.marker, i))

        for bci in bc:
            bci.apply(self.L)

        if self.incremental:
            # move mesh with previous deformation
            self.bc_old = boundary_conditions
            up = project(self.bc_old, self.FS)
            upi = Function(self.FS)
            upi.vector().axpy(-1., up.vector())
            ALE.move(self.mesh, up)

        u_ = Function(self.FS)

        self.solver.solve(u_.vector(), self.L)

        if self.incremental:
            # move mesh back
            ALE.move(self.mesh, upi)

        if self.save_ext:
            self.iter +=1
            self.xdmf_output.write_checkpoint(u_, "output_biharmonic_ext", self.iter, XDMFFile.Encoding.HDF5, append=True)

        return u_
    
   
class LearnExtension(ExtensionOperator):
    def __init__(self, mesh, NN_path, threshold=None, marker=None, ids=None, save_extension=False, save_filename=None, incremental=False, incremental_corrected=False):
        super().__init__(mesh, marker, ids)

        T = VectorElement("CG", self.mesh.ufl_cell(), 1)
        T2 = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, T)
        self.FS2 = FunctionSpace(self.mesh, T2)
        self.bc_old = Function(self.FS2)
        self.net = ANN(NN_path)

        self.save_ext = save_extension
        self.incremental = incremental
        self.incremental_corrected = incremental_corrected

        if threshold == None:
            print('Threshold set to 0')
            self.threshold = 0.0
        else:
            self.threshold = threshold

        if self.save_ext:
            self.iter = -1
            if save_filename == None:
                raise Exception('save_filename (str) not specified')
            self.xdmf_output = XDMFFile(str(save_filename))
            self.xdmf_output.write(self.mesh)

    @ExtensionOperator.timings_extension
    def extend(self, boundary_conditions, params = None):
        """ harmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        if params != None:
            try:
                b_old = params["b_old"]
            except:
                pass
            try:
                displacementy = params["displacementy"]
            except:
                displacementy = None

        if self.incremental == True and self.incremental_correct == False:
            trafo = True
        elif self.incremental == True and self.incremental_correct == True:
            if displacementy == None:
                Warning("displacementy == None; set trafo to False")
                trafo = False
            elif abs(displacementy) <= 0.005:
                print('displacementy <= 0.005: displacementy = ', displacementy)
                trafo = False
            else:
                trafo = True
        else:
            trafo = False

        if b_old != None:
            self.bc_old = project(b_old, self.FS)

        if trafo:
            up = project(self.bc_old, self.FS)
            upi = Function(self.FS)
            upi.vector().axpy(-1.0, up.vector())
            try:
                ALE.move(self.mesh, up, annotate=False)
            except:
                ALE.move(self.mesh, up)

        if trafo:
            u = TrialFunction(self.FS2)
        else:
            u = Function(self.FS2)
        v = TestFunction(self.FS2)

        dx = Measure('dx', domain=self.mesh, metadata={'quadrature_degree': 4})

        if trafo:
            E = inner(crf.NN_der(self.threshold, inner(grad(self.bc_old), grad(self.bc_old)), self.net) * grad(u), grad(v)) * dx
        else:
            E = inner(crf.NN_der(self.threshold, inner(grad(u), grad(u)), self.net) * grad(u), grad(v)) * dx

        # solve PDE
        if trafo:
            bc_func = project(boundary_conditions - self.bc_old, self.FS2)
        else:
            bc_func = boundary_conditions
        bc = DirichletBC(self.FS2, bc_func, 'on_boundary')

        if trafo:
            u = Function(self.FS2)
            solve(lhs(E) == rhs(E), u, bc)
        else:
            solve(E == 0, u, bc, solver_parameters={"nonlinear_solver": "newton", "newton_solver":
                {"maximum_iterations": 200}})

        if trafo:
            u = project(u + self.bc_old, self.FS2)
        self.bc_old.assign(project(u, self.FS))

        if self.save_ext:
            self.iter +=1
            self.xdmf_output.write_checkpoint(u_, "output_biharmonic_ext", self.iter, XDMFFile.Encoding.HDF5, append=True)

        if trafo:
            try:
                ALE.move(self.mesh, up, annotate=False)
            except:
                ALE.move(self.mesh, up)

        return u
    
class LearnExtensionSimplified(ExtensionOperator):
    def __init__(self, mesh, network_path: str):
        super().__init__(mesh, marker=None, ids=None)

        T = df.VectorElement("CG", self.mesh.ufl_cell(), 1)
        T2 = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = df.FunctionSpace(self.mesh, T)
        self.FS2 = df.FunctionSpace(self.mesh, T2)
        self.bc_old = df.Function(self.FS)
        self.net = ANN(network_path)
        self.threshold = 0.001

    def extend(self, boundary_conditions, params = None):
        """ harmonic extension of boundary_conditions (Function on self.mesh) to the interior """


        u = df.Function(self.FS2)
        v = df.TestFunction(self.FS2)

        dx = df.Measure('dx', domain=self.mesh, metadata={'quadrature_degree': 4})

        E = df.inner(crf.NN_der(self.threshold, df.inner(df.grad(u), df.grad(u)), self.net) * df.grad(u), df.grad(v)) * dx


        # solve PDE
        bc = df.DirichletBC(self.FS2, boundary_conditions, 'on_boundary')


        df.solve(E == 0, u, bc, solver_parameters={"nonlinear_solver": "newton", "newton_solver":
            {"maximum_iterations": 200}})


        self.bc_old.assign(df.project(u, self.FS))

        return u
    

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc    
class LearnExtensionSimplifiedSNES(ExtensionOperator):

    class SNESProblem():
        def __init__(self, F, u, bc):
            V = u.function_space()
            du = TrialFunction(V)
            self.L = F
            self.a = derivative(F, u, du)
            self.bcs = bc
            self.u = u
            return

        def F(self, snes, x, F):
            x = PETScVector(x)
            F = PETScVector(F)
            assemble(self.L, tensor=F)
            for bc in self.bcs:
                bc.apply(F, x)    
            return            

        def J(self, snes, x, J, P):
            J = PETScMatrix(J)
            assemble(self.a, tensor=J)
            for bc in self.bcs:
                bc.apply(J)
            return

    def __init__(self, mesh, network_path: str):
        super().__init__(mesh, marker=None, ids=None)

        T = df.VectorElement("CG", self.mesh.ufl_cell(), 1)
        T2 = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = df.FunctionSpace(self.mesh, T)
        self.FS2 = df.FunctionSpace(self.mesh, T2)
        self.net = ANN(network_path)
        self.threshold = 0.001

        self.snes_ls = PETSc.SNES().create(MPI.comm_world) 
        opts_ls = PETSc.Options()
        opts_ls.setValue('snes_monitor', None)
        opts_ls.setValue('snes_type', 'newtonls')
        opts_ls.setValue('snes_linesearch_type', 'l2')
        opts_ls.setValue('snes_divergence_tolerance', 1e12)
        self.snes_ls.setFromOptions()
        self.snes_ls.setErrorIfNotConverged(True)


        return

    def extend(self, boundary_conditions, params = None):
        """ harmonic extension of boundary_conditions (Function on self.mesh) to the interior """


        u = df.Function(self.FS2)
        v = df.TestFunction(self.FS2)

        dx = df.Measure('dx', domain=self.mesh, metadata={'quadrature_degree': 4})

        E = df.inner(crf.NN_der(self.threshold, df.inner(df.grad(u), df.grad(u)), self.net) * df.grad(u), df.grad(v)) * dx


        # solve PDE
        bc = df.DirichletBC(self.FS2, boundary_conditions, 'on_boundary')

        problem = self.SNESProblem(E, u, [bc])
            
        b = PETScVector()  # same as b = PETSc.Vec()
        J_mat = PETScMatrix()   

        ksp = self.snes_ls.getKSP()
        ksp.getPC().setType('lu')
        ksp.getPC().setFactorSolverType('mumps')
        ksp.setType('preonly')

        self.snes_ls.setFunction(problem.F, b.vec())
        self.snes_ls.setJacobian(problem.J, J_mat.mat())
        self.snes_ls.solve(None, problem.u.vector().vec())

        return u


import torch
import torch.nn as nn
import dolfin as df
from torch_extension.clement import clement_interpolate
from torch_extension.tools import poisson_mask_custom, CG1_vector_plus_grad_to_array_w_coords
class TorchExtension(ExtensionOperator):

    def __init__(self, mesh, model: nn.Module | str, T_switch: float = 0.0, mask_rhs: str | None = None, silent: bool = False):
        super().__init__(mesh, marker=None, ids=None)
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

        # PETSc-matrices for efficient interpolation of functions.
        self.interp_mat_2_1 = df.PETScDMCollection.create_transfer_matrix(self.F, self.F_cg1)
        self.interp_mat_1_2 = df.PETScDMCollection.create_transfer_matrix(self.F_cg1, self.F)

        _, self.clement_interpolater = clement_interpolate(df.grad(self.harm_cg1), with_CI=True)

        # Pytorch model
        if isinstance(model, str):
            from torch_extension.loading import load_model
            model = load_model(model)
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

    def extend(self, boundary_conditions, params={"t": np.inf}):
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
            # boundary dof locations, not just vertices.
            bc.apply(self.u_.vector())

        return self.u_
    
class TorchExtensionRecord(TorchExtension):
    def __init__(self, mesh, model, T_switch=0.0, mask_rhs = None, T_record=0.0, run_name="Data0", silent: bool = False):
        super().__init__(mesh, model, T_switch=T_switch, mask_rhs=mask_rhs, silent=silent)

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



if __name__ == "__main__":
    # load mesh
    mesh = Mesh()
    with XDMFFile("../../Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)
    mvc2 = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile("../../Output/Mesh_Generation/facet_mesh.xdmf") as infile:
        infile.read(mvc, "name_to_read")
    with XDMFFile("../../Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
        infile.read(mvc2, "name_to_read")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    domains = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
    bdfile = File("../../Output/Mesh_Generation/boundary.pvd")
    bdfile << boundaries
    bdfile = File("../../Output/Mesh_Generation/domains.pvd")
    bdfile << domains

    # subdomains
    fluid_domain = MeshView.create(domains, 7)
    mvcf = MeshValueCollection("size_t", fluid_domain, 1)
    boundariesf = cpp.mesh.MeshFunctionSizet(fluid_domain, mvcf)
    bdfile = File("../../Output/Mesh_Generation/boundaryf.pvd")
    bdfile << boundariesf

    class Boundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    bound = Boundary()

    bound.mark(boundariesf, 0)
    bdfile << boundariesf

    # boundary conditions on whole outer part of fluid domain
    Biharmonic(fluid_domain).extend(Expression(("x[0]", "x[1]"), degree=1))
