import dolfin as df
import numpy as np
import os

class ExtensionOperator(object):
    def __init__(self, mesh):
        self.mesh = mesh

    def extend(self, boundary_conditions, params=None):
        """extend the boundary_conditions to the interior of the mesh"""
        raise NotImplementedError

    def custom(self, FSI):
        """custom function for extension operator"""
        return False
    

class BiharmonicExtension(ExtensionOperator):
    def __init__(self, mesh):
        super().__init__(mesh)

        T = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = df.FunctionSpace(self.mesh, df.MixedElement(T, T))

    def extend(self, boundary_conditions, params=None):
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        uz = df.TrialFunction(self.FS)
        puz = df.TestFunction(self.FS)
        (u, z) = df.split(uz)
        (psiu, psiz) = df.split(puz)

        dx = df.Measure('dx', domain=self.mesh)

        a = df.inner(df.grad(z), df.grad(psiu)) * dx + df.inner(z, psiz) * dx - df.inner(df.grad(u), df.grad(psiz)) * dx
        L = df.Constant(0.0) * psiu[0] * dx

        bc = df.DirichletBC(self.FS.sub(0), boundary_conditions, 'on_boundary')

        uz = df.Function(self.FS)

        df.solve(a == L, uz, bc)

        u_, z_ = uz.split(deepcopy=True)

        return u_

class HarmonicExtension(ExtensionOperator):

    def __init__(self, mesh, order: int = 2):
        super().__init__(mesh)

        self.order = order
        self.V = df.VectorFunctionSpace(self.mesh, "CG", self.order)

        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        f = df.Constant((0.0, 0.0))
        self.a = df.inner(df.grad(u), df.grad(v)) * df.dx
        self.L = df.inner(f, v) * df.dx

        return

    def extend(self, boundary_conditions, params=None):

        u = df.Function(self.V)
        bc = df.DirichletBC(self.V, boundary_conditions, "on_boundary")

        df.solve(self.a == self.L, u, [bc])

        return u

class NNCorrectionExtension(ExtensionOperator):

    def __init__(self, mesh, model_dir: os.PathLike):
        super().__init__(mesh)

        from torch_extension.loading import load_model
        net = load_model(model_dir)
        net.eval()

        from FSIsolver.extension_operator.extension import TorchExtension
        self.extension_operator = TorchExtension(mesh, net, T_switch=0.0, silent=True)

        return
    
    def extend(self, boundary_conditions, params={"t": 1.0}):
        
        return self.extension_operator.extend(boundary_conditions, params=params)

from learnExt.NeuralNet.neural_network_custom import ANN
from learnExt.learnext_hybridPDENN import Custom_Reduced_Functional as crf
class LearnExtension(ExtensionOperator):
    def __init__(self, mesh, network_path: str):
        super().__init__(mesh)

        T = df.VectorElement("CG", self.mesh.ufl_cell(), 1)
        T2 = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = df.FunctionSpace(self.mesh, T)
        self.FS2 = df.FunctionSpace(self.mesh, T2)
        self.bc_old = df.Function(self.FS)
        # network_path = "example/learned_networks/trained_network.pkl"
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
    