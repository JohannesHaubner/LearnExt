from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../FSIsolver/extension_operator')
import extension
sys.path.insert(1, '../FSIsolver/fsi_solver')
import solver
sys.path.insert(1, '../learnExt')
from NeuralNet.neural_network_custom import ANN, generate_weights
from learnext import LearnExt

# create mesh: first create mesh by running ./create_mesh/create_mesh_FSI.py

# load mesh
mesh = Mesh()
with XDMFFile("./../Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
mvc2 = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("./../Output/Mesh_Generation/facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
with XDMFFile("./../Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mvc2, "name_to_read")
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = cpp.mesh.MeshFunctionSizet(mesh,mvc2)
bdfile = File("./../Output/Mesh_Generation/boundary.pvd")
bdfile << boundaries
bdfile = File("./../Output/Mesh_Generation/domains.pvd")
bdfile << domains

# boundary parts
params = np.load('../Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "obstacle_fluid", "obstacle_solid"]

# subdomains
fluid_domain = MeshView.create(domains, params["fluid"])
solid_domain = MeshView.create(domains, params["solid"])
#plot(solid_domain)
#plt.show()

# parameters for FSI system
FSI_param = {}

FSI_param['fluid_mesh'] = fluid_domain
FSI_param['solid_mesh'] = solid_domain

FSI_param['lambdas'] = 2.0e6
FSI_param['mys'] = 0.5e6
FSI_param['rhos'] = 1.0e4
FSI_param['rhof'] = 1.0e3
FSI_param['nyf'] = 1.0e-3

FSI_param['t'] = 0.0
FSI_param['deltat'] = 0.01
FSI_param['T'] = 15.0

FSI_param['displacement_point'] = Point((0.6, 0.2))

# boundary conditions, need to be 0 at t = 0
Ubar = 1.0
FSI_param['boundary_cond'] = Expression(("(t < 2)?(1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))):"
                                         "(1.5*Ubar*4.0*x[1]*(0.41 -x[1]))/ 0.1681", "0.0"),
                                        Ubar=Ubar, t=FSI_param['t'], degree=2)

threshold = 0.001

# extension operator
class LearnExtension(extension.ExtensionOperator):
    def __init__(self, mesh):
        super().__init__(mesh)

        T = VectorElement("CG", self.mesh.ufl_cell(), 1)
        T2 = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, T)
        self.FS2 = FunctionSpace(self.mesh, T2)
        self.incremental = True
        self.incremental_correct = True
        self.bc_old = Function(self.FS)
        output_directory = str("../example/learned_networks/")
        self.net = ANN(output_directory + "trained_network_0903_1.pkl")

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
            elif displacementy <= 0.01:
                trafo = False
            else:
                trafo = True
        else:
            trafo = False


        save_ext = True
        if save_ext:
            file = File('./../Output/Extension/function.pvd')
            file << boundary_conditions

        if b_old != None:
            self.bc_old = project(b_old, self.FS2)

        if trafo:
            up = project(self.bc_old, self.FS2)
            upi = project(-1.0*up, self.FS2)
            ALE.move(self.mesh, up, annotate=False)

        u = Function(self.FS2)
        v = TestFunction(self.FS2)

        dx = Measure('dx', domain=self.mesh)

        E = inner(LearnExt.NN_der(threshold, inner(grad(self.bc_old), grad(self.bc_old)), self.net) * grad(u), grad(v)) * dx

        # solve PDE
        if trafo:
            bc_func = project(boundary_conditions - self.bc_old, self.FS2)
        else:
            bc_func = boundary_conditions
        bc = DirichletBC(self.FS2, bc_func, 'on_boundary')


        solve(E == 0, u, bc)

        if trafo:
            u = project(u + self.bc_old, self.FS2)
        self.bc_old = project(u, self.FS2)

        if save_ext:
            file << u
        if trafo:
            ALE.move(self.mesh, upi, annotate=False)

        return u

extension_operator = LearnExtension(fluid_domain)

# save options
FSI_param['save_directory'] = str('./../Output/FSIbenchmarkII_0903'
                                  '') #no save if set to None
#FSI_param['save_every_N_snapshot'] = 4 # save every 8th snapshot

# initialize FSI solver
fsisolver = solver.FSIsolver(mesh, boundaries, domains, params, FSI_param, extension_operator, warmstart=False)
fsisolver.solve()

