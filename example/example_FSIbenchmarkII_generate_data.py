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
from coeff_machine_learning import NN_der

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
FSI_param['deltat'] = 0.0025 # 0.01
FSI_param['T'] = 17.0

FSI_param['displacement_point'] = Point((0.6, 0.2))

# boundary conditions, need to be 0 at t = 0
Ubar = 1.0
FSI_param['boundary_cond'] = Expression(("(t < 2)?(1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))):"
                                         "(1.5*Ubar*4.0*x[1]*(0.41 -x[1]))/ 0.1681", "0.0"),
                                        Ubar=Ubar, t=FSI_param['t'], degree=2)

# extension operator
class Biharmonic_DataGeneration(extension.ExtensionOperator):
    def __init__(self, mesh):
        super().__init__(mesh)

        T = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, MixedElement(T, T))
        self.iter = -1

        # Create time series
        self.xdmf_input = XDMFFile("../Output/Extension/Data/input_.xdmf")
        self.xdmf_output = XDMFFile("../Output/Extension/Data/output_.xdmf")

    def extend(self, boundary_conditions, params):
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        t = params["t"]

        uz = TrialFunction(self.FS)
        puz = TestFunction(self.FS)
        (u, z) = split(uz)
        (psiu, psiz) = split(puz)

        dx = Measure('dx', domain=self.mesh)

        a = inner(grad(z), grad(psiu)) * dx + inner(z, psiz) * dx - inner(grad(u), grad(psiz)) * dx
        L = Constant(0.0) * psiu[0] * dx

        bc = DirichletBC(self.FS.sub(0), boundary_conditions, 'on_boundary')

        uz = Function(self.FS)

        solve(a == L, uz, bc)

        u_, z_ = uz.split(deepcopy=True)

        save_ext = False
        if save_ext:
            file = File('../../Output/Extension/function.pvd')
            file << u_

        if t > 11:
            self.iter +=1
            self.xdmf_input.write_checkpoint(boundary_conditions, "input", self.iter, XDMFFile.Encoding.HDF5, append=True)
            self.xdmf_output.write_checkpoint(u_, "output", self.iter, XDMFFile.Encoding.HDF5, append=True)

        return u_

# hacky solution:
meshfile = XDMFFile("../Output/Mesh_Generation/mesh.xdmf")
mesh = Mesh()
with meshfile as meshfile:
    meshfile.read(mesh)

extension_operator = Biharmonic_DataGeneration(mesh)

breakpoint()

# save options
FSI_param['save_directory'] = str('./../Output/FSIbenchmarkII_generate_data') #no save if set to None
#FSI_param['save_every_N_snapshot'] = 4 # save every 8th snapshot

# initialize FSI solver
fsisolver = solver.FSIsolver(mesh, boundaries, domains, params, FSI_param, extension_operator, warmstart=True) #warmstart needs to be set to False for the first run
fsisolver.solve()

