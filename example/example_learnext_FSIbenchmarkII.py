from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from pyadjoint.overloaded_type import create_overloaded_object

import sys
sys.path.insert(1, '../FSIsolver/extension_operator')
import extension
sys.path.insert(1, '../FSIsolver/fsi_solver')
import solver
sys.path.insert(1, '../learnExt')
from NeuralNet.neural_network_custom import ANN, generate_weights
from learn import threshold
from coeff_machine_learning import NN_der
import coeff_opt_control_new as opt_cont
import coeff_machine_learning_new as opt_ml
import os

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
fluid_domain = create_overloaded_object(fluid_domain)
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

# boundaries for fluid domain
boundary_marker = 1
interface_marker = 2
params_ext = {}
params_ext["no_slip"] = boundary_marker
params_ext["interface"] = interface_marker
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and between(x[0], (0.2, 0.6)) and between(x[1], (0.19, 0.21))
boundary = Boundary()
interface = Interface()
boundaries_ext = cpp.mesh.MeshFunctionSizet(fluid_domain, 1)
boundary.mark(boundaries_ext, boundary_marker)
interface.mark(boundaries_ext, interface_marker)

#file = File("../Output/learnExt/results/boundaries.pvd")
#file << boundaries
#breakpoint()

# save mesh
def_boundary_parts = ["interface"]
zero_boundary_parts = ["no_slip"]

# extension operator
class LearnExtension(extension.ExtensionOperator):
    def __init__(self, mesh, def_boundary_parts, zero_boundary_parts, params, boundaries, output_path):
        super().__init__(mesh)

        self.flag = False
        self.def_boundary_parts = def_boundary_parts
        self.zero_boundary_parts = zero_boundary_parts
        self.params = params
        self.boundaries = boundaries
        self.V = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Vs = FunctionSpace(self.mesh, "CG", 1)

        T = VectorElement("CG", self.mesh.ufl_cell(), 1)
        T2 = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, T)
        self.FS2 = FunctionSpace(self.mesh, T2)
        self.output_directory = output_path + "/learn"
        import os
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
        self.counter = 0
        self.learned = False
        self.net = None
        self.NN_x = np.linspace(0, 15, 100)
        np.savetxt(self.output_directory + "/NN_x.txt", self.NN_x)
        self.NN_y = []

        self.t_FSI = 0

        self.NN_times = []

    def update_flag(self, flag):
        self.flag = flag

    def custom(self, FSI):
        """ set options such that extension operator is learned"""
        if FSI.dt < FSI.dt_min and FSI.t != self.t_FSI:
            FSI.t = FSI.t - FSI.dt
            self.t_FSI = FSI.t
            FSI.dt = FSI.dt_max # reset time_step
            self.flag = True
            self.NN_times.append(FSI.t)
            np.savetxt(self.output_directory + "/NN_times.txt", self.NN_times)
            return True


    def learn_NN(self, boundary_conditions):
        self.counter += 1
        threshold = 0.001
        # save boundary conditions
        bc_filename = self.output_directory + "/" + str(self.counter) + "_boundary_conditions.xdmf"
        self.xdmf_bc = XDMFFile(bc_filename)
        self.xdmf_bc.write_checkpoint(boundary_conditions, "bc", 0, XDMFFile.Encoding.HDF5, append=True)
        # learn NN
        opt_cont.compute_optimal_coefficient_new(self.mesh, self.V, self.Vs, self.params, boundary_conditions,
                                                 self.def_boundary_parts, self.zero_boundary_parts, self.boundaries,
                                                 self.output_directory, net=self.net, threshold=threshold)
        output_path = self.output_directory + "/neural_network_" + str(self.counter) + ".pkl"
        net_old = self.net
        self.net = opt_ml.compute_machine_learning_new(self.mesh, self.Vs, self.output_directory,
                                                       output_path, threshold, net=self.net)

        opt_ml.visualize(self.mesh, self.V, self.Vs, self.params, boundary_conditions, self.def_boundary_parts,
                         self.zero_boundary_parts, self.boundaries, self.output_directory,
                         threshold, net=self.net, counter=self.counter, net_old=net_old)

        self.NN_y.append([project(self.net(Constant(i)), self.Vs).vector().get_local()[0] for i in self.NN_x])

        np.savetxt(self.output_directory + "/NN_values.txt", self.NN_y)
        pass

    def extend(self, boundary_conditions):
        """extension of boundary_conditions (Function on self.mesh) to the interior """

        if self.flag == True:
            self.learned = True
            self.flag = False
            self.learn_NN(boundary_conditions)

        save_ext = True
        if save_ext:
            file = File('../Output/Extension/function.pvd')
            file << boundary_conditions

        u = Function(self.FS2)
        v = TestFunction(self.FS2)

        dx = Measure('dx', domain=self.mesh)

        if self.learned == False:
            E = inner(grad(u), grad(v)) * dx(self.mesh)
        else:
            E = inner(NN_der(threshold, inner(grad(u), grad(u)), self.net) * grad(u), grad(v)) * dx(self.mesh)

        # solve PDE
        bc = DirichletBC(self.FS2, boundary_conditions, 'on_boundary')


        solve(E == 0, u, bc)

        if save_ext:
            file << u

        return u

warmstart = False

output_path = str('./../Output/FSIbenchmarkII_learn_adaptive')
if not os.path.exists(output_path):
    os.mkdir(output_path)

extension_operator = LearnExtension(fluid_domain, def_boundary_parts, zero_boundary_parts, params_ext, boundaries_ext,
                                    output_path)

# save options
FSI_param['save_directory'] = output_path

# initialize FSI solver
fsisolver = solver.FSIsolver(mesh, boundaries, domains, params, FSI_param, extension_operator, warmstart=warmstart)
fsisolver.solve()

