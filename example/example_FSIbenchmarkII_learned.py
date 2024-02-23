from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))
import FSIsolver.extension_operator.extension as extension
import FSIsolver.fsi_solver.solver as solver
from learnExt.NeuralNet.neural_network_custom import ANN, generate_weights
from learnExt.learnext_hybridPDENN import Custom_Reduced_Functional as crf

# create mesh: first create mesh by running ./create_mesh/create_mesh_FSI.py

# load mesh
mesh = Mesh()
with XDMFFile(str(here.parent) + "/Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
mvc2 = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(str(here.parent) + "/Output/Mesh_Generation/facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
with XDMFFile(str(here.parent) + "/Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mvc2, "name_to_read")
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = cpp.mesh.MeshFunctionSizet(mesh,mvc2)
bdfile = File(str(here.parent) + "/Output/Mesh_Generation/boundary.pvd")
bdfile << boundaries
bdfile = File(str(here.parent) + "/Output/Mesh_Generation/domains.pvd")
bdfile << domains

# boundary parts
params = np.load(str(here.parent) + '/Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

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
FSI_param["save_data_on"] = True


# boundary conditions, need to be 0 at t = 0
Ubar = 1.0
FSI_param['boundary_cond'] = Expression(("(t < 2)?(1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))):"
                                         "(1.5*Ubar*4.0*x[1]*(0.41 -x[1]))/ 0.1681", "0.0"),
                                        Ubar=Ubar, t=FSI_param['t'], degree=2)

threshold = 0.001

test_case = int(sys.argv[1])

if test_case == 1:
    extension_operator = extension.LearnExtension(fluid_domain, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold)# learned
elif test_case == 2:
    extension_operator = extension.LearnExtension(fluid_domain, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold)# learned artificial dataset
elif test_case == 3:
    extension_operator = extension.LearnExtension(fluid_domain, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=False)# learned linearized
elif test_case == 4:
    extension_operator = extension.LearnExtension(fluid_domain, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=False)# learned linearized artificial dataset
elif test_case == 5:
    extension_operator = extension.LearnExtension(fluid_domain, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=True)# learned linearized corrected
elif test_case == 6:
    extension_operator = extension.LearnExtension(fluid_domain, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=True)# learned linearized corrected artificial dataset
elif test_case == 7:
    extension_operator = extension.TorchExtension(fluid_domain, "torch_extension/models/yankee", T_switch=0.0, silent=True) # NN-corrected
elif test_case == 8:
    extension_operator = extension.TorchExtension(fluid_domain, "torch_extension/models/foxtrot", T_switch=0.0, silent=True) # NN-corrected artificial dataset
else:
    raise ValueError

out_path = f'/Output/FSIbenchmarkII_supervised_300322_{test_case}'

# save options
FSI_param['save_directory'] = str(here.parent) + out_path #no save if set to None
#FSI_param['save_every_N_snapshot'] = 4 # save every 8th snapshot

# initialize FSI solver
fsisolver = solver.FSIsolver(mesh, boundaries, domains, params, FSI_param, extension_operator, warmstart=False)
fsisolver.solve()

