import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))
import FSIsolver.fsi_solver.solver as solver


# create mesh: first create mesh by running ./create_mesh/create_mesh_FSI.py

# load mesh
mesh = df.Mesh()
with df.XDMFFile(str(here.parent) + "/Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mesh)
mvc = df.MeshValueCollection("size_t", mesh, 2)
mvc2 = df.MeshValueCollection("size_t", mesh, 2)
with df.XDMFFile(str(here.parent) + "/Output/Mesh_Generation/facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
with df.XDMFFile(str(here.parent) + "/Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mvc2, "name_to_read")
boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = df.cpp.mesh.MeshFunctionSizet(mesh,mvc2)
bdfile = df.File(str(here.parent) + "/Output/Mesh_Generation/boundary.pvd")
bdfile << boundaries
bdfile = df.File(str(here.parent) + "/Output/Mesh_Generation/domains.pvd")
bdfile << domains

# boundary parts
params = np.load(str(here.parent) + '/Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "obstacle_fluid", "obstacle_solid"]

# subdomains
fluid_domain = df.MeshView.create(domains, params["fluid"])
solid_domain = df.MeshView.create(domains, params["solid"])
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
# FSI_param['deltat'] = 0.0025 # 0.01
FSI_param['deltat'] = 0.01
# FSI_param['T'] = 17.0
FSI_param['T'] = 15.0

FSI_param['displacement_point'] = df.Point((0.6, 0.2))

# boundary conditions, need to be 0 at t = 0
Ubar = 1.0
FSI_param['boundary_cond'] = df.Expression(("(t < 2)?(1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))):"
                                         "(1.5*Ubar*4.0*x[1]*(0.41 -x[1]))/ 0.1681", "0.0"),
                                        Ubar=Ubar, t=FSI_param['t'], degree=2)


from torch_extension.extension import TorchExtension, TorchExtensionRecord
import torch
import torch.nn as nn
from torch_extension.loading import load_model
net = load_model("torch_extension/models/yankee")
net.eval()

net(torch.rand((3935, 8))) # Check everything works before run.

# extension_operator = TorchExtension(fluid_domain, net, T_switch=0.0)
extension_operator = TorchExtensionRecord(fluid_domain, net, T_switch=0.0, T_record=18.0, run_name="Data0")

# save options
FSI_param['save_directory'] = str(str(here.parent) + '/TorchOutput/dataanalysis/learnext_dataset/yankee') #no save if set to None
FSI_param["save_data_on"] = True
# FSI_param['save_directory'] = None
df.set_log_active(False)
fsisolver = solver.FSIsolver(mesh, boundaries, domains, params, FSI_param, extension_operator, warmstart=False) #warmstart needs to be set to False for the first run
fsisolver.solve()

print("Solver complete")
