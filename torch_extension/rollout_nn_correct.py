import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

import tqdm
from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))


# create mesh: first create mesh by running ./create_mesh/create_mesh_FSI.py
mesh_dir = Path("/mnt/LearnExt/Output/Mesh_Generation")

# load mesh
mesh = df.Mesh()
with df.XDMFFile(str(mesh_dir / "mesh_triangles.xdmf")) as infile:
    infile.read(mesh)
mvc = df.MeshValueCollection("size_t", mesh, 2)
mvc2 = df.MeshValueCollection("size_t", mesh, 2)
with df.XDMFFile(str(mesh_dir / "facet_mesh.xdmf")) as infile:
    infile.read(mvc, "name_to_read")
with df.XDMFFile(str(mesh_dir / "mesh_triangles.xdmf")) as infile:
    infile.read(mvc2, "name_to_read")
boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = df.cpp.mesh.MeshFunctionSizet(mesh,mvc2)

# boundary parts
params = np.load(mesh_dir / 'params.npy', allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "obstacle_fluid", "obstacle_solid"]

# subdomains
fluid_domain = df.MeshView.create(domains, params["fluid"])
solid_domain = df.MeshView.create(domains, params["solid"])

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
FSI_param['T'] = 1.0

FSI_param['displacement_point'] = df.Point((0.6, 0.2))

# boundary conditions, need to be 0 at t = 0
Ubar = 1.0
FSI_param['boundary_cond'] = df.Expression(("(t < 2)?(1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))):"
                                         "(1.5*Ubar*4.0*x[1]*(0.41 -x[1]))/ 0.1681", "0.0"),
                                        Ubar=Ubar, t=FSI_param['t'], degree=2)

Fspace = df.VectorFunctionSpace(fluid_domain, "CG", 2)
fspace = df.VectorFunctionSpace(fluid_domain, "CG", 1)


import torch
# from torch_extension.loading import load_model
# network = load_model("torch_extension/models/foxtrot")
# network.eval()

# print(network)

# network(torch.rand((fspace.dim()//2, 8))) # Check everything works before run.

from FSIsolver.extension_operator.extension import TorchExtension

# extension_operator = TorchExtension(fluid_domain, network, T_switch=0.0, silent=True)
# print(extension_operator.model)
extension_operator = TorchExtension(fluid_domain, "torch_extension/models/foxtrot", T_switch=0.0, silent=True)
# print(extension_operator.model)


input_xdmf_file = df.XDMFFile("Output/Extension/Data/input_.xdmf")
if Path("foxtrotRolloutP1.xdmf").exists():
    Path("foxtrotRolloutP1.xdmf").unlink()
    Path("foxtrotRolloutP1.h5").unlink
# if Path("foxtrotRolloutP2.xdmf").exists():
#     Path("foxtrotRolloutP2.xdmf").unlink()
#     Path("foxtrotRolloutP2.h5").unlink
output_xdmf_file_p1 = df.XDMFFile("foxtrotRolloutP1.xdmf")
output_xdmf_file_p1.write(fluid_domain)
# output_xdmf_file_p2 = df.XDMFFile("foxtrotRolloutP2.xdmf")
# output_xdmf_file_p2.write(fluid_domain)


u_bc = df.Function(Fspace)
u_p1 = df.Function(fspace)
for k in tqdm.tqdm(range(206)): # Number of time steps that make it loop best.
    input_xdmf_file.read_checkpoint(u_bc, "input_harmonic_ext", k)
    u_ext = extension_operator.extend(u_bc, {"t": 1.0})
    u_p1.interpolate(u_ext)
    # output_xdmf_file_p2.write_checkpoint(u_ext, "uh", k, append=True)
    output_xdmf_file_p1.write_checkpoint(u_p1, "uh", k, append=True)


