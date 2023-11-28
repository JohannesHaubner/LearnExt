# from dolfin import *
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))
import FSIsolver.extension_operator.extension as extension
import FSIsolver.fsi_solver.solver as solver



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

import torch
import torch.nn as nn
from torch_extension.networks import MLP_BN

widths = [8] + [64]*10 + [2]
mlp = MLP_BN(widths, activation=nn.ReLU())
mlp.load_state_dict(torch.load("torch_extension/model/sierra/mlp_state_dict.pt"))
mlp.double()
mlp.eval()



from torch_extension.extension import TorchExtension

extender = TorchExtension(fluid_domain, mlp)



refined_mesh = df.refine(fluid_domain)

CG2 = df.VectorFunctionSpace(fluid_domain, "CG", 2)
CG1 = df.VectorFunctionSpace(fluid_domain, "CG", 1)
CG1_ref = df.VectorFunctionSpace(refined_mesh, "CG", 1)
DG1 = df.VectorFunctionSpace(fluid_domain, "DG", 1)
DG1_ref = df.VectorFunctionSpace(refined_mesh, "DG", 1)

u = df.Function(CG2)
u_cg1 = df.Function(CG1)
u_ref = df.Function(CG1_ref)
# u_corr = df.Function(CG2)
# u_corr_ref = df.Function(CG1_ref)

from torch_extension.extension import clement_interpolate, CG1_vector_plus_grad_to_array_w_coords, poisson_mask_custom
poisson_mask_f = "2.0 * (x[0]+1.0) * (1-x[0]) * exp( -3.5*pow(x[0], 7) ) + 0.1"

mask     = poisson_mask_custom(df.FunctionSpace(fluid_domain, "CG", 1), poisson_mask_f, normalize=True)
mask_ref = poisson_mask_custom(df.FunctionSpace(refined_mesh, "CG", 1), poisson_mask_f, normalize=True)

def extend(model: nn.Module, u_harm: df.Function, mask: df.Function) -> df.Function:
    Q = df.VectorFunctionSpace(u_harm.function_space().mesh(), "DG", 1)
    mask_np = mask.vector().get_local().reshape(-1,1)

    qh_harm = df.interpolate(u_harm, Q)
    gh_harm = clement_interpolate(df.grad(qh_harm))

    harmonic_plus_grad_w_coords_np = CG1_vector_plus_grad_to_array_w_coords(u_harm, gh_harm)
    harmonic_plus_grad_w_coords_torch = torch.tensor(harmonic_plus_grad_w_coords_np, dtype=torch.float64).reshape(1,-1,8)

    with torch.no_grad():
        corr_np = model(harmonic_plus_grad_w_coords_torch).detach().numpy().reshape(-1,2)
        corr_np = corr_np * mask_np

    u_corr = df.Function(u_harm.function_space())
    # new_dofs = np.zeros_like(u_cg1.vector().get_local())
    new_dofs = u_harm.vector().get_local()
    new_dofs[0::2] += corr_np[:,0]
    new_dofs[1::2] += corr_np[:,1]
    u_corr.vector().set_local(new_dofs)

    # u_ = df.interpolate(u_cg1, self.F)

    return u_corr

input_file_name = "TorchOutput/Extension/Data/input_.xdmf"
input_file = df.XDMFFile(input_file_name)

output_file_name = "TorchOutput/same_mesh.xdmf"
output_file = df.XDMFFile(output_file_name)
output_file_ref_name = "TorchOutput/refined_mesh.xdmf"
output_file_ref = df.XDMFFile(output_file_ref_name)

from tqdm import tqdm
checkpoints = tqdm(range(3100, 3152+1))

for k in checkpoints:
    input_file.read_checkpoint(u, "input_harmonic_ext", k)
    u_cg1.interpolate(u)
    u_ref.interpolate(u_cg1)

    u_corr = extend(mlp, u_cg1, mask)
    u_corr_ref = extend(mlp, u_ref, mask_ref)

    output_file.write_checkpoint(u_corr, "corrected", k, df.XDMFFile.Encoding.HDF5, append=True)
    output_file_ref.write_checkpoint(u_corr_ref, "refined_corrected", k, df.XDMFFile.Encoding.HDF5, append=True)


input_file.close()
output_file.close()



