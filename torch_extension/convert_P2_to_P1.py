import numpy as np
import dolfin as df
from pathlib import Path

here = Path(__file__).parent.resolve()

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


V_cg2 = df.VectorFunctionSpace(fluid_domain, "CG", 2)
V_cg1 = df.VectorFunctionSpace(fluid_domain, "CG", 1)


input_file_name = "TorchOutput/Extension/Data/output_.xdmf"
output_file_name = "TorchOutput/Extension/Data/output_cg1.xdmf"

input_file = df.XDMFFile(input_file_name)
output_file = df.XDMFFile(output_file_name)

from tqdm import tqdm

checkpoints = tqdm(range(0, 3152+1))

u_cg2 = df.Function(V_cg2)
u_cg1 = df.Function(V_cg1)


for k in checkpoints:
    input_file.read_checkpoint(u_cg2, "output_pytorch_ext", k)
    u_cg1.interpolate(u_cg2)
    output_file.write_checkpoint(u_cg1, "output_pytorch_ext", k, df.XDMFFile.Encoding.HDF5, append=True)



input_file.close()
output_file.close()
