# from dolfin import *
import dolfin as df

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here))
import FSIsolver.extension_operator.extension as extension
import FSIsolver.fsi_solver.solver as solver

""" Loading the mesh in the same way as in the example scripts """


# create mesh: first create mesh by running ./create_mesh/create_mesh_FSI.py

# load mesh
mesh = df.Mesh()
with df.XDMFFile(str(here) + "/Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mesh)
mvc = df.MeshValueCollection("size_t", mesh, 2)
mvc2 = df.MeshValueCollection("size_t", mesh, 2)
with df.XDMFFile(str(here) + "/Output/Mesh_Generation/facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
with df.XDMFFile(str(here) + "/Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mvc2, "name_to_read")
boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = df.cpp.mesh.MeshFunctionSizet(mesh,mvc2)
bdfile = df.File(str(here) + "/Output/Mesh_Generation/boundary.pvd")
bdfile << boundaries
bdfile = df.File(str(here) + "/Output/Mesh_Generation/domains.pvd")
bdfile << domains

# boundary parts
params = np.load(str(here) + '/Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "obstacle_fluid", "obstacle_solid"]

# subdomains
fluid_domain = df.MeshView.create(domains, params["fluid"])
solid_domain = df.MeshView.create(domains, params["solid"])
f_domain = df.SubMesh(mesh, domains, params["fluid"])



def transfer_subfunction_to_parent(f, f_full):
    """
    Transfers a function from a MeshView submesh to its parent mesh
    keeps f_full on the other subpart of the mesh unchanged
    """

    # Extract meshes
    V_full = f_full.function_space()
    f_f = df.Function(V_full)
    f_f.vector()[:] = f_full.vector()[:]
    mesh = V_full.mesh()
    V = f.function_space()
    submesh = V.mesh()

    # Build cell mapping between sub and parent meshes
    cell_map = submesh.topology().mapping()[mesh.id()].cell_map()

    # Get cell dofmaps
    dofmap = V.dofmap()
    dofmap_full = V_full.dofmap()

    # Transfer dofs
    for c in df.cells(submesh):
        f_f.vector()[dofmap_full.cell_dofs(cell_map[c.index()])] = f.vector()[dofmap.cell_dofs(c.index())]

    return f_f



print(fluid_domain.num_vertices())

old_xdmf = df.XDMFFile("Output/Extension/Data/FSIbenchmarkII_data_.xdmf")

new_xdmf = df.XDMFFile("Output/Extension/Data/FSIbenchmarkII_data_new.xdmf")
new_xdmf.write(f_domain)

V = df.VectorFunctionSpace(fluid_domain, "CG", 2)
V_full = df.VectorFunctionSpace(mesh, "CG", 2)
V_f = df.VectorFunctionSpace(f_domain, "CG", 2)

def MeshView_to_Submesh(u):
    u_full = df.Function(V_full)
    u_fluid = transfer_subfunction_to_parent(u, u_full)
    u_f = df.interpolate(u_fluid, V_f)
    return u_f
    


u = df.Function(V)
old_xdmf.read_checkpoint(u, "output_biharmonic_ext", 0)

print(u.function_space())


from tqdm import tqdm
checkpoints = range(0, 80)
for k in tqdm(checkpoints):
    old_xdmf.read_checkpoint(u, "output_biharmonic_ext", k)
    uf = MeshView_to_Submesh(u)
    new_xdmf.write_checkpoint(uf, "data", k, df.XDMFFile.Encoding.HDF5, append=True)

old_xdmf.close()
new_xdmf.close()
