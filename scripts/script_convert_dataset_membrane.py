# from dolfin import *
import dolfin as df

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import sys, os



# load mesh
mesh = df.Mesh()
with df.XDMFFile("Output/Mesh_Generation/mesh_triangles_fsi2.xdmf") as infile:
    infile.read(mesh)
mvc = df.MeshValueCollection("size_t", mesh, 2)
mvc2 = df.MeshValueCollection("size_t", mesh, 2)
with df.XDMFFile("Output/Mesh_Generation/facet_mesh_fsi2.xdmf") as infile:
    infile.read(mvc, "name_to_read")
with df.XDMFFile("Output/Mesh_Generation/mesh_triangles_fsi2.xdmf") as infile:
    infile.read(mvc2, "name_to_read")
boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = df.cpp.mesh.MeshFunctionSizet(mesh,mvc2)


# print('mesh read')
mesh = boundaries.mesh()


# boundary parts
params = np.load('Output/Mesh_Generation/params_fsi2.npy', allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "solid_left", "solid_right"]

# dictionary of tags for the boundaries/facets
boundary_labels = {
    "inflow": 1,
    "outflow": 2,
    "walls": 3,
    "solid_left": 4,
    "solid_right": 5,
    "interface": 6,
}

# dictionary of tags for the subdomains
subdomain_labels = {
    "fluid": 7,
    "solid": 8,
}

# Dictionary with facet-labels from the boundary of each subdomain
subdomain_boundaries = {
    "fluid": ("inflow", "outflow", "walls", "interface"),
    "solid": ("interface", "solid_left", "solid_right"),
}

#  call SubMeshCollection
# meshes = SubMeshCollection(domains, boundaries, subdomain_labels, boundary_labels, subdomain_boundaries)

# print('set up finished')

# subdomains
# fluid_domain = meshes.subdomains["fluid"].mesh
# solid_domain = meshes.subdomains["solid"].mesh

# markers_fluid = meshes.subdomains["fluid"].boundaries
# markers_solid = meshes.subdomains["solid"].boundaries



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



if Path("Data_MoF/membrane_test_p1.xdmf").exists():
    Path("Data_MoF/membrane_test_p1.xdmf").unlink()
    Path("Data_MoF/membrane_test_p1.h5").unlink()
if Path("Data_MoF/membrane_test_p2.xdmf").exists():
    Path("Data_MoF/membrane_test_p2.xdmf").unlink()
    Path("Data_MoF/membrane_test_p2.h5").unlink()

old_xdmf = df.XDMFFile("Data_MoF/membrane_test.xdmf")

new_xdmf_p1 = df.XDMFFile("Data_MoF/membrane_test_p1.xdmf")
new_xdmf_p1.write(f_domain)
new_xdmf_p2 = df.XDMFFile("Data_MoF/membrane_test_p2.xdmf")
new_xdmf_p2.write(f_domain)

V = df.VectorFunctionSpace(fluid_domain, "CG", 2)
V_full = df.VectorFunctionSpace(mesh, "CG", 2)
V_f = df.VectorFunctionSpace(f_domain, "CG", 2)
V_f_p1 = df.VectorFunctionSpace(f_domain, "CG", 1)

def MeshView_to_Submesh(u):
    u_full = df.Function(V_full)
    u_fluid = transfer_subfunction_to_parent(u, u_full)
    u_f = df.interpolate(u_fluid, V_f)
    return u_f
    


u = df.Function(V)
old_xdmf.read_checkpoint(u, "output_biharmonic_ext", 0)
u_cg1 = df.Function(V_f_p1)


f_domain_CG2_to_CG1 = df.PETScDMCollection.create_transfer_matrix(V_f, V_f_p1)

from tqdm import tqdm
checkpoints = range(0, 89)
for k in tqdm(checkpoints): 
    old_xdmf.read_checkpoint(u, "output_biharmonic_ext", k)
    uf = MeshView_to_Submesh(u)
    f_domain_CG2_to_CG1.mult(uf.vector(), u_cg1.vector())
    new_xdmf_p2.write_checkpoint(uf, "uh", k, df.XDMFFile.Encoding.HDF5, append=True)
    new_xdmf_p1.write_checkpoint(u_cg1, "uh", k, df.XDMFFile.Encoding.HDF5, append=True)

old_xdmf.close()
new_xdmf_p2.close()
new_xdmf_p1.close()
