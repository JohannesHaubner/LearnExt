from dolfin import *
import numpy as np
#import matplotlib.pyplot as plt

from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))
import FSIsolver.extension_operator.extension as extension
import FSIsolver.fsi_solver.solver as solver
from FSIsolver.tools.subdomains import SubMeshCollection

# create mesh: first create mesh by running ./create_mesh/create_mesh_FSI.py

# load mesh
mesh = Mesh()
with XDMFFile(str(here.parent) + "/Output/Mesh_Generation/mesh_triangles_fsi2.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
mvc2 = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(str(here.parent) + "/Output/Mesh_Generation/facet_mesh_fsi2.xdmf") as infile:
    infile.read(mvc, "name_to_read")
with XDMFFile(str(here.parent) + "/Output/Mesh_Generation/mesh_triangles_fsi2.xdmf") as infile:
    infile.read(mvc2, "name_to_read")
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = cpp.mesh.MeshFunctionSizet(mesh,mvc2)
bdfile = File(str(here.parent) + "/Output/Mesh_Generation/boundary.pvd")
bdfile << boundaries
bdfile = File(str(here.parent) + "/Output/Mesh_Generation/domains.pvd")
bdfile << domains

print('mesh read')

mesh = boundaries.mesh()

# boundary parts
params = np.load(str(here.parent) + '/Output/Mesh_Generation/params_fsi2.npy', allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "solid_left", "solid_right"]

# dictionary of tags for the boundaries/facets
boundary_labels = {
    "inflow": 1,
    "outflow": 2,
    "walls": 3,
    "solid_left": 4,
    "solid_righ": 5,
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
meshes = SubMeshCollection(domains, boundaries, subdomain_labels, boundary_labels, subdomain_boundaries)

print('set up finished')

# subdomains
fluid_domain = meshes.subdomains["fluid"].mesh
solid_domain = meshes.subdomains["solid"].mesh

markers_fluid = meshes.subdomains["fluid"].boundaries
markers_solid = meshes.subdomains["solid"].boundaries

# parameters for FSI system
FSI_param = {}

FSI_param['fluid_mesh'] = fluid_domain
FSI_param['solid_mesh'] = solid_domain

FSI_param['lambdas'] = 1e5
FSI_param['mys'] = 2e7
FSI_param['rhos'] = 0.8e4
FSI_param['rhof'] = 1.0e4
FSI_param['nyf'] = 4e-3

FSI_param['t'] = 0.0
FSI_param['deltat'] = 0.01 #0.0025
FSI_param['T'] = 5.0

FSI_param['material_model'] = "IMR"
FSI_param['bc_type'] = "pressure"

FSI_param['displacement_point'] = Point((0.6, 0.2))

# boundary conditions, need to be 0 at t = 0
FSI_param['boundary_cond'] = Expression("5e5*t", t=FSI_param['t'], degree=2)


# extension operator
ids = [boundary_labels[i] for i in subdomain_boundaries["fluid"]]
extension_operator = extension.Biharmonic(fluid_domain, markers_fluid, ids)

# save options
FSI_param['save_directory'] = str(here.parent)+ '/Output/FSIbenchmarkII_biharmonic_adaptive_n' #no save if set to None

# initialize FSI solver
fsisolver = solver.FSIsolver(mesh, boundaries, domains, params, FSI_param, extension_operator, warmstart=False)
fsisolver.solve()

