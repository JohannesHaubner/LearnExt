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

mesh = boundaries.mesh()

# boundary parts
params = np.load(str(here.parent) + '/Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "obstacle_fluid", "obstacle_solid"]

# dictionary of tags for the boundaries/facets
boundary_labels = {
    "inflow": 1,
    "outflow": 2,
    "walls": 3,
    "obstacle_fluid": 5,
    "obstacle_solid": 4,
    "interface": 6,
}

# dictionary of tags for the subdomains
subdomain_labels = {
    "fluid": 7,
    "solid": 8,
}

# Dictionary with facet-labels from the boundary of each subdomain
subdomain_boundaries = {
    "fluid": ("inflow", "outflow", "walls", "obstacle_fluid", "interface"),
    "solid": ("interface", "obstacle_solid"),
}

#  call SubMeshCollection
meshes = SubMeshCollection(domains, boundaries, subdomain_labels, boundary_labels, subdomain_boundaries)

# subdomains
fluid_domain = meshes.subdomains["fluid"].mesh
solid_domain = meshes.subdomains["solid"].mesh

markers_fluid = meshes.subdomains["fluid"].boundaries
markers_solid = meshes.subdomains["solid"].boundaries

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
FSI_param['deltat'] = 0.01 #0.0025
FSI_param['T'] = 15.0

FSI_param['material_model'] = "IMR"

FSI_param['displacement_point'] = Point((0.6, 0.2))

# boundary conditions, need to be 0 at t = 0
Ubar = 1.0
FSI_param['boundary_cond'] = Expression(("(t < 2)?(1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))):"
                                         "(1.5*Ubar*4.0*x[1]*(0.41 -x[1]))/ 0.1681", "0.0"),
                                        Ubar=Ubar, t=FSI_param['t'], degree=2)


# extension operator
ids = [boundary_labels[i] for i in subdomain_boundaries["fluid"]]
extension_operator = extension.Biharmonic(fluid_domain, markers_fluid, ids)

# save options
FSI_param['save_directory'] = str(here.parent)+ '/Output/FSIbenchmarkII_biharmonic_adaptive_n' #no save if set to None

# initialize FSI solver
fsisolver = solver.FSIsolver(mesh, boundaries, domains, params, FSI_param, extension_operator, warmstart=False)
fsisolver.solve()

