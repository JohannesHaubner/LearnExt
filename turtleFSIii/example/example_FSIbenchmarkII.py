from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../extension_operator')
import extension
sys.path.insert(1, '../fsi_solver')
import solver


# load mesh
mesh = Mesh()
with XDMFFile("../../Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
mvc2 = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("../../Output/Mesh_Generation/facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
with XDMFFile("../../Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mvc2, "name_to_read")
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = cpp.mesh.MeshFunctionSizet(mesh,mvc2)
bdfile = File("../../Output/Mesh_Generation/boundary.pvd")
bdfile << boundaries
bdfile = File("../../Output/Mesh_Generation/domains.pvd")
bdfile << domains

# boundary parts
params = np.load('../../Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

# subdomains
fluid_domain = MeshView.create(domains, params["fluid"])
solid_domain = MeshView.create(domains, params["solid"])
#plot(solid_domain)
#plt.show()

# parameters for FSI system
FSI_param = {}

FSI_param['lambdas'] = 2.0e6
FSI_param['mys'] = 0.5e6
FSI_param['rhos'] = 1.0e4
FSI_param['rhof'] = 1.0e3
FSI_param['nyf'] = 1.0e-3

FSI_param['t'] = 0.0
FSI_param['deltat'] = 0.0025
FSI_param['T'] = 15.0

# initial and boundary conditions
FSI_param['initial_cond'] = Constant((0., 0.))
Ubar = 1.0
FSI_param['boundary_cond'] = Expression(("(t < 2)?(1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))):"
                                         "(1.5*Ubar*4.0*x[1]*(0.41 -x[1]))/ 0.1681", "0.0"),
                                        Ubar=Ubar, t=FSI_param['t'], degree=2)

# extension operator
extension_operator = extension.Biharmonic(fluid_domain)

# initialize FSI solver
fsisolver = solver.FSIsolver(mesh, boundaries, params, FSI_param, extension_operator)
fsisolver.solve()

