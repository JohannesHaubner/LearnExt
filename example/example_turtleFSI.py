from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../turtleFSIii/extension_operator')
import extension
sys.path.insert(1, '../turtleFSIii/fsi_solver')
import solver

# example from the turtleFSI package -- slightly changed

# load mesh
mesh = Mesh()
with XDMFFile("./create_mesh/turtle_demo/turtle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
mvc2 = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("./create_mesh/turtle_demo/mf.xdmf") as infile:
    infile.read(mvc, "name_to_read")
with XDMFFile("./create_mesh/turtle_demo/mc.xdmf") as infile:
    infile.read(mvc2, "name_to_read")
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = cpp.mesh.MeshFunctionSizet(mesh,mvc2)
bdfile = File("./../Output/Mesh_Generation/boundary_turtle.pvd")
bdfile << boundaries
bdfile = File("./../Output/Mesh_Generation/domains_turtle.pvd")
bdfile << domains

# boundary parts
params = {}
params["inflow"] = 14
params["outflow"] = 12
params["bottom_wall"] = 11
params["top_wall"] = 13
params["turtle_head_tail"] = 15
params["fluid"] = 1
params["solid"] = 2

params["no_slip_ids"] = ["bottom_wall", "top_wall", "turtle_head_tail"]

# subdomains
fluid_domain = MeshView.create(domains, params["fluid"])
solid_domain = MeshView.create(domains, params["solid"])
#plot(solid_domain)
#plt.show()

# parameters for FSI system
FSI_param = {}

FSI_param['fluid_mesh'] = fluid_domain
FSI_param['solid_mesh'] = solid_domain

FSI_param['lambdas'] = 4.5e5
FSI_param['mys'] = 0.5e4
FSI_param['rhos'] = 1.0e3
FSI_param['rhof'] = 1.0e3
FSI_param['nyf'] = 1.0e-3

FSI_param['t'] = 0.0
FSI_param['deltat'] = 0.005
FSI_param['T'] = 2.0

FSI_param['displacement_point'] = Point((0.6, 0.2))

# boundary conditions, need to be 0 at t = 0
Ubar = 1.0
FSI_param['boundary_cond'] = Expression(("(Ubar*4.0*(0.25 -x[1]*x[1])*0.5*(1-cos(pi/2*t)))", "0.0"),
                                        Ubar=Ubar, t=FSI_param['t'], degree=2)

# extension operator
extension_operator = extension.Biharmonic(fluid_domain)

# save options
FSI_param['save_directory'] = str('./../Output/turtleFSI') #no save if set to None
FSI_param['save_every_N_snapshot'] = 1 # save every 8th snapshot

# initialize FSI solver
fsisolver = solver.FSIsolver(mesh, boundaries, domains, params, FSI_param, extension_operator)
fsisolver.solve()

