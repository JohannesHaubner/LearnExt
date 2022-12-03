from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../FSIsolver/extension_operator')
import extension
sys.path.insert(1, '../FSIsolver/fsi_solver')
import solver
sys.path.insert(1, '../learnExt')
from NeuralNet.neural_network_custom import ANN, generate_weights

import numpy as np
import pygmsh, meshio


# create mesh: first create mesh by running ./create_mesh/create_mesh_FSI.py

# load mesh
mesh = Mesh()
with XDMFFile("./../Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
mvc2 = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("./../Output/Mesh_Generation/facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
with XDMFFile("./../Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mvc2, "name_to_read")
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = cpp.mesh.MeshFunctionSizet(mesh,mvc2)
bdfile = File("./../Output/Mesh_Generation/boundary.pvd")
bdfile << boundaries
bdfile = File("./../Output/Mesh_Generation/domains.pvd")
bdfile << domains

# boundary parts
params = np.load('../Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "obstacle_fluid", "obstacle_solid"]

# subdomains
fluid_domain = MeshView.create(domains, params["fluid"])
solid_domain = MeshView.create(domains, params["solid"])

# function space
T = VectorElement("CG", fluid_domain.ufl_cell(), 2)
FS = FunctionSpace(fluid_domain, T)

# read data
xdmf_input = XDMFFile("../Output/Extension/Data/input.xdmf")
xdmf_output = XDMFFile("../Output/Extension/Data/output.xdmf")

ifile = File("../Output/Extension/input_func.pvd")
ofile = File("../Output/Extension/output_func.pvd")

input = Function(FS)
output = Function(FS)

##
meshfile = XDMFFile("../Output/Mesh_Generation/mesh.xdmf")
meshfile.write(fluid_domain)

mesh = Mesh()
with meshfile as meshfile:
    meshfile.read(mesh)
V = VectorFunctionSpace(mesh, "CG", 1)

xdmf_input_sm = XDMFFile("../Output/Extension/Data/input_submesh.xdmf")
xdmf_output_sm = XDMFFile("../Output/Extension/Data/output_submesh.xdmf")

def project_data():
    i = 0
    error = False
    while i < 200000 and not error:
        try:
            xdmf_input.read_checkpoint(input, "input", i)
            input_proj = project(input, V)
            ifile << input
            xdmf_output.read_checkpoint(output, "output", i)
            output_proj = project(output, V)
            ofile << output
            i = i + 1
            print(i)
            xdmf_input_sm.write_checkpoint(input_proj, "input", i, XDMFFile.Encoding.HDF5,
                                             append=True)
            xdmf_output_sm.write_checkpoint(output_proj, "output", i, XDMFFile.Encoding.HDF5, append=True)
        except Exception as e:
            print(e)
            error = True

def read_data():
    i = 0
    error = False
    while i < 20 and not error:
        try:
            xdmf_input.read_checkpoint(input, "input", i)
            ifile << input
            xdmf_output.read_checkpoint(output, "output", i)
            ofile << output
            i = i+1
            print(i)
        except Exception as e:
            print(e)
            error = True


project_data()