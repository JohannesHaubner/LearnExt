from fenics import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint.overloaded_type import create_overloaded_object
from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent))
from FSIsolver.extension_operator.extension import *
from learnExt.NeuralNet.neural_network_custom import ANN, generate_weights
from learnExt.learnext import *

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
#boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = cpp.mesh.MeshFunctionSizet(mesh, mvc2)

params = np.load('./../Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

# subdomains
fluid_domain = MeshView.create(domains, params["fluid"])
solid_domain = MeshView.create(domains, params["solid"])
fluid_domain = create_overloaded_object(fluid_domain)

# boundaries
boundary_marker = 1
interface_marker = 2
params = {}
params["no_slip"] = boundary_marker
params["interface"] = interface_marker
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and between(x[0], (0.2, 0.6)) and between(x[1], (0.19, 0.21))
boundary = Boundary()
interface = Interface()
boundaries = cpp.mesh.MeshFunctionSizet(fluid_domain, 1)
boundary.mark(boundaries, boundary_marker)
interface.mark(boundaries, interface_marker)

params["def_boundary_parts"] = ["interface"]
params["zero_boundary_parts"] = ["no_slip"]

V_mesh = VectorFunctionSpace(mesh, "CG", 2)
deformation = Function(V_mesh)
def_file_name = "../Output/files/learned/states.xdmf" # "./Mesh/deformation.xdmf"
with XDMFFile(def_file_name) as infile:
    infile.read_checkpoint(deformation, "u")

output_path = "../Output/learnExt/results/"

threshold = 0.001

learnExt = LearnExt(fluid_domain, boundaries, params, output_path, 2)

learnExt.learn(deformation, threshold)