from fenics import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint.overloaded_type import create_overloaded_object
from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent))
from FSIsolver.extension_operator.extension import *
sys.path.insert(1, '../learnExt')
from learnext_hybridPDENN import LearnExt

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

option_data = True
option_data_1 = True
# 0 one measurement
# 1 several measurements

# read data
if option_data_1:
    deformation = Function(V_mesh)
    def_file_name = "../Output/Extension/Data/output_.xdmf" # "./Mesh/deformation.xdmf"
    try:
        with XDMFFile(def_file_name) as infile:
            infile.read_checkpoint(deformation, "output")
    except ImportError:
        print("run example_FSIbenchmarkII_generate_data.py first")
    # biharmonic extension
    Biharmonic = Biharmonic(fluid_domain)
    ext_deformation = Biharmonic.extend(deformation)
    deformation = [deformation]
    ext_deformation = [ext_deformation]
else:
    deformation = []
    ext_deformation = []
if option_data:
    # function space
    T = VectorElement("CG", fluid_domain.ufl_cell(), 2)
    FS = FunctionSpace(fluid_domain, T)

    xdmf_input = XDMFFile("../Output/Extension/Data/input.xdmf")
    xdmf_output = XDMFFile("../Output/Extension/Data/output.xdmf")

    ifile = File("../Output/Extension/input_func.pvd")
    ofile = File("../Output/Extension/output_func.pvd")

    i = 0
    error = False
    while not error:
        try:
            input = Function(FS)
            output = Function(FS)
            xdmf_input.read_checkpoint(input, "input", i)
            ifile << input
            xdmf_output.read_checkpoint(output, "output", i)
            ofile << output
            if i%40 == 0:
                deformation.append(project(input, FS))
                ext_deformation.append(project(output, FS))
            i = i+1
            print(i)
        except Exception as e:
            #print(e)
            error = True

data = {}
data["input"] = deformation
data["output"] = ext_deformation

output_path = "../Output/learnExt/results/"

threshold = 0.0005

learnExt = LearnExt(fluid_domain, boundaries, params, output_path, 2)

learnExt.learn(data, threshold)
