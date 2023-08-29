from fenics import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint.overloaded_type import create_overloaded_object
from pathlib import Path


from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))
from FSIsolver.extension_operator.extension import *
from learnExt.learnext_hybridPDENN import LearnExt

# copy 
# ottarph/learnext-learning-problem/data-prep/artificial/working space into str(here.parent)+'/Output'

# load mesh
fluid_mesh = Mesh()
with HDF5File(fluid_mesh.mpi_comm(), str(here.parent) + "/Output/working_space/fluid.h5", 'r') as h5:
    h5.read(fluid_mesh, 'mesh', False)

tdim = fluid_mesh.topology().dim()
fluid_boundaries = MeshFunction('size_t', fluid_mesh, tdim-1, 0)
with HDF5File(fluid_mesh.mpi_comm(), str(here.parent) + "/Output/working_space/fluid.h5", 'r') as h5:
    h5.read(fluid_boundaries, 'boundaries')

fluid_tags = set(fluid_boundaries.array()) - set((0, ))
iface_tags = {6, 9}
zero_displacement_tags = fluid_tags - iface_tags

fluid_mesh = create_overloaded_object(fluid_mesh)

# new boundary markers (should be replaced)
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
boundaries = cpp.mesh.MeshFunctionSizet(fluid_mesh, 1)
boundary.mark(boundaries, boundary_marker)
interface.mark(boundaries, interface_marker)

params["def_boundary_parts"] = ["interface"]
params["zero_boundary_parts"] = ["no_slip"]

# function space
T = VectorElement("CG", fluid_mesh.ufl_cell(), 1)
FS = FunctionSpace(fluid_mesh, T)

# collect data

deformation = []
ext_deformation = []

ifile = File(str(here.parent) + "/Output/Extension/input_func.pvd")
ofile = File(str(here.parent) + "/Output/Extension/output_func.pvd")

for num in range(4):

    xdmf_input = XDMFFile(str(here.parent) + "/Output/working_space/harmonic" + str(num + 1) + ".xdmf")
    xdmf_output = XDMFFile(str(here.parent) + "/Output/working_space/biharmonic" + str(num + 1) + ".xdmf")

    input = Function(FS, name = "input")
    output = Function(FS, name = "output")
    input_FS = Function(FS, name = "input")
    output_FS = Function(FS, name = "output")

    i = 0
    error = False
    while not error:
        try:
            xdmf_input.read_checkpoint(input, "u_harm_cg1", i)
            input_FS.assign(project(input, FS))
            ifile << input_FS
            xdmf_output.read_checkpoint(output, "u_biharm_cg1", i)
            output_FS.assign(project(output, FS))
            ofile << output_FS
            if i%20 == 0:
                deformation.append(input_FS)
                ext_deformation.append(output_FS)
            i = i+1
            print(i)
        except Exception as e:
            #print(e)
            error = True

data = {}
data["input"] = deformation
data["output"] = ext_deformation

output_path = str(here.parent) + "/Output/learnExt/results/"

threshold = 0.0005

learnExt = LearnExt(fluid_mesh, boundaries, params, output_path, 2)

learnExt.learn(data, threshold)
