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

mesh_dir = Path("Output/Mesh_Generation")
# load mesh
mesh = Mesh()
with XDMFFile(str(mesh_dir / "mesh_triangles.xdmf")) as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
mvc2 = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(str(mesh_dir / "facet_mesh.xdmf")) as infile:
    infile.read(mvc, "name_to_read")
with XDMFFile(str(mesh_dir / "mesh_triangles.xdmf")) as infile:
    infile.read(mvc2, "name_to_read")
#boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = cpp.mesh.MeshFunctionSizet(mesh, mvc2)

params = np.load(str(mesh_dir / "params.npy"), allow_pickle='TRUE').item()

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



T = VectorElement("CG", fluid_domain.ufl_cell(), 2)
FS = FunctionSpace(fluid_domain, T)

deformation = []
ext_deformation = []

data_dir = Path("Output/Extension/Data")
input_file = df.XDMFFile(str(data_dir / "input_.xdmf"))
input_tag = "input_harmonic_ext"
output_file = df.XDMFFile(str(data_dir / "output_.xdmf"))
output_tag = "output_biharmonic_ext"

for k in range(0,800,20):
    func = df.Function(FS)
    input_file.read_checkpoint(func, input_tag, k)
    deformation.append(func)
    func = df.Function(FS)
    output_file.read_checkpoint(func, output_tag, k)
    ext_deformation.append(func)

data = {}
data["input"] = deformation
data["output"] = ext_deformation

output_path = str(here.parent) + "/Output/learnExt/results/"

threshold = 0.0005

learnExt = LearnExt(fluid_domain, boundaries, params, output_path, 2)

learnExt.learn(data, threshold)
