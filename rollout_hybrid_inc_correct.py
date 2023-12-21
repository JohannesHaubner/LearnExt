import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

import tqdm
from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))


# create mesh: first create mesh by running ./create_mesh/create_mesh_FSI.py
mesh_dir = Path("/mnt/LearnExt/Output/Mesh_Generation")

# load mesh
mesh = df.Mesh()
with df.XDMFFile(str(mesh_dir / "mesh_triangles.xdmf")) as infile:
    infile.read(mesh)
mvc = df.MeshValueCollection("size_t", mesh, 2)
mvc2 = df.MeshValueCollection("size_t", mesh, 2)
with df.XDMFFile(str(mesh_dir / "facet_mesh.xdmf")) as infile:
    infile.read(mvc, "name_to_read")
with df.XDMFFile(str(mesh_dir / "mesh_triangles.xdmf")) as infile:
    infile.read(mvc2, "name_to_read")
boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = df.cpp.mesh.MeshFunctionSizet(mesh,mvc2)

# boundary parts
params = np.load(mesh_dir / 'params.npy', allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "obstacle_fluid", "obstacle_solid"]

# subdomains
fluid_domain = df.MeshView.create(domains, params["fluid"])
solid_domain = df.MeshView.create(domains, params["solid"])

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
# FSI_param['deltat'] = 0.0025 # 0.01
FSI_param['deltat'] = 0.01
# FSI_param['T'] = 17.0
FSI_param['T'] = 1.0

FSI_param['displacement_point'] = df.Point((0.6, 0.2))

# boundary conditions, need to be 0 at t = 0
Ubar = 1.0
FSI_param['boundary_cond'] = df.Expression(("(t < 2)?(1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))):"
                                         "(1.5*Ubar*4.0*x[1]*(0.41 -x[1]))/ 0.1681", "0.0"),
                                        Ubar=Ubar, t=FSI_param['t'], degree=2)

Fspace = df.VectorFunctionSpace(fluid_domain, "CG", 2)
fspace = df.VectorFunctionSpace(fluid_domain, "CG", 1)


import FSIsolver.extension_operator.extension as extension
from learnExt.NeuralNet.neural_network_custom import ANN
from learnExt.learnext_hybridPDENN import Custom_Reduced_Functional as crf


# extension operator
class LearnExtension(extension.ExtensionOperator):
    def __init__(self, mesh):
        super().__init__(mesh)

        T = df.VectorElement("CG", self.mesh.ufl_cell(), 1)
        T2 = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = df.FunctionSpace(self.mesh, T)
        self.FS2 = df.FunctionSpace(self.mesh, T2)
        self.bc_old = df.Function(self.FS)
        network_path = "example/learned_networks/trained_network.pkl"
        self.net = ANN(network_path)
        self.threshold = 0.001

        self.first = True


    def extend(self, boundary_conditions, params = None):
        """ harmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        displacementy = params["displacementy"]

        if abs(displacementy) <= 0.005:
            trafo = False
        elif self.first:
            trafo = False
            self.first = False
        else:
            trafo = True

        u = df.Function(self.FS2)
        v = df.TestFunction(self.FS2)


        if trafo:
            up = df.project(self.bc_old, self.FS)
            upi = df.project(-1.0*up, self.FS)
            df.ALE.move(self.mesh, up, annotate=False)

        dx = df.Measure('dx', domain=self.mesh, metadata={'quadrature_degree': 4})

        if trafo:
            E = df.inner(crf.NN_der(self.threshold, df.inner(df.grad(self.bc_old), df.grad(self.bc_old)), self.net) * df.grad(u), df.grad(v)) * dx
        
        else:
            E = df.inner(crf.NN_der(self.threshold, df.inner(df.grad(u), df.grad(u)), self.net) * df.grad(u), df.grad(v)) * dx

        # solve PDE
        if trafo:
            bc_func = df.project(boundary_conditions - self.bc_old, self.FS2)
        else:
            bc_func = boundary_conditions

        bc = df.DirichletBC(self.FS2, bc_func, 'on_boundary')


        df.solve(E == 0, u, bc, solver_parameters={"nonlinear_solver": "newton", "newton_solver":
            {"maximum_iterations": 200}})

        if trafo:
            u = df.project(u + self.bc_old, self.FS2)


        self.bc_old.assign(df.project(u, self.FS))

        if trafo:
            df.ALE.move(self.mesh, upi, annotate=False)

        return u

extension_operator = LearnExtension(fluid_domain)



input_xdmf_file = df.XDMFFile("Output/Extension/Data/input_.xdmf")
if Path("hybridIncCorrRolloutP1.xdmf").exists():
    Path("hybridIncCorrRolloutP1.xdmf").unlink()
    Path("hybridIncCorrRolloutP1.h5").unlink
# if Path("hybridIncCorrRolloutP2.xdmf").exists():
#     Path("hybridIncCorrRolloutP2.xdmf").unlink()
#     Path("hybridIncCorrRolloutP2.h5").unlink
output_xdmf_file_p1 = df.XDMFFile("hybridIncCorrRolloutP1.xdmf")
output_xdmf_file_p1.write(fluid_domain)
# output_xdmf_file_p2 = df.XDMFFile("hybridIncCorrRolloutP2.xdmf")
# output_xdmf_file_p2.write(fluid_domain)

df.set_log_active(False)
u_bc = df.Function(Fspace)
u_p1 = df.Function(fspace)

dofs = np.arange(Fspace.dim())[np.linalg.norm(Fspace.tabulate_dof_coordinates() - np.array([0.6, 0.2]), axis=1) < 1e-7]
dof = dofs[1]
for k in tqdm.tqdm(range(2*206)): # 2 * number of time steps that make it loop best.
    input_xdmf_file.read_checkpoint(u_bc, "input_harmonic_ext", k)
    disp_y = u_bc.vector()[dof]
    u_ext = extension_operator.extend(u_bc, {"displacementy": disp_y})
    u_p1.interpolate(u_ext)
    # output_xdmf_file_p2.write_checkpoint(u_ext, "uh", k, append=True)
    output_xdmf_file_p1.write_checkpoint(u_p1, "uh", k, append=True)


