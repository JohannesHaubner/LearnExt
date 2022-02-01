from fenics import *
from dolfin_adjoint import *
import numpy as np
import coeff_opt_control as opt_cont
import coeff_machine_learning as opt_ml

threshold  = 0.05 # first part of NN is linear

if __name__ == "__main__":
    # load mesh
    mesh = Mesh()
    with XDMFFile("./Mesh/mesh_triangles.xdmf") as infile:
        infile.read(mesh)

    # read boundary parts
    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile("./Mesh/facet_mesh.xdmf") as infile:
        infile.read(mvc, "name_to_read")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    params = np.load('./Mesh/params.npy', allow_pickle='TRUE').item()

    bfile = File("../Output/learnExt/boundaries.pvd")
    bfile << boundaries

    # save mesh
    deformation = Expression(("1e-2 * exp(x[0]-6.)", "0."), degree = 2)
    def_boundary_parts = ["design"]
    zero_boundary_parts = ["inflow", "outflow", "noslip"]
    output_directory = str("../Output/learnExt/results/")

    # function spaces
    Vs = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)

    recompute_optimal_control = False
    if recompute_optimal_control:
        opt_cont.compute_optimal_coefficient(mesh, V, Vs, params, deformation, def_boundary_parts,
                                             zero_boundary_parts, boundaries, output_directory)

    recompute_neural_net = True
    if recompute_neural_net:
        opt_ml.compute_machine_learning(mesh, V, Vs, params, boundaries, output_directory, threshold)

    opt_ml.visualize(mesh, V, Vs, params, deformation, def_boundary_parts,
                                             zero_boundary_parts, boundaries, output_directory, threshold)