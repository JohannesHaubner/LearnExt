from fenics import *
from dolfin_adjoint import *
import numpy as np
import coeff_opt_control as opt_cont
import coeff_machine_learning as opt_ml
from pyadjoint.overloaded_type import create_overloaded_object

threshold  = 0.001 # first part of NN is linear

if __name__ == "__main__":
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
    params = {}
    params["on_boundary"] = boundary_marker
    class Boundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    boundary = Boundary()
    boundaries = cpp.mesh.MeshFunctionSizet(fluid_domain, 1)
    boundary.mark(boundaries, boundary_marker)

    # save mesh
    def_boundary_parts = ["on_boundary"]
    zero_boundary_parts = []
    output_directory = str("../Output/learnExt/results/")

    # function spaces
    Vs = FunctionSpace(fluid_domain, "CG", 1)
    V = VectorFunctionSpace(fluid_domain, "CG", 2)
    V2 = VectorFunctionSpace(fluid_domain, "CG", 2)
    V2_mesh = VectorFunctionSpace(mesh, "CG", 2)

    deformation = Function(V2_mesh)
    with XDMFFile("./Mesh/deformation.xdmf") as infile:
        infile.read_checkpoint(deformation, "u_")
    deformation = interpolate(deformation, V2)
    #deformation = project(deformation, V)

    file = File('../Output/learnExt/results/deformation.pvd')
    file << deformation
    file = File('../Output/learnExt/results/boundaries.pvd')
    file << boundaries

    recompute_optimal_control = True
    if recompute_optimal_control:
        opt_cont.compute_optimal_coefficient(fluid_domain, V, Vs, params, deformation, def_boundary_parts,
                                             zero_boundary_parts, boundaries, output_directory)

    recompute_neural_net = True
    if recompute_neural_net:
        opt_ml.compute_machine_learning(fluid_domain, V, Vs, params, boundaries, output_directory, threshold)

    opt_ml.visualize(fluid_domain, V, Vs, params, deformation, def_boundary_parts,
                                             zero_boundary_parts, boundaries, output_directory, threshold)