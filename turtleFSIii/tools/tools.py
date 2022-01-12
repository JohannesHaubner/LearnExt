from dolfin import *
import numpy as np

def transfer_to_subfunc(f, Vbf):
    """
    Transfer function that lives on the whole mesh to the function space Vbf, which lives on a subpart of the

    """
    return interpolate(f, Vbf)

def transfer_subfunction_to_parent(f, f_full):
    """
    Transfers a function from a MeshView submesh to its parent mesh
    keeps f_full on the other subpart of the mesh unchanged
    """

    # Extract meshes
    V_full = f_full.function_space()
    f_f = Function(V_full)
    f_f.vector()[:] = f_full.vector()[:]
    mesh = V_full.mesh()
    V = f.function_space()
    submesh = V.mesh()

    # Build cell mapping between sub and parent meshes
    cell_map = submesh.topology().mapping()[mesh.id()].cell_map()

    # Get cell dofmaps
    dofmap = V.dofmap()
    dofmap_full = V_full.dofmap()

    # Transfer dofs
    for c in cells(submesh):
        f_f.vector()[dofmap_full.cell_dofs(cell_map[c.index()])] = f.vector()[dofmap.cell_dofs(c.index())]

    return f_f


if __name__ == "__main__":
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
    domains = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
    bdfile = File("../../Output/Mesh_Generation/boundary.pvd")
    bdfile << boundaries
    bdfile = File("../../Output/Mesh_Generation/domains.pvd")
    bdfile << domains

    # boundary parts
    params = np.load('../../Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

    # subdomains
    fluid_domain = MeshView.create(domains, params["fluid"])
    solid_domain = MeshView.create(domains, params["solid"])

    # function on whole domain
    V = FunctionSpace(mesh, "CG", 1)
    v = interpolate(Expression("x[0]*x[1]", degree=2), V)
    v.rename("function", "function")

    file = File('../../Output/Tools/function.pvd')
    file << v

    Vf = FunctionSpace(fluid_domain, "CG", 1)
    vf = interpolate(v, Vf)
    vf.rename("function", "function")
    file << vf

    bmesh_fluid = BoundaryMesh(fluid_domain, "exterior")
    Vbf = FunctionSpace(bmesh_fluid, "CG", 1)
    vbf = interpolate(v, Vbf)
    vbf.rename("function", "function")
    file << vbf

    f = interpolate(Constant(1.0), V)
    ff = transfer_subfunction_to_parent(vf, f)
    ff.rename("function", "function")
    file << ff
