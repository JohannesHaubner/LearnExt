from dolfin import *
import numpy as np
import mpi4py as MPI

def transfer_to_subfunc(f, Vbf):
    """
    Transfer function that lives on the whole mesh to the function space Vbf, which lives on a subpart of the

    """
    return interpolate(f, Vbf)

def transfer_subfunction_to_parent(f, f_full):
    """
    Transfers a function from a MeshView submesh to its parent mesh
    keeps f_full on the other subpart of the mesh unchanged
    # TODO write this in parallel
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


    d_full = []
    d = []
    imin, imax = dofmap_full.ownership_range()

    # Transfer dofs
    for c in cells(submesh):
        d_full.append([dofmap_full.local_to_global_index(i) for i in dofmap_full.cell_dofs(cell_map[c.index()])])
        d.append([dofmap.local_to_global_index(i) for i in dofmap.cell_dofs(c.index())])  
        #f_f.vector().set_local(dofmap_full.cell_dofs(cell_map[c.index()])) = f.vector()[dofmap.cell_dofs(c.index())]


    d_f_a = np.asarray(d_full).flatten()
    d_a = np.asarray(d).flatten()
    d = np.column_stack((d_f_a, d_a))
    reduced = np.asarray(list(set([tuple(i) for i in d.tolist()])), dtype='int')
   
    # gather data on process 0
    data = comm.gather(reduced, root=0)
    # send data to all processes
    data = comm.bcast(data, root=0)
    data = np.concatenate(data, axis=0)


    #len data , data2

    f_f_vec = f_f.vector().gather(range(f_full.vector().size()))
    f_vec = f.vector().gather(range(f.vector().size()))
    f_f_vec[data[:,0]] = f_vec[data[:,1]]


    f_f.vector().set_local(f_f_vec[imin:imax])
    f_f.vector().apply("")

    return f_f


if __name__ == "__main__":
    from pathlib import Path
    comm = MPI.MPI.COMM_WORLD
    id = comm.Get_rank()

    here = Path(__file__).parent.parent.parent.resolve()
    # load mesh
    mesh = Mesh()
    with XDMFFile(str(here) + "/Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)
    mvc2 = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(str(here) + "/Output/Mesh_Generation/facet_mesh.xdmf") as infile:
        infile.read(mvc, "name_to_read")
    with XDMFFile(str(here) + "/Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
        infile.read(mvc2, "name_to_read")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    domains = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
    bdfile = File(str(here) + "/Output/Mesh_Generation/boundary.pvd")
    bdfile << boundaries
    bdfile = File(str(here) + "/Output/Mesh_Generation/domains.pvd")
    bdfile << domains

    # boundary parts
    params = np.load(str(here) + '/Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

    # subdomains
    fluid_domain = MeshView.create(domains, params["fluid"])
    solid_domain = MeshView.create(domains, params["solid"])

    # function on whole domain
    V = FunctionSpace(mesh, "CG", 2)
    v = interpolate(Expression("x[0]*x[1]", degree=2), V)
    v.rename("function", "function")

    file = File(str(here) + '/Output/Tools/function.pvd')
    file << v

    Vf = FunctionSpace(fluid_domain, "CG", 2)
    vf = interpolate(v, Vf)
    vf.rename("function", "function")
    file << vf


    f = interpolate(Constant(1.0), V)
    ff = transfer_subfunction_to_parent(vf, f)
    ff.rename("function", "function")
    file << ff
