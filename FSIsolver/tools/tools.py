from dolfin import *
import numpy as np
import mpi4py as MPI
comm = MPI.MPI.COMM_WORLD
id = comm.Get_rank()

class Tools():
    def __init__(self, V_full, Vbf):
        self.V_full = V_full
        self.Vbf = Vbf

        self.mesh = V_full.mesh()
        self.submesh = Vbf.mesh()

        # Build cell mapping between sub and parent meshes
        cell_map = self.submesh.topology().mapping()[self.mesh.id()].cell_map()

        # Get cell dofmaps
        dofmap = self.Vbf.dofmap()
        dofmap_full = self.V_full.dofmap()

        self.imin, self.imax = dofmap_full.ownership_range()
        self.iminl, self.imaxl = dofmap.ownership_range()

        d_full = []
        d = []

        # Transfer dofs
        for c in cells(self.submesh):
            d_full.append([dofmap_full.local_to_global_index(i) for i in dofmap_full.cell_dofs(cell_map[c.index()])])
            d.append([dofmap.local_to_global_index(i) for i in dofmap.cell_dofs(c.index())])  

        d_f_a = np.asarray(d_full).flatten()
        d_a = np.asarray(d).flatten()
        d = np.column_stack((d_f_a, d_a))
        self.reduced = np.asarray(list(set([tuple(i) for i in d.tolist()])), dtype='int')




    def transfer_to_subfunc(self, f):
        """
        Transfer function that lives on the whole mesh to the function space Vbf, which lives on a subpart of the

        """
        # Extract meshes
        f_f = Function(self.Vbf)
    
        # gather data on process 0
        data = comm.gather(self.reduced, root=0)
        # send data to all processes
        data = comm.bcast(data, root=0)
        data = np.concatenate(data, axis=0)


        #len data , data2

        f_vec = f.vector().gather(range(f.vector().size()))
        f_f_vec = f_f.vector().gather(range(f_f.vector().size()))
        f_f_vec[data[:,1]] = f_vec[data[:,0]]


        f_f.vector().set_local(f_f_vec[self.iminl:self.imaxl])
        f_f.vector().apply("")
        return f_f

    def transfer_subfunction_to_parent(self, f, f_full):
        """
        Transfers a function from a MeshView submesh to its parent mesh
        keeps f_full on the other subpart of the mesh unchanged
        f needs to be a function in Vbf
        f_full needs to be a function in V_full
        """

        # Extract meshes
        f_f = Function(self.V_full)
        f_f.vector()[:] = f_full.vector()[:]
    
        # gather data on process 0
        data = comm.gather(self.reduced, root=0)
        # send data to all processes
        data = comm.bcast(data, root=0)
        data = np.concatenate(data, axis=0)


        #len data , data2

        f_f_vec = f_f.vector().gather(range(f_full.vector().size()))
        f_vec = f.vector().gather(range(f.vector().size()))
        f_f_vec[data[:,0]] = f_vec[data[:,1]]


        f_f.vector().set_local(f_f_vec[self.imin:self.imax])
        f_f.vector().apply("")

        return f_f


if __name__ == "__main__":
    from pathlib import Path
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

    version = 0

    if version == 1:

        # subdomains
        fluid_domain = MeshView.create(domains, params["fluid"])
        solid_domain = MeshView.create(domains, params["solid"])

    else:
        from subdomains import SubMeshCollection
        # dictionary of tags for the boundaries/facets
        boundary_labels = {
            "inflow": 1,
            "outflow": 2,
            "walls": 3,
            "obstacle_fluid": 5,
            "obstacle_solid": 4,
            "interface": 6,
        }

        # dictionary of tags for the subdomains
        subdomain_labels = {
            "fluid": 7,
            "solid": 8,
        }

        # Dictionary with facet-labels from the boundary of each subdomain
        subdomain_boundaries = {
            "fluid": ("inflow", "outflow", "walls", "obstacle_fluid", "interface"),
            "solid": ("interface", "obstacle_solid"),
        }

        #  call SubMeshCollection
        meshes = SubMeshCollection(domains, boundaries, subdomain_labels, boundary_labels, subdomain_boundaries)

        markers_fluid = meshes.subdomains["fluid"].boundaries
        markers_solid = meshes.subdomains["solid"].boundaries

        fluid_domain = markers_fluid.mesh()
        solid_domain = markers_solid.mesh()

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

    tools = Tools(V, Vf)


    f = interpolate(Constant(1.0), V)
    ff = tools.transfer_subfunction_to_parent(vf, f)
    ff.rename("function", "function")
    file << ff

    f = tools.transfer_to_subfunc(ff)
    f.rename("function", "function")
    file << f
