from dolfin import *

class ExtensionOperator(object):
    def __init__(self, mesh):
        self.mesh = mesh

    def extend(self, boundary_conditions):
        """extend the boundary_conditions to the interior of the mesh"""
        raise NotImplementedError

    def custom(self, FSI):
        """custom function for extension operator"""
        return False

class Biharmonic(ExtensionOperator):
    def __init__(self, mesh):
        super().__init__(mesh)

        T = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, MixedElement(T, T))

    def extend(self, boundary_conditions):
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        uz = TrialFunction(self.FS)
        puz = TestFunction(self.FS)
        (u, z) = split(uz)
        (psiu, psiz) = split(puz)

        dx = Measure('dx', domain=self.mesh)

        a = inner(grad(z), grad(psiu)) * dx + inner(z, psiz) * dx - inner(grad(u), grad(psiz)) * dx
        L = Constant(0.0) * psiu[0] * dx

        bc = DirichletBC(self.FS.sub(0), boundary_conditions, 'on_boundary')

        uz = Function(self.FS)

        solve(a == L, uz, bc)

        u_, z_ = uz.split(deepcopy=True)

        save_ext = False
        if save_ext:
            file = File('../../Output/Extension/function.pvd')
            file << u_

        return u_



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

    # subdomains
    fluid_domain = MeshView.create(domains, 7)
    mvcf = MeshValueCollection("size_t", fluid_domain, 1)
    boundariesf = cpp.mesh.MeshFunctionSizet(fluid_domain, mvcf)
    bdfile = File("../../Output/Mesh_Generation/boundaryf.pvd")
    bdfile << boundariesf

    class Boundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    bound = Boundary()

    bound.mark(boundariesf, 0)
    bdfile << boundariesf

    # boundary conditions on whole outer part of fluid domain
    Biharmonic(fluid_domain).extend(Expression(("x[0]", "x[1]"), degree=1))
