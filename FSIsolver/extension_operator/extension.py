from dolfin import *
import xml.etree.ElementTree as ET

from pathlib import Path
here = Path(__file__).parent.parent.parent.resolve()
import sys, os
sys.path.insert(0, str(here))
from learnExt.NeuralNet.neural_network_custom import ANN, generate_weights
from learnExt.learnext_hybridPDENN import Custom_Reduced_Functional as crf

class ExtensionOperator(object):
    def __init__(self, mesh, marker, ids):
        self.mesh = mesh
        self.marker = marker
        self.ids = ids

    def extend(self, boundary_conditions, params=None):
        """extend the boundary_conditions to the interior of the mesh"""
        raise NotImplementedError

    def custom(self, FSI):
        """custom function for extension operator"""
        return False
    
    @staticmethod
    def timings_extension(func):
        """decorator to get timings for extension"""
        def wrapper(self, *arg, **kw):
            dump_timings_to_xml('test_timings.xml', TimingClear.clear)

            with Timer("do extension"):
                res = func(self, *arg, **kw)

            dump_timings_to_xml('test_timings.xml', TimingClear.keep)

            tree = ET.parse('test_timings.xml')
            root = tree.getroot()

            mpi_kind = 'MPI_MAX'

            data = {}
            for table in root:
                if mpi_kind not in table.get('name'):
                    continue
                
                for row in table:
                    row_data = {}
                    for col in row:
                        row_data[col.get('key')] = col.get('value')
                    data[row.get('key')] = row_data

            col = 'wall tot'
            for process in data:
                print(process, data[process]['reps'], data[process][col])

            return res
        return wrapper

class Biharmonic(ExtensionOperator):
    def __init__(self, mesh, marker=None, ids=None, save_extension=False, save_filename=None):
        super().__init__(mesh, marker, ids)

        # options
        self.save_ext = save_extension

        if self.save_ext:
            self.iter = -1
            if save_filename == None:
                raise Exception('save_filename (str) not specified')
            self.xdmf_output = XDMFFile(str(save_filename))
            self.xdmf_output.write(self.mesh)

        T = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, MixedElement(T, T))

        uz = TrialFunction(self.FS)
        puz = TestFunction(self.FS)
        (u, z) = split(uz)
        (psiu, psiz) = split(puz)

        dx = Measure('dx', domain=self.mesh)

        a = inner(grad(z), grad(psiu)) * dx + inner(z, psiz) * dx - inner(grad(u), grad(psiz)) * dx
        L = Constant(0.0) * psiu[0] * dx


        self.A = assemble(a)

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS.sub(0), Constant((0.,0.)), 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS.sub(0), Constant((0., 0.)), self.marker, i))
        self.bc = bc

        for bci in self.bc:
            bci.apply(self.A)

        self.solver = LUSolver(self.A)

        self.L = assemble(L)


    @ExtensionOperator.timings_extension
    def extend(self, boundary_conditions, params=None):
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS.sub(0), boundary_conditions, 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS.sub(0), boundary_conditions, self.marker, i))

        for bci in bc:
            bci.apply(self.L)

        uz = Function(self.FS)

        self.solver.solve(uz.vector(), self.L)

        u_, z_ = uz.split(deepcopy=True)

        if self.save_ext:
            self.iter +=1
            self.xdmf_output.write_checkpoint(u_, "output_biharmonic_ext", self.iter, XDMFFile.Encoding.HDF5, append=True)

        return u_
    
class Harmonic(ExtensionOperator):
    def __init__(self, mesh, marker=None, ids=None, save_extension=True, save_filename=None, incremental=False):
        super().__init__(mesh, marker, ids)

        # options
        self.save_ext = save_extension

        if self.save_ext:
            self.iter = -1
            if save_filename == None:
                raise Exception('save_filename (str) not specified')
            self.xdmf_output = XDMFFile(str(save_filename))
            self.xdmf_output.write(self.mesh)

        T = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, T)

        self.incremental = incremental
        if self.incremental:
            self.bc_old = Function(self.FS)

        u = TrialFunction(self.FS)
        v = TestFunction(self.FS)

        dx = Measure('dx', domain=self.mesh)

        a = inner(grad(u), grad(v)) * dx
        L = Constant(0.0) * v[0] * dx


        self.A = assemble(a)

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS, Constant((0.,0.)), 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS, Constant((0., 0.)), self.marker, i))
        self.bc = bc

        for bci in self.bc:
            bci.apply(self.A)

        self.solver = LUSolver(self.A)

        self.L = assemble(L)


    @ExtensionOperator.timings_extension
    def extend(self, boundary_conditions, params=None):
        """ biharmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        bc = []
        if self.marker == None:
            bc.append(DirichletBC(self.FS, boundary_conditions, 'on_boundary'))
        else:
            for i in self.ids:
                bc.append(DirichletBC(self.FS, boundary_conditions, self.marker, i))

        for bci in bc:
            bci.apply(self.L)

        if self.incremental:
            # move mesh with previous deformation
            self.bc_old = boundary_conditions
            up = project(self.bc_old, self.FS)
            upi = Function(self.FS)
            upi.vector().axpy(-1., up.vector())
            ALE.move(self.mesh, up)

        u_ = Function(self.FS)

        self.solver.solve(u_.vector(), self.L)

        if self.incremental:
            # move mesh back
            ALE.move(self.mesh, upi)

        if self.save_ext:
            self.iter +=1
            self.xdmf_output.write_checkpoint(u_, "output_biharmonic_ext", self.iter, XDMFFile.Encoding.HDF5, append=True)

        return u_
    
   
class LearnExtension(ExtensionOperator):
    def __init__(self, mesh, NN_path, threshold=None, marker=None, ids=None, save_extension=False, save_filename=None, incremental=False, incremental_corrected=False):
        super().__init__(mesh, marker, ids)

        T = VectorElement("CG", self.mesh.ufl_cell(), 1)
        T2 = VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.FS = FunctionSpace(self.mesh, T)
        self.FS2 = FunctionSpace(self.mesh, T2)
        self.bc_old = Function(self.FS2)
        self.net = ANN(NN_path)

        self.save_ext = save_extension
        self.incremental = incremental
        self.incremental_corrected = incremental_corrected

        if threshold == None:
            print('Threshold set to 0')
            self.threshold = 0.0
        else:
            self.threshold = threshold

        if self.save_ext:
            self.iter = -1
            if save_filename == None:
                raise Exception('save_filename (str) not specified')
            self.xdmf_output = XDMFFile(str(save_filename))
            self.xdmf_output.write(self.mesh)

    @ExtensionOperator.timings_extension
    def extend(self, boundary_conditions, params = None):
        """ harmonic extension of boundary_conditions (Function on self.mesh) to the interior """

        if params != None:
            try:
                b_old = params["b_old"]
            except:
                pass
            try:
                displacementy = params["displacementy"]
            except:
                displacementy = None

        if self.incremental == True and self.incremental_correct == False:
            trafo = True
        elif self.incremental == True and self.incremental_correct == True:
            if displacementy == None:
                Warning("displacementy == None; set trafo to False")
                trafo = False
            elif abs(displacementy) <= 0.005:
                print('displacementy <= 0.005: displacementy = ', displacementy)
                trafo = False
            else:
                trafo = True
        else:
            trafo = False

        if b_old != None:
            self.bc_old = project(b_old, self.FS)

        if trafo:
            up = project(self.bc_old, self.FS)
            upi = Function(self.FS)
            upi.vector().axpy(-1.0, up.vector())
            try:
                ALE.move(self.mesh, up, annotate=False)
            except:
                ALE.move(self.mesh, up)

        if trafo:
            u = TrialFunction(self.FS2)
        else:
            u = Function(self.FS2)
        v = TestFunction(self.FS2)

        dx = Measure('dx', domain=self.mesh, metadata={'quadrature_degree': 4})

        if trafo:
            E = inner(crf.NN_der(self.threshold, inner(grad(self.bc_old), grad(self.bc_old)), self.net) * grad(u), grad(v)) * dx
        else:
            E = inner(crf.NN_der(self.threshold, inner(grad(u), grad(u)), self.net) * grad(u), grad(v)) * dx

        # solve PDE
        if trafo:
            bc_func = project(boundary_conditions - self.bc_old, self.FS2)
        else:
            bc_func = boundary_conditions
        bc = DirichletBC(self.FS2, bc_func, 'on_boundary')

        if trafo:
            u = Function(self.FS2)
            solve(lhs(E) == rhs(E), u, bc)
        else:
            solve(E == 0, u, bc, solver_parameters={"nonlinear_solver": "newton", "newton_solver":
                {"maximum_iterations": 200}})

        if trafo:
            u = project(u + self.bc_old, self.FS2)
        self.bc_old.assign(project(u, self.FS))

        if self.save_ext:
            self.iter +=1
            self.xdmf_output.write_checkpoint(u_, "output_biharmonic_ext", self.iter, XDMFFile.Encoding.HDF5, append=True)

        if trafo:
            try:
                ALE.move(self.mesh, up, annotate=False)
            except:
                ALE.move(self.mesh, up)

        return u


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
