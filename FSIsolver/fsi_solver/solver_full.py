import sympy as sym
from dolfin import *
import numpy as np

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent) + "/tools")

from tools import transfer_to_subfunc, transfer_subfunction_to_parent

class Solver(object):
    def __init__(self, mesh, boundaries, domains):
        self.mesh = mesh
        self.boundaries = boundaries
        self.domains = domains

    def solve(self):
        """solve PDE"""
        raise NotImplementedError

    def save_snapshot(self):
        """save snapshot"""
        raise NotImplementedError

class Context(object):
    def __init__(self, params):
        """
        :param params: contains deltat, t and T, boundary_cond
        """
        self.t = params["t"]
        self.T = params["T"]
        self.dt = params["deltat"]
        self.bc = params["boundary_cond"]

    def check_termination(self):
        return (not self.t <= self.T)

    def advance_time(self):
        """
        update of time variables and boundary conditions
        """
        self.t += self.dt
        self.bc.t = self.t


class FSI(Context):
    def __init__(self, mesh, boundaries, domains, param, FSI_params):
        super().__init__(FSI_params)
        self.mesh = mesh
        self.boundaries = boundaries
        self.domains = domains
        self.param = param
        self.FSI_params = FSI_params

        self.theta = 0.5 + self.FSI_params["deltat"] # theta-time-stepping parameter

        # help variables
        self.aphat = 1e-9
        self.auhat = 1e-9

        self.savedir = FSI_params["save_directory"]
        self.N = FSI_params["save_every_N_snapshot"]

        self.displacement_filename = self.savedir + "/displacementy.txt"
        self.determinant_filename = self.savedir + "/determinant.txt"
        self.displacement = []
        self.determinant_deformation = []

        if not self.savedir == None:
            velocity_filename = self.savedir + "/velocity.pvd"
            charfunc_filename = self.savedir + "/char.pvd"
            pressure_filename = self.savedir + "/pressure.pvd"
            deformation_filename = self.savedir + "/displacement.pvd"

            self.vfile = File(velocity_filename)
            self.cfile = File(charfunc_filename)
            self.pfile = File(pressure_filename)
            self.dfile = File(deformation_filename)

        dx = Measure("dx", domain=self.mesh, subdomain_data=self.domains)

        self.dxf = dx(self.param["fluid"])
        self.dxs = dx(self.param["solid"])

        # define projectors
        # initialize projectors
        class Projector():
            def __init__(self, V):
                self.v = TestFunction(V)
                u = TrialFunction(V)
                form = inner(u, self.v)*dx
                self.A = assemble(form)
                self.solver = LUSolver(self.A)
                self.uh = Function(V)
            def project(self, f):
                L = inner(f, self.v)*dx
                b = assemble(L)
                self.solver.solve(self.uh.vector(), b)
                return self.uh

        self.projectorP = Projector(FunctionSpace(mesh, "CG", 1))
        self.projectorV = Projector(VectorFunctionSpace(mesh, "CG", 1))
        self.projectorV0 = Projector(FunctionSpace(mesh, "DG", 0))
        self.projectorU = Projector(VectorFunctionSpace(mesh, "CG", 2))

    def save_snapshot(self, vpu):
        if self.savedir == None:
            pass
        elif self.N == 0:
            print("N has to be larger than 0, continue without saving snapshots...")
        else:
            if abs(self.t/(self.dt * self.N) - round(self.t/(self.dt * self.N))) < 1e-10:
                print('save snapshot...', self.t)

                # save velocity and pressure
                (v, p, u) = vpu.split(deepcopy=True)
                u = self.projectorU.project(u) 
                # save displacement
                u.rename("displacement", "displacement")
                self.dfile << u
                ui = Function(u.function_space())
                ui.vector()[:] = -1.0 * u.vector()[:]
                pmed = assemble(p * self.dxf)
                vol = assemble(Constant("1.0") * self.dxf)
                try:
                    ALE.move(self.mesh, u, annotate=False)
                except:
                    ALE.move(self.mesh, u)
                pp = self.projectorP.project(p- pmed/vol*Constant(1.0)) 
                v.rename("velocity", "velocity")
                p.rename("pressure", "pressure")
                self.pfile << p
                self.vfile << v
                try:
                    ALE.move(self.mesh, ui, annotate=False)
                except:
                    ALE.move(self.mesh, ui)

                # save characteristic function of solid mesh
                Vs = VectorFunctionSpace(self.FSI_params["solid_mesh"], "CG", 2)
                c = interpolate(Constant(1.0), FunctionSpace(self.FSI_params["solid_mesh"], "CG", 1))
                c.rename("charfunc", "charfunc")
                us = transfer_to_subfunc(u, Vs)
                usi = Function(us.function_space())
                usi.vector()[:] = -1.0 * us.vector()[:]
                try:
                    ALE.move(self.FSI_params["solid_mesh"], us, annotate=False)
                except:
                    ALE.move(self.FSI_params["solid_mesh"], us)
                self.cfile << c
                try:
                    ALE.move(self.FSI_params["solid_mesh"], usi, annotate=False)
                except:
                    ALE.move(self.FSI_params["solid_mesh"], usi)


    def save_displacement(self, vpu, save_det=False):
        (v, p, u) = vpu.split(deepcopy=True)
        try:
            self.displacement.append(u(self.FSI_params["displacement_point"])[1])
            np.savetxt(self.displacement_filename, self.displacement)
        except:
            print('Displacement can not be saved. Does FSI_params contain displacement_point?'
                  ' Does folder exist where .txt-file should be saved to?')
        try:
            if save_det == True:
                det_u = self.projectorP.project(det(Identity(2) + grad(u)))
                self.determinant_deformation.append(det_u.vector().min())
                np.savetxt(self.determinant_filename, self.determinant_deformation)
        except:
            print('Maximum determinant value can not be saved.')


    def solve_system(self, vpu, vpu_):
        if not hasattr(self, 'nvs_solver0'):
            print('compile self.nvs_solver')
            vpu.vector()[:] = vpu_.vector()[:]
            psi = TestFunction(vpu_.function_space())

            bc = self.get_boundary_conditions(vpu_.function_space())
            F = self.get_weak_form(vpu, vpu_, psi)
            Jac = derivative(F, vpu)
            nv_problem = NonlinearVariationalProblem(F, vpu, bc, J=Jac)
            self.nvs_solver = NonlinearVariationalSolver(nv_problem)
            solver_parameters = {"nonlinear_solver": "newton", "newton_solver": {"maximum_iterations": 10}}
            self.nvs_solver.parameters.update(solver_parameters)
            print('solve nonlinear system')
            self.nvs_solver.solve()
        else:
            print('solve nonlinear system')
            self.nvs_solver.solve()

        return vpu

    def get_boundary_conditions(self, VPU):
        """
        :param VP: function space in which velocity and pressure and deformation live
        :return:
        """
        bc = []
        bc.append(DirichletBC(VPU.sub(0), self.bc, self.boundaries, self.param["inflow"]))
        bc.append(DirichletBC(VPU.sub(2), Constant((0.0,0.0)), self.boundaries, self.param["inflow"]))
        bc.append(DirichletBC(VPU.sub(2), Constant((0.0, 0.0)), self.boundaries, self.param["outflow"]))

        for i in self.param["no_slip_ids"]:
            bc.append(DirichletBC(VPU.sub(0), Constant((0.0,0.0)), self.boundaries, self.param[i]))
            bc.append(DirichletBC(VPU.sub(2), Constant((0.0,0.0)), self.boundaries, self.param[i]))

        return bc


    def get_weak_form(self, vpu, vpu_, psi):
        # 0 to solve system 1, 1 to solve system 3
        k = self.dt
        theta = self.theta
        lambdas = self.FSI_params["lambdas"]
        mys = self.FSI_params["mys"]
        rhof = self.FSI_params["rhof"]
        rhos = self.FSI_params["rhos"]
        nyf = self.FSI_params["nyf"]

        dx = Measure("dx", domain=self.mesh, subdomain_data=self.domains)

        dxf = self.dxf
        dxs = self.dxs

        # split functions
        (v, p, u) = split(vpu)
        (v_, p_, u_) = split(vpu_)
        (psiv, psip, psiu) = split(psi)

        # variables for variational form
        I = Identity(2)

        Fhat = I + grad(u)

        Fhatt = Fhat.T
        Fhati = inv(Fhat)
        Fhatti = Fhati.T
        Ehat = 0.5 * (Fhatt * Fhat - I)
        Jhat = det(Fhat)

        sFhat = Fhat
        sFhatt = Fhatt
        sFhati = Fhati
        sFhatti = Fhatti
        sEhat = Ehat
        sJhat = Jhat

        # stress tensors
        sigmafp = -p * I
        sigmafv = rhof * nyf * (grad(v) * Fhati + Fhatti *grad(v).T)
        sigmasv = inv(sJhat) * sFhat * (lambdas * tr(sEhat) * I + 2.0 * mys * sEhat) * sFhatt # STVK

        # variables for previous time-step
        Fhat_ = I + grad(u_)
        Fhatt_ = Fhat_.T
        Fhati_ = inv(Fhat_)
        Fhatti_ = Fhati_.T
        Ehat_ = 0.5 *(Fhatt_ * Fhat_ - I)
        Jhat_ = det(Fhat_)
        Jhattheta = theta * Jhat + (1.0 - theta) * Jhat_

        sigmafv_ = rhof * nyf * (grad(v_) * Fhati_ + Fhatti_ * grad(v_).T)
        sigmasv_ = inv(Jhat_) * Fhat_ * (lambdas * tr(Ehat_) * I + 2.0 * mys * Ehat_) * Fhatt_ # STVK

        # weak form

        # terms with time derivative
        A_T = (1.0/k * inner(rhof * Jhattheta * (v - v_), psiv) * dxf
               + 1.0/k * inner(rhos * (v - v_), psiv) * dxs
               + 1.0 / k * inner(rhos * (u - u_), psiu) * dxs
               )

        A_T += -1.0/k * inner(rhof * Jhat * grad(v) * Fhati * (u - u_), psiv)*dxf

        # pressure terms
        A_P = inner(Jhat * Fhati * sigmafp, grad(psiv).T) * dxf


        # implicit terms (e.g. incompressibility)
        A_I = (inner(tr(grad(Jhat * Fhati * v).T), psip) * dxf
               + inner(self.aphat * grad(p), grad(psip)) * dxs
               + inner(self.auhat * grad(u).T, grad(psiu).T) * dxf
               )

        # remaining explicit terms
        A_E = (inner(Jhat * Fhati * sigmafv, grad(psiv).T) * dxf
               + inner(Jhat * Fhati * sigmasv, grad(psiv).T) * dxs
               - inner(rhos * v, psiu) * dxs
               )

        # this term vanishes in the fully Lagrangian setting
        A_E += inner(rhof * Jhat * grad(v) * Fhati * v, psiv) * dxf

        # explicit terms of previous time-step
        A_E_rhs = (inner(Jhat_ * Fhati_ * sigmafv_, grad(psiv).T) * dxf
                   + inner(Jhat_ * Fhati_ * sigmasv_, grad(psiv).T) * dxs
                   - inner(rhos * v_, psiu) * dxs
                   )

        # this term vanishes in the fully Lagrangian setting
        A_E_rhs += inner(rhof * Jhat_ * grad(v_) * Fhati_ * v_, psiv) * dxf

        F = A_T + A_P + A_I + theta * A_E + (1 - theta)*A_E_rhs

        return F




class FSIsolver(Solver):
    def __init__(self, mesh, boundaries, domains, param, FSI_params):
        """
        solves the FSI system on mesh
        :param mesh: computational mesh (with fluid and solid part)
        :param boundaries: MeshFunction that contains boundary information
        :param domains: MeshFunction that contains subdomain information
        :param param: contains id's for subdomains and boundary parts
        :param FSI_params: contains FSI parameters
        lambdas, mys, rhos, rhof, nyf
        also contains the information for the time-stepping
        deltat, t (start time), and T (end time)
        and initial and boundary conditions
        initial_cond, boundary_cond
        :param extension_operator: object of the ExtensionOperator-class
        """
        super().__init__(mesh, boundaries, domains)
        self.param = param
        self.FSI_params = FSI_params

        output_directory = FSI_params["save_directory"]
        self.xdmf = XDMFFile(output_directory + "/deformation.xdmf")

        # function space
        V2 = VectorElement("CG", mesh.ufl_cell(), 2)
        S1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        self.VPU = FunctionSpace(mesh, MixedElement(V2, S1, V2))

        # FSI
        self.FSI = FSI(self.mesh, self.boundaries, self.domains, self.param, self.FSI_params)

    def solve(self):
        # velocity and pressure
        vpu = Function(self.VPU)
        vpu_ = Function(self.VPU)    # previous time-step

        while not self.FSI.check_termination():
            self.FSI.save_snapshot(vpu)
            self.FSI.save_displacement(vpu, save_det=True)
            self.FSI.advance_time()

            # save u_ as xdmf, in order to be able to 'learn' extension from here
            (v, p, u) = vpu.split(deepcopy=True)
            u2 = self.FSI.projectorU.project(u)
            self.xdmf.write_checkpoint(u2, "u", 0, append=True)

            vpu_.assign(vpu)
            vpu.assign(self.FSI.solve_system(vpu, vpu_))





