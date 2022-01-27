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
    def __init__(self, mesh, boundaries, domains, param, FSI_params, extension_operator):
        super().__init__(FSI_params)
        self.mesh = mesh
        self.boundaries = boundaries
        self.domains = domains
        self.param = param
        self.FSI_params = FSI_params
        self.extension_operator = extension_operator

        self.theta = 0.5 + self.FSI_params["deltat"] # theta-time-stepping parameter

        # help variables
        self.aphat = 1e-9

        self.savedir = FSI_params["save_directory"]
        self.N = FSI_params["save_every_N_snapshot"]

        self.displacement_filename = self.savedir + "/displacementy.txt"
        self.displacement = []

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

    def save_snapshot(self, vp, u):
        if self.savedir == None:
            pass
        elif self.N == 0:
            print("N has to be larger than 0, continue without saving snapshots...")
        else:
            if abs(self.t/(self.dt * self.N) - round(self.t/(self.dt * self.N))) < 1e-10:
                print('save snapshot...', self.t)

                # save displacement
                u.rename("displacement", "displacement")
                self.dfile << u

                # save velocity and pressure
                (v, p) = vp.split(deepcopy=True)
                ui = Function(u.function_space())
                ui.vector()[:] = -1.0 * u.vector()[:]
                pmed = assemble(p * self.dxf)
                vol = assemble(Constant("1.0") * self.dxf)
                ALE.move(self.mesh, u, annotate=False)
                pp = project(p - pmed / vol * Constant("1.0"), p.function_space())
                v.rename("velocity", "velocity")
                p.rename("pressure", "pressure")
                self.pfile << p
                self.vfile << v

                ALE.move(self.mesh, ui, annotate=False)

                # save characteristic function of solid mesh
                Vs = VectorFunctionSpace(self.FSI_params["solid_mesh"], "CG", 2)
                c = interpolate(Constant(1.0), FunctionSpace(self.FSI_params["solid_mesh"], "CG", 1))
                c.rename("charfunc", "charfunc")
                us = transfer_to_subfunc(u, Vs)
                usi = Function(us.function_space())
                usi.vector()[:] = -1.0 * us.vector()[:]
                ALE.move(self.FSI_params["solid_mesh"], us, annotate=False)
                self.cfile << c
                ALE.move(self.FSI_params["solid_mesh"], usi, annotate=False)


    def save_displacement(self, u):
        try:
            self.displacement.append(u(self.FSI_params["displacement_point"])[1])
            np.savetxt(self.displacement_filename, self.displacement)
        except:
            print('Displacement can not be saved. Does FSI_params contain displacement_point?'
                  ' Does folder exist where .txt-file should be saved to?')

    def get_deformation(self, vp, vp_, u_):
        u = Function(u_.function_space())
        (v_, p_) = vp_.split(deepcopy=True)
        (v, p) = vp.split(deepcopy=True)
        u.vector()[:] = u_.vector()[:] + self.dt*(self.theta*v_.vector()[:] + (1-self.theta)*v.vector()[:])
        # 0 displacement on outer fluid boundary
        bc = DirichletBC(u.function_space(), Constant((0.0,0.0)), 'on_boundary')
        bc.apply(u.vector())
        ##
        fluid_domain = self.FSI_params["fluid_mesh"]
        Vbf = VectorFunctionSpace(fluid_domain, "CG", 2)
        boundary_def = transfer_to_subfunc(u, Vbf)

        unew = self.extension_operator.extend(boundary_def)
        u = transfer_subfunction_to_parent(unew, u)
        return u

    def solve_system(self, vp_, u, u_, option):
        vp = Function(vp_.function_space())
        vp.vector()[:] = vp_.vector()[:]
        psi = TestFunction(vp_.function_space())

        bc = self.get_boundary_conditions(vp_.function_space())
        F = self.get_weak_form(vp, vp_, u, u_, psi, option)

        solve(F== 0, vp, bc, solver_parameters={"nonlinear_solver": "newton", "newton_solver":
            {"maximum_iterations": 20}})

        return vp

    def get_boundary_conditions(self, VP):
        """
        :param VP: function space in which velocity and pressure live
        :return:
        """
        bc = []
        bc.append(DirichletBC(VP.sub(0), self.bc, self.boundaries, self.param["inflow"]))

        for i in self.param["no_slip_ids"]:
            bc.append(DirichletBC(VP.sub(0), Constant((0.0,0.0)), self.boundaries, self.param[i]))

        return bc


    def get_weak_form(self, vp, vp_, u, u_, psi, option):
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
        (v, p) = split(vp)
        (v_, p_) = split(vp_)
        (psiv, psip) = split(psi)

        # variables for variational form
        I = Identity(2)

        if option == 0:
            Fhat = I + grad(u_ + k*(theta*v + (1-theta)*v_))
        elif option == 1:
            Fhat = I + grad(u)

        Fhatt = Fhat.T
        Fhati = inv(Fhat)
        Fhatti = Fhati.T
        Ehat = 0.5 * (Fhatt * Fhat - I)
        Jhat = det(Fhat)

        if option == 0:
            sFhat = Fhat
            sFhatt = Fhatt
            sFhati = Fhati
            sFhatti = Fhatti
            sEhat = Ehat
            sJhat = Jhat
        elif option == 1:
            sFhat = I + grad(u_ + k*(theta * v + (1 - theta)*v_))
            sFhatt = sFhat.T
            sFhati = inv(sFhat)
            sFhatti = sFhati.T
            sEhat = 0.5 * (sFhatt * sFhat - I)
            sJhat = det(sFhat)

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
               )

        if option == 1:
            # this term vanishes in the fully Lagrangian setting
            A_T += -1.0/k * inner(rhof * Jhat * grad(v) * Fhati * (u - u_), psiv)*dxf

        # pressure terms
        A_P = inner(Jhat * Fhati * sigmafp, grad(psiv).T) * dxf


        # implicit terms (e.g. incompressibility)
        A_I = (inner(tr(grad(Jhat * Fhati * v).T), psip) * dxf
               + inner(self.aphat * grad(p),grad(psip)) * dxs
               )

        # remaining explicit terms
        A_E = (inner(Jhat * Fhati * sigmafv, grad(psiv).T) * dxf
               + inner(Jhat * Fhati * sigmasv, grad(psiv).T) * dxs
               )

        if option == 1:
            # this term vanishes in the fully Lagrangian setting
            A_E += inner(rhof * Jhat * grad(v) * Fhati * v, psiv) * dxf

        # explicit terms of previous time-step
        A_E_rhs = (inner(Jhat_ * Fhati_ * sigmafv_, grad(psiv).T) * dxf
                   + inner(Jhat_ * Fhati_ * sigmasv_, grad(psiv).T) * dxs
                   )

        if option == 1:
            # this term vanishes in the fully Lagrangian setting
            A_E += inner(rhof * Jhat_ * grad(v_) * Fhati_ * v_, psiv) * dxf

        F = A_T + A_P + A_I + theta * A_E + (1 - theta)*A_E_rhs

        return F




class FSIsolver(Solver):
    def __init__(self, mesh, boundaries, domains, param, FSI_params, extension_operator):
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
        self.extension_operator = extension_operator

        # function space
        V2 = VectorElement("CG", mesh.ufl_cell(), 2)
        S1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        self.VP = FunctionSpace(mesh, MixedElement(V2, S1))

        self.U = VectorFunctionSpace(mesh, "CG", 2)

        # FSI
        self.FSI = FSI(self.mesh, self.boundaries, self.domains, self.param, self.FSI_params, self.extension_operator)

    def solve(self):
        # velocity and pressure
        vp = Function(self.VP)
        vp_fl = Function(self.VP)  # fully Lagrangian
        vp_ = Function(self.VP)    # previous time-step

        # deformation
        u = Function(self.U)
        u_ = Function(self.U)      # previous time-step

        while not self.FSI.check_termination():
            self.FSI.save_snapshot(vp, u)
            self.FSI.save_displacement(u)
            self.FSI.advance_time()

            u_.assign(u)
            vp_.assign(vp)
            vp.assign(self.FSI.solve_system(vp_, u, u_, 0))
            u.assign(self.FSI.get_deformation(vp, vp_, u_))
            vp.assign(self.FSI.solve_system(vp_, u, u_, 1))




