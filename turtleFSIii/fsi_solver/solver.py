import sympy as sym
from dolfin import *

class Solver(object):
    def __init__(self, mesh):
        self.mesh = mesh

    def solve(self):
        """solve PDE"""
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
        return (not self.t < self.T)

    def advance_time(self):
        """
        update of time variables and boundary conditions
        """
        self.t += self.dt
        self.bc.t = self.t

class FSI(Context):
    def __init__(self, mesh, param, FSI_params, option):
        super().__init__(FSI_params)
        self.init = FSI_params["initial_cond"]
        self.mesh = mesh
        self.param = param
        self.FSI_params = FSI_params
        self.option #0 to solve system 1, 1 to solve system 3

        self.theta = 0.5 + self.FSI_params["deltat"] # theta-time-stepping parameter

        # help variables
        self.aphat = 1e-9

    def weak_form(self, vp, vp_, u_, psi, option):
        k = self.dt
        theta = self.theta
        lambdas = self.FSI_params["lambdas"]
        mys = self.FSI_params["mys"]
        rhof = self.FSI_params["rhof"]
        rhos = self.FSI_params["rhos"]
        nyf = self.FSI_params["nyf"]

        dxf = dx(mesh)(self.param["fluid"])
        dxs = dx(mesh)(self.param["solid"])

        # split functions
        (v, p) = split(vp)
        (v_, p_) = split(vp_)
        (psiv, psip) = split(psi)

        # variables for variational form
        I = Identity(2)
        Fhat = I + grad(u_ + k*(theta*v + (1-theta)*v_))
        Fhatt = Fhat.T
        Fhati = inv(Fhat)
        Fhatti = Fhati.T
        Ehat = 0.5 * (Fhatt * Fhat - I)
        Jhat = det(Fhat)

        # stress tensors
        sigmafp = -p * I
        sigmafv = rhof * nyf * (grad(v) * Fhati + Fhatti *grad(v).T)
        sigmasv = inv(Jhat) * Fhat * (lambdas * tr(Ehat) * I + 2.0 * mys * Ehat) * Fhatt # STVK

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
    def __init__(self, mesh, param, FSI_params, extension_operator):
        """
        solves the FSI system on mesh
        :param mesh: computational mesh (with fluid and solid part)
        :param param: contains id's for subdomains and boundary parts
        :param FSI_params: contains FSI parameters
        lambdas, mys, rhos, rhof, nyf
        also contains the information for the time-stepping
        deltat, t (start time), and T (end time)
        and initial and boundary conditions
        initial_cond, boundary_cond
        :param extension_operator: object of the ExtensionOperator-class
        """
        super().__init__(mesh)
        self.param = param
        self.FSI_params = FSI_params
        self.extension_operator = extension_operator

    def solve(self):
        print('here')



