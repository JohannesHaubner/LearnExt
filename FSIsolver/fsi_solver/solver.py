import sympy as sym
from dolfin import *
import numpy as np

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent) + "/tools")

from tools import Tools

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
    
class SNESProblem():
    def __init__(self, F, u, bc):
        V = u.function_space()
        du = TrialFunction(V)
        self.L = F
        self.a = derivative(F, u, du)
        self.bcs = bc
        self.u = u

    def F(self, snes, x, F):
        x = PETScVector(x)
        F  = PETScVector(F)
        assemble(self.L, tensor=F)
        for bc in self.bcs:
            bc.apply(F, x)                

    def J(self, snes, x, J, P):
        J = PETScMatrix(J)
        assemble(self.a, tensor=J)
        for bc in self.bcs:
            bc.apply(J)

class Context(object):
    def __init__(self, params):
        """
        :param params: contains deltat, t and T, boundary_cond
        """
        self.t = params["t"]
        self.T = params["T"]
        self.dt = params["deltat"]
        self.dt_max = params["deltat"]
        self.dt_min = 1/128 * self.dt_max
        self.bc = params["boundary_cond"]
        self.success = False

    def check_termination(self):
        return (not self.t <= self.T)

    def choose_t(self):
        if abs(self.t/0.04 - np.round(self.t /0.04))*0.04 < self.dt_min/2:
            self.t += self.dt
        elif np.floor((self.t + self.dt)/0.04) > np.floor(self.t/0.04):
            t_new = np.floor((self.t + self.dt)/0.04)*0.04
            self.dt = float(t_new - self.t) #without float self.dt is a numpy.float which causes problems
            self.t = t_new
        else:
            self.t += self.dt

    def advance_time(self):
        """
        update of time variables and boundary conditions
        """
        self.choose_t()
        self.bc.t = self.t
        print('update time: t = ', self.t)

    def check_timestep_success(self):
        print('check_timestep_sucess', self.dt, self.success)
        return (self.dt < 2/3*self.dt_min) or self.success == True

    def adapt_dt(self):
        if self.success:
            if self.dt*2 <= self.dt_max:
                print("adapt dt ", self.dt, self.dt * 2)
            self.dt = min(self.dt*2, self.dt_max)
        else:
            print("adapt dt ", self.dt, self.dt * 1 / 2)
            self.t = self.t - self.dt
            self.dt = self.dt * 1 / 2

    def reset_success(self):
        self.success = False

    def timestep_success(self):
        print('timestep successful')
        self.success = True


class FSI(Context):
    def __init__(self, mesh, boundaries, domains, param, FSI_params, extension_operator):
        super().__init__(FSI_params)
        self.mesh = mesh
        self.boundaries = boundaries
        self.domains = domains
        self.param = param
        self.FSI_params = FSI_params
        self.extension_operator = extension_operator

        if "material_model" in self.FSI_params.keys():
            if self.FSI_params["material_model"] == "IMR":
                self.theta = 1.0
            else:
                self.theta = 0.5 + self.FSI_params["deltat"] # theta-time-stepping parameter
        else:
            self.theta = 0.5 + self.FSI_params["deltat"] # theta-time-stepping parameter

        if "bc_type" in self.FSI_params.keys():
            bc_type = self.FSI_params["bc_type"]
            print("Choice for boundary type: ", self.FSI_params["bc_type"])
        else:
            print("Default choice for boundary type: inflow")
            bc_type = "inflow"

        self.bc_type = bc_type

        # help variables
        self.aphat = 1e-9

        self.savedir = FSI_params["save_directory"]
        #self.N = FSI_params["save_every_N_snapshot"]
        try:
            self.save_snapshot_on = FSI_params["save_snapshot_on"]
        except:
            self.save_snapshot_on = False
        try:
            self.save_data_on = FSI_params["save_data_on"]
        except:
            self.save_data_on = False
        try:
            self.save_states_on = FSI_params["save_states_on"]
        except:
            self.save_states_on = False

        if self.savedir is not None and self.save_data_on:
            self.displacement_filename = self.savedir + "/displacementy.txt"
            self.determinant_filename = self.savedir + "/determinant.txt"
            self.times_filename = self.savedir + "/times.txt"
            self.displacement = []
            self.determinant_deformation = []
            self.times = []

        # files for warmstart
        if self.savedir is not None and self.save_states_on:
            output_directory = self.FSI_params["save_directory"]
            self.xdmf_states = XDMFFile(output_directory + "/warmstart/states.xdmf")
            self.xdmf_load = XDMFFile(output_directory[:-2] + "/states.xdmf")
            self.time_save = str(output_directory + "/warmstart/t.npy")
            self.time_load = str(output_directory[:-2] + "/t.npy")

        if not self.savedir == None and self.save_states_on:
            velocity_filename = self.savedir + "/velocity.pvd"
            charfunc_filename = self.savedir + "/char.pvd"
            pressure_filename = self.savedir + "/pressure.pvd"
            deformation_filename = self.savedir + "/displacement.pvd"

            self.vfile = File(velocity_filename)
            self.cfile = File(charfunc_filename)
            self.pfile = File(pressure_filename)
            self.dfile = File(deformation_filename)

        dx = Measure("dx", domain=self.mesh, subdomain_data=self.domains)
        ds = Measure("ds", domain=self.mesh, subdomain_data=boundaries)

        self.dxf = dx(self.param["fluid"])
        self.dxs = dx(self.param["solid"])
        self.ds = ds

        V = VectorFunctionSpace(self.mesh, "CG", 2)
        Vs = VectorFunctionSpace(self.FSI_params["solid_mesh"], "CG", 2)
        Vf = VectorFunctionSpace(self.FSI_params["fluid_mesh"], "CG", 2)
        self.tools_solid = Tools(V, Vs)
        self.tools_fluid = Tools(V, Vf)

        self.bc_weak_form = []

        self.snes = PETSc.SNES().create(MPI.comm_world) 
        opts = PETSc.Options()
        opts.setValue('snes_monitor', None)
        #opts.setValue('ksp_view', None)
        #opts.setValue('pc_view', None)
        #opts.setValue('log_view', None)
        opts.setValue('snes_type', 'newtonls')
        #opts.setValue('snes_view', None)
        opts.setValue('snes_divergence_tolerance', 1e2)
        opts.setValue('snes_linesearch_type', 'l2')
        self.snes.setFromOptions()

        self.snes.setErrorIfNotConverged(True)


        class Projector():
            def __init__(self, V):
                self.v = TestFunction(V)
                u = TrialFunction(V)
                form = inner(u, self.v)*dx
                self.A = assemble(form)
                self.solver = LUSolver(self.A)
                self.V = V
            
            def project(self, f):
                L = inner(f, self.v)*dx
                b = assemble(L)
                
                uh = Function(self.V)
                self.solver.solve(uh.vector(), b)
                
                return uh
        
        self.projector_scalar_dg0 = Projector(FunctionSpace(self.mesh, "DG", 0))
        self.projector_scalar_cg1 = Projector(FunctionSpace(self.mesh, "CG", 1))
        self.projector_vector_cg1 = Projector(VectorFunctionSpace(self.mesh, "CG", 1))


    def warmstart(self, t):
        self.warmstart = True
        self.t = t
        self.bc.t = t

    def save_states(self, u, u_, vp, vp_):
        v, p = vp.split(deepcopy=True)
        v_, p_ = vp_.split(deepcopy=True)
        self.xdmf_states.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, append=True)
        self.xdmf_states.write_checkpoint(u_, "u_", 0, XDMFFile.Encoding.HDF5, append=True)
        self.xdmf_states.write_checkpoint(v, "v", 0, XDMFFile.Encoding.HDF5, append=True)
        self.xdmf_states.write_checkpoint(v_, "v_", 0, XDMFFile.Encoding.HDF5, append=True)
        self.xdmf_states.write_checkpoint(p, "p", 0, XDMFFile.Encoding.HDF5, append=True)
        self.xdmf_states.write_checkpoint(p_, "p_", 0, XDMFFile.Encoding.HDF5, append=True)
        np.save(self.time_save, self.t)
        return u, u_, vp, vp_

    def load_states(self, u, u_, vp, vp_):
        v, p = vp.split(deepcopy=True)
        v_, p_ = vp_.split(deepcopy=True)
        self.xdmf_load.read_checkpoint(u, "u")
        self.xdmf_load.read_checkpoint(u_, "u_")
        self.xdmf_load.read_checkpoint(v, "v")
        self.xdmf_load.read_checkpoint(v_, "v_")
        self.xdmf_load.read_checkpoint(p, "p")
        self.xdmf_load.read_checkpoint(p_, "p_")
        assign(vp.sub(0), v)
        assign(vp.sub(1), p)
        assign(vp_.sub(0), v_)
        assign(vp_.sub(1), p_)
        t = np.load(self.time_load)
        self.warmstart(t)
        ui = project(-1.0*u, u.function_space())
        ALE.move(self.mesh, u, annotate=False)
        file = File(self.savedir + '/test.pvd')
        file << vp
        file << v
        ALE.move(self.mesh, ui, annotate=False)
        return u, u_, vp, vp_

    def save_snapshot(self, vp, u):
        if self.savedir == None:
            pass
        else:
            print('Check if snapshot is saved for t = ', self.t)
            t_frac = abs(self.t / 0.04 - round(self.t / 0.04))*0.04  # make a snapshot every 1/25 s
            if abs(t_frac) < self.dt_min * 0.5: # or True:
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
                try:
                    ALE.move(self.mesh, u, annotate=False)
                except:
                    ALE.move(self.mesh, u)
                pp = self.projector_scalar_cg1.project(p - pmed / vol * Constant("1.0"))
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
                us = self.tools_solid.transfer_to_subfunc(u)
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
                print("snapshot saved")
            else:
                print('Snapshot is not stored for t = ', self.t,
                      ' since it only taken close to every multiple of 1/25s.')


    def save_displacement(self, u, save_det=False):
        try:
            self.displacement.append(u(self.FSI_params["displacement_point"])[1])
            np.savetxt(self.displacement_filename, self.displacement)
        except:
            print('Displacement can not be saved. Does FSI_params contain displacement_point?'
                  ' Does folder exist where .txt-file should be saved to?')
        self.times.append(float(self.t))
        np.savetxt(self.times_filename, self.times)
        try:
            if save_det == True:
                up = self.projector_vector_cg1.project(u)
                det_u = self.projector_scalar_dg0.project(det(Identity(2) + grad(up)))
                self.determinant_deformation.append(det_u.vector().min())
                np.savetxt(self.determinant_filename, self.determinant_deformation)
        except:
            print('Maximum determinant value can not be saved.')


    def get_deformation(self, vp, vp_, u_, b_old=None):
        u = Function(u_.function_space())
        (v_, p_) = vp_.split(deepcopy=True)
        (v, p) = vp.split(deepcopy=True)
        u.vector()[:] = u_.vector()[:] + self.dt*((1-self.theta)*v_.vector()[:] + self.theta*v.vector()[:])
        # 0 displacement on outer fluid boundary
        bc = DirichletBC(u.function_space(), Constant((0.0,0.0)), 'on_boundary')
        bc.apply(u.vector())
        ##
        fluid_domain = self.FSI_params["fluid_mesh"]
        Vbf = VectorFunctionSpace(fluid_domain, "CG", 2)
        boundary_def = self.tools_fluid.transfer_to_subfunc(u)

        params = {}
        params["b_old"] = b_old
        params["t"] = self.t
        try:
            params["displacementy"] = self.displacement[-1]
        except:
            pass

        unew = self.extension_operator.extend(boundary_def, params)
        u = self.tools_fluid.transfer_subfunction_to_parent(unew, u)
        return u

    def solve_system(self, vp_, u, u_, option):
        vp = Function(vp_.function_space())
        vp.vector()[:] = vp_.vector()[:]
        psi = TestFunction(vp_.function_space())

        bc = self.get_boundary_conditions(vp_.function_space())

        F = self.get_weak_form(vp, vp_, u, u_, psi, option)

        ## see https://fenicsproject.discourse.group/t/using-petsc4py-petsc-snes-directly/2368/12
                    
        problem = SNESProblem(F, vp, bc)
            
        b = PETScVector()  # same as b = PETSc.Vec()
        J_mat = PETScMatrix()   

        ksp = self.snes.getKSP()
        ksp.getPC().setType('lu')
        ksp.getPC().setFactorSolverType('mumps')
        ksp.setType('preonly')

        self.snes.setFunction(problem.F, b.vec())
        self.snes.setJacobian(problem.J, J_mat.mat())
        self.snes.solve(None, problem.u.vector().vec())

        #if snes.converged == False:
        #    raise Exception("ERROR: SNES solver not converged")

        
        #problem = NonlinearVariationalProblem(F, vp, bc, J)
        #solver = NonlinearVariationalSolver(problem)
        #prm = solver.parameters
        #prm['nonlinear_solver'] = 'snes'

        #info(solver.parameters, True)
        #from IPython import embed; embed()
        #solver.solve()
        #exit(0)

        #solve(F == 0, vp, bc, solver_parameters={"nonlinear_solver": "newton", "newton_solver":
        #    {"maximum_iterations": 20}})
        
        return vp

    def get_boundary_conditions(self, VP):
        """
        :param VP: function space in which velocity and pressure live
        :return:
        """

        # pressure BC
        class PressureB(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], (0.1)) and near(x[1], (0.05))
            
        pressureb = PressureB()

        bc = []
        if self.bc_type == "inflow":
            bc.append(DirichletBC(VP.sub(1), Constant(0.0), pressureb, method='pointwise'))
            bc.append(DirichletBC(VP.sub(0), self.bc, self.boundaries, self.param["inflow"]))
        elif self.bc_type == "pressure":
            pass
        else:
            raise("Not Implemented")

        for i in self.param["no_slip_ids"]:
            bc.append(DirichletBC(VP.sub(0), Constant((0.0,0.0)), self.boundaries, self.param[i]))

        return bc


    def get_weak_form(self, vp, vp_, u, u_, psi, option):
        # 0 to solve system 1, 1 to solve system 3
        k = self.dt
        theta = Constant(self.theta)
        lambdas = self.FSI_params["lambdas"]
        mys = self.FSI_params["mys"]
        rhof = self.FSI_params["rhof"]
        rhos = self.FSI_params["rhos"]
        nyf = self.FSI_params["nyf"]

        dx = Measure("dx", domain=self.mesh, subdomain_data=self.domains)

        dxf = self.dxf
        dxs = self.dxs
        ds = self.ds
        
        n = FacetNormal(self.mesh)

        # split functions
        (v, p) = split(vp)
        (v_, p_) = split(vp_)
        (psiv, psip) = split(psi)

        # variables for variational form
        I = Identity(2)

        if option == 0:
            Fhat = I + grad(u_ + k*(theta*v + (Constant(1.)-theta)*v_))
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
            sFhat = I + grad(u_ + k*(theta * v + (Constant(1.) - theta)*v_))
            sFhatt = sFhat.T
            sFhati = inv(sFhat)
            sFhatti = sFhati.T
            sEhat = 0.5 * (sFhatt * sFhat - I)
            sJhat = det(sFhat)

        # variables for previous time-step
        Fhat_ = I + grad(u_)
        Fhatt_ = Fhat_.T
        Fhati_ = inv(Fhat_)
        Fhatti_ = Fhati_.T
        Ehat_ = 0.5 *(Fhatt_ * Fhat_ - I)
        Jhat_ = det(Fhat_)
        Jhattheta = theta * Jhat + (1.0 - theta) * Jhat_

        if option == 0:
            sFhat_ = Fhat_
            sFhatt_ = Fhatt_
            sFhati_ = Fhati_
            sFhatti_ = Fhatti_
            sEhat_ = Ehat_
            sJhat_ = Jhat_
        elif option == 1:
            sFhat_ = I + grad(u_)
            sFhatt_ = sFhat_.T
            sFhati_ = inv(sFhat_)
            sFhatti_ = sFhati_.T
            sEhat_ = 0.5 * (sFhatt_ * sFhat_ - I)
            sJhat_ = det(sFhat_)

        # stress tensors
        sigmafp = -p * I
        sigmafv = rhof * nyf * (grad(v) * Fhati + Fhatti *grad(v).T)
        sigmafv_ = rhof * nyf * (grad(v_) * Fhati_ + Fhatti_ * grad(v_).T)

        if "material_model" in self.FSI_params.keys():
            material = self.FSI_params["material_model"]
            print("Choice for material model: ", self.FSI_params["material_model"])
        else:
            print("Default choice for material model: STVK")
            material = "STVK"
  
        if material == "STVK":
            sigmasv = inv(sJhat) * sFhat * (lambdas * tr(sEhat) * I + 2.0 * mys * sEhat) * sFhatt # STVK
            sigmasv_ = inv(sJhat_) * sFhat_ * (lambdas * tr(sEhat_) * I + 2.0 * mys * sEhat_) * sFhatt_ # STVK
            sigmasp = Constant(0.0)
            imr = Constant(0.0)
        elif material == "IMR": 
            sigmasv = mys * sFhat * sFhatt - lambdas * sFhatti * sFhati
            sigmasv_ = mys * sFhat_ * sFhatt_ - lambdas * sFhatti_ * sFhati_ 
            sigmasp =  Constant(-1.0) * p * I 
            imr = Constant(1.0)
        else:
            print('Material not defined yet.')

        # weak form

        # terms with time derivative
        A_T = (1.0/k * inner(rhof * Jhattheta * (v - v_), psiv) * dxf
               + 1.0/k * inner(rhos * (v - v_), psiv) * dxs
               )

        if option == 1:
            # this term vanishes in the fully Lagrangian setting
            A_T += -1.0/k * inner(rhof * Jhat * grad(v) * Fhati * (u - u_), psiv)*dxf

        # pressure terms
        A_P = inner(Jhat * Fhati * sigmafp, grad(psiv).T) * dxf + imr*inner(sJhat*sFhati*sigmasp, grad(psiv)) * dxs

        # implicit terms (e.g. incompressibility)
        A_I = (inner(tr(grad(Jhat * Fhati * v).T), psip) * dxf
               + (Constant(1.0) - imr ) *inner(self.aphat * grad(p),grad(psip)) * dxs
               + imr * inner(sJhat - Constant(1.0), psip) * dxs
               )

        # remaining explicit terms
        A_E = (inner(Jhat * Fhati * sigmafv, grad(psiv).T) * dxf
               + inner(sJhat * sFhati * sigmasv, grad(psiv).T) * dxs
               )

        if option == 1:
            # this term vanishes in the fully Lagrangian setting
            A_E += inner(rhof * Jhat * grad(v) * Fhati * v, psiv) * dxf

        # explicit terms of previous time-step
        A_E_rhs = (inner(Jhat_ * Fhati_ * sigmafv_, grad(psiv).T) * dxf
                   + inner(sJhat_ * sFhati_ * sigmasv_, grad(psiv).T) * dxs
                   )

        if option == 1:
            # this term vanishes in the fully Lagrangian setting
            A_E_rhs += inner(rhof * Jhat_ * grad(v_) * Fhati_ * v_, psiv) * dxf

        F = A_T + A_P + A_I + theta * A_E + (1 - theta)*A_E_rhs

        # add boundary conditions that appear in weak form (get_boundary_conditions)
        if self.bc_type == "pressure":
            F += inner(self.bc* n, psiv)*ds(self.param["inflow"])
            F -= rhof * nyf *inner(grad(v).T*n, psiv)*ds(self.param["inflow"])
            F -= rhof * nyf *inner(grad(v).T*n, psiv)*ds(self.param["outflow"])



        return F




class FSIsolver(Solver):
    def __init__(self, mesh, boundaries, domains, param, FSI_params, extension_operator, warmstart=False):
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

        if warmstart == True:
            self.ws_path = self.FSI_params["save_directory"]
            self.FSI_params["save_directory"] += "/warmstarted"

        self.warmstart = warmstart

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
        vp_ = Function(self.VP)    # previous time-step

        # deformation
        u = Function(self.U)
        u_ = Function(self.U)      # previous time-step

        #vp__ = Function(self.VP)
        #u__ = Function(self.U)

        if self.warmstart:
            self.FSI.displacement = np.loadtxt(self.ws_path + "/displacementy.txt").tolist()[:-1]
            self.FSI.determinant_deformation = np.loadtxt(self.ws_path + "/determinant.txt").tolist()[:-1]
            self.FSI.times = np.loadtxt(self.ws_path + "/times.txt").tolist()[:-1]
            u, u_, vp, vp_ = self.FSI.load_states(u, u_, vp, vp_)
            zero = interpolate(Constant((0., 0., 0.)), vp.function_space())
            u.assign(self.FSI.get_deformation(zero, zero, u, b_old=u_))
            #file_test = File(self.ws_path + '/test_u.pvd')
            #file_test << u
            vp.assign(self.FSI.solve_system(vp_, u, u_, 1))

        while not self.FSI.check_termination():
            if self.FSI.savedir is not None:
                if self.FSI.save_snapshot_on:
                    self.FSI.save_snapshot(vp, u)
                if self.FSI.save_data_on:
                    self.FSI.save_displacement(u, save_det=True)
                if self.FSI.save_states_on:
                    self.FSI.save_states(u, u_, vp, vp_)

            #u__.assign(u_)
            #vp__.assign(vp_)
            u_.assign(u)
            vp_.assign(vp)

            while not self.FSI.check_timestep_success():
                self.FSI.advance_time()
                print(self.FSI.t, self.FSI.dt)
                try:
                    vp.assign(self.FSI.solve_system(vp_, u, u_, 0))   #u = u_ here in this system
                    u.assign(self.FSI.get_deformation(vp, vp_, u_))
                    vp.assign(self.FSI.solve_system(vp_, u, u_, 1))
                    self.FSI.timestep_success()
                    self.FSI.adapt_dt()
                except Exception as e:
                    print(e)
                    self.FSI.adapt_dt()
                    flag = self.extension_operator.custom(self.FSI)
                    if flag == True:
                        u_.assign(self.FSI.get_deformation(vp, vp_, u_))

            if self.FSI.success == False:
                raise ValueError('System not solvable with minimal time-step size.')
            else:
                self.FSI.reset_success()

