from dolfin import *
from dolfin_adjoint import *
from pathlib import Path

from optimization_custom.ipopt_solver import IPOPTSolver_, IPOPTProblem_
from optimization_custom.preprocessing import Preprocessing

from NeuralNet.neural_network_custom import ANN, generate_weights
import numpy as np
from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy
import scipy.sparse as sps

import matplotlib.pyplot as plt

from NeuralNet.tools import *

import moola

class Custom_Reduced_Functional(object):
    def __init__(self, posfunc, posfunc_der, net, normgradtraf, b_opt, init_weights, threshold, fb):
        self.posfunc = posfunc
        self.posfunc_der = posfunc_der
        self.net = net
        self.normgradtraf = normgradtraf
        self.b_opt = b_opt
        self.threshold = threshold

        J = assemble((LearnExt.NN_der(0.05, self.normgradtraf, net) - 1.0 - fb(self.b_opt)) ** 2 * dx)
        Jhat = ReducedFunctional(J, net.weights_ctrls())
        self.Jhat = Jhat
        self.controls = Enlist(net.weights_ctrls())
        self.ctrls = weights_to_list(init_weights)
        self.init_weights = init_weights

    def eval(self, x):
        x = list_to_weights(x, self.init_weights)
        y = trafo_weights(x, self.posfunc)
        return self.Jhat(y)

    def __call__(self, values):
        values = Enlist(values)
        if len(values) != len(self.controls):
            raise ValueError("values should be a list of same length as controls.")
        self.ctrls = values
        # for i, value in enumerate(values):
        #    self.controls[i].update(value)
        val = self.eval(values)
        print('eval', val)
        return val

    def derivative(self):
        # print(list_to_array(self.controls))
        x = list_to_weights(self.ctrls, self.init_weights)
        y = trafo_weights(x, self.posfunc)
        self.net.set_weights(y)
        # self.Jhat(y)
        Jhat_der = self.Jhat.derivative()
        djx = trafo_weights_chainrule(Jhat_der, x, self.posfunc_der)
        return self.controls.delist(djx)  # djx

    def first_order_test(self, init_weights):
        x0 = weights_to_list(init_weights)
        ds = weights_to_list(init_weights)

        print(list_to_array(x0))

        j0 = self.__call__(x0)
        djx = self.derivative()  # x0)

        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001, 0.0000001]
        xlist = [weights_list_add(x0, ds, eps) for eps in epslist]
        jlist = [self.__call__(x) for x in xlist]

        ds_ = list_to_array(ds)
        djx_ = list_to_array(djx)

        self.perform_first_order_check(jlist, j0, djx_, ds_, epslist)

    def perform_first_order_check(self, jlist, j0, gradj0, ds, epslist):
        # j0: function value at x0
        # gradj0: gradient value at x0
        # epslist: list of decreasing eps-values
        # jlist: list of function values at x0+eps*ds for all eps in epslist
        diff0 = []
        diff1 = []
        order0 = []
        order1 = []
        i = 0
        for eps in epslist:
            je = jlist[i]
            di0 = je - j0
            di1 = je - j0 - eps * np.dot(gradj0, ds)
            diff0.append(abs(di0))
            diff1.append(abs(di1))
            if i == 0:
                order0.append(0.0)
                order1.append(0.0)
            if i > 0:
                order0.append(np.log(diff0[i - 1] / diff0[i]) / np.log(epslist[i - 1] / epslist[i]))
                order1.append(np.log(diff1[i - 1] / diff1[i]) / np.log(epslist[i - 1] / epslist[i]))
            i = i + 1
        for i in range(len(epslist)):
            print('eps\t', epslist[i], '\t\t check continuity\t', order0[i], '\t\t diff0 \t', diff0[i],
                  '\t\t check derivative \t', order1[i], '\t\t diff1 \t', diff1[i], '\n'),
        return

def smoothmax(r, eps=1e-4):
    return conditional(gt(r, eps), r - eps / 2, conditional(lt(r, 0), 0, r ** 2 / (2 * eps)))

class LearnExt:
    def __init__(self, mesh, boundaries, params, output_path, order):
        """
        :param mesh:
        :param boundaries:
        :param params:
        :param output_path:
        :param order: polynomial degree of FEM function
        """
        #net2 = ANN("../example/learned_networks/trained_network.pkl")

        self.mesh = mesh
        self.boundaries = boundaries
        self.params = params
        self.output_path = output_path
        self.order = 1# order

        self.Vs = FunctionSpace(mesh, "CG", order)
        self.V = VectorFunctionSpace(mesh, "CG", order)
        #self.V1 = VectorFunctionSpace(mesh, "CG", 1)

        self.normgradtraf = None
        self.b_opt = None

        self.threshold = None

        self.fb = lambda x: x

    @staticmethod
    def NN_der(eta, s, net):
        return 1.0 + smoothmax(s - eta) * net(s)

    def learn(self, deformation, threshold):
        self.optimal_control(deformation)
        self.machine_learning(threshold)
        self.visualize(deformation, threshold)

    def optimal_control(self, deformation):
        # interpolate
        deformation = interpolate(deformation, self.V)

        # boundary conditions
        bc = []
        bc2 = []
        zero = Constant(("0.0", "0.0"))
        zeros = Constant(0.)
        for i in self.params["def_boundary_parts"]:
            bc.append(DirichletBC(self.V, deformation, self.boundaries, self.params[i]))
        for i in self.params["zero_boundary_parts"]:
            bc.append(DirichletBC(self.V, zero, self.boundaries, self.params[i]))
            bc2.append(DirichletBC(self.Vs, zeros, self.boundaries, self.params[i]))

        set_working_tape(Tape())

        # function spaces
        alpha = interpolate(Constant(0.05), self.Vs)

        b = TrialFunction(self.Vs)
        vb = TestFunction(self.Vs)

        u = TestFunction(self.V)
        v = TestFunction(self.V)

        # alpha -> b
        E1 = inner(grad(b), grad(vb)) * dx - inner(alpha, vb) * dx
        b = Function(self.Vs)
        solve(lhs(E1) == rhs(E1), b, bc2)

        # b -> u
        u = Function(self.V)
        E = inner((1.0 + self.fb(b)) * grad(u), grad(v)) * dx
        solve(E == 0, u, bc)

        # reduced objective
        eta = 1.0
        Fhat = Identity(2) + grad(u)
        Fhati = inv(Fhat)
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
        J = assemble(pow((1.0 / (det(Identity(2) + grad(u))) + det(Identity(2) + grad(u))), 2) * ds(2)
                     + inner(grad(det(Identity(2) + grad(u))), grad(det(Identity(2) + grad(u)))) * ds(2)
                     + inner(grad(det(Identity(2) + grad(u))), grad(det(Identity(2) + grad(u)))) * dx
                     )
        control = Control(alpha)

        rf = ReducedFunctional(J, control)

        test = False
        if test == True:
            h = interpolate(Expression("100*x[0]*x[1]", degree=1), alpha.function_space())
            taylor_test(rf, alpha, h)
            breakpoint()

        # optimization
        # save initial
        ufile = File(self.output_path + "/displacement.pvd")
        afile = File(self.output_path + "/alpha_opt.pvd")
        up = project(u, self.V)
        upi = project(-1.0 * u, self.V)
        ALE.move(self.mesh, up, annotate=False)
        ufile << up
        afile << alpha
        ALE.move(self.mesh, upi, annotate=False)

        av = TrialFunction(self.Vs)
        aw = TestFunction(self.Vs)
        A = assemble((1e-3*inner(av, aw) + inner(grad(av), grad(aw))) * dx)

        reg = 1e-2
        preprocessing = Preprocessing(self.Vs)
        problem = IPOPTProblem_([rf], [1], [None], [None], [None], preprocessing, sps.csc_matrix(A.array()), reg)
        ipopt = IPOPTSolver_(problem)
        alpha_opt = preprocessing.dof_to_control(ipopt.solve(alpha.vector()[:]))

        b_opt = Function(self.Vs)
        E1 = inner(grad(b_opt), grad(vb)) * dx - inner(alpha_opt, vb) * dx
        solve(E1 == 0, b_opt, bc2)

        # solve u_opt
        E = inner((1.0 + self.fb(b_opt)) * grad(u), grad(v)) * dx
        solve(E == 0, u, bc)

        up = project(u, self.V)
        upi = project(-u, self.V)
        ALE.move(self.mesh, up, annotate=False)
        ufile << up
        afile << b_opt
        ALE.move(self.mesh, upi, annotate=False)

        # breakpoint()
        output_directory = self.output_path
        normgradtraf = project(inner(grad(u), grad(u)), self.Vs)

        xdmf = XDMFFile(output_directory + "optimal_control_data.xdmf")

        xdmf.write_checkpoint(b_opt, "b_opt", 0, append=True)
        xdmf.write_checkpoint(normgradtraf, "normgradtraf", 0, append=True)

        # save normgradtraf
        self.normgradtraf = normgradtraf
        self.b_opt = b_opt

    def machine_learning(self, threshold=0):
        self.threshold = threshold
        if self.b_opt == None or self.normgradtraf == None:
            b_opt = Function(Vs)
            normgradtraf = Function(Vs)

            # load data
            with XDMFFile(output_directory + "optimal_control_data.xdmf") as infile:
                infile.read_checkpoint(b_opt, "b_opt")
                infile.read_checkpoint(normgradtraf, "normgradtraf")

            file = File('../Output/learnExt/results/bopt.pvd')
            file << b_opt
        else:
            b_opt = self.b_opt
            normgradtraf = self.normgradtraf

        set_working_tape(Tape())

        # neural net for coefficient
        layers = [1, 10, 1]
        bias = [True, True]
        x, y = SpatialCoordinate(self.mesh)
        net = ANN(layers, bias=bias, mesh=mesh)  # , init_method="fixed")

        parameters["form_compiler"]["quadrature_degree"] = 4

        # transform net.weights
        init_weights = generate_weights(layers, bias=bias)

        posfunc = lambda x: x ** 2
        posfunc_der = lambda x: 2 * x

        rf = Custom_Reduced_Functional(posfunc, posfunc_der, net, normgradtraf, b_opt, init_weights, threshold, self.fb)
        rfn = ReducedFunctionalNumPy(rf)

        opt_theta = minimize(rfn, options={"disp": True, "gtol": 1e-12, "ftol": 1e-12,
                                           "maxiter": 100})  # minimize(Jhat, method= "L-BFGS-B") #

        transformed_opt_theta = trafo_weights(list_to_weights(opt_theta, init_weights), posfunc)
        net.set_weights(transformed_opt_theta)

        # net save
        net.save(self.output_path + "/trained_network.pkl")
        self.net = net
        pass

    def visualize(self, deformation, threshold=None):
        if threshold == None and self.threshold != None:
            threshold = self.threshold
        elif self.threshold == None:
            threshold = 0

        if self.net == None:
            net = ANN(self.output_path + "/trained_network.pkl")
            breakpoint()
        else:
            net = self.net

        if self.b_opt == None:
            b_opt = Function(Vs)
            # load data
            with XDMFFile(output_directory + "/optimal_control_data.xdmf") as infile:
                infile.read_checkpoint(b_opt, "b_opt")
        else:
            b_opt = self.b_opt

        # boundary deformation
        g = deformation
        zero = Constant(("0.0", "0.0"))
        # boundary conditions
        bc = []
        for i in self.params["def_boundary_parts"]:
            bc.append(DirichletBC(self.V, g, self.boundaries, self.params[i]))
        for i in self.params["zero_boundary_parts"]:
            bc.append(DirichletBC(self.V, zero, self.boundaries, self.params[i]))

        ufile = File(self.output_path + "/comparison_ml_vs_harmonic.pvd")

        # harmonic extension
        u = Function(self.V)
        v = TestFunction(self.V)
        E = inner(grad(u), grad(v)) * dx
        solve(E == 0, u, bc)

        up = project(u, self.V)
        upi = project(-1.0 * u, self.V)
        ALE.move(self.mesh, up, annotate=False)
        ufile << up
        ALE.move(self.mesh, upi, annotate=False)

        # opt control extension
        u1 = Function(self.V)
        v = TestFunction(self.V)
        E = inner((1.0 + self.fb(b_opt)) * grad(u1), grad(v))*dx
        solve(E == 0, u1, bc)

        up = project(u1, self.V)
        upi = project(-1.0 * u1, self.V)
        ALE.move(self.mesh, up, annotate=False)
        ufile << up
        ALE.move(self.mesh, upi, annotate=False)

        # learned extension
        E = inner(self.NN_der(threshold, inner(grad(u), grad(u)), net)
                  * grad(u), grad(v)) * dx
        solve(E == 0, u, bc)

        up = project(u, self.V)
        upi = project(-1.0 * u, self.V)
        ALE.move(self.mesh, up, annotate=False)
        ufile << up
        ALE.move(self.mesh, upi, annotate=False)

        pass

