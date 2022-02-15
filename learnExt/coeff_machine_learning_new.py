from fenics import *
from dolfin_adjoint import *
from NeuralNet.neural_network_custom import ANN, generate_weights
import numpy as np
from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

import matplotlib.pyplot as plt
import NeuralNet.tools as tools


class Custom_Reduced_Functional(object):
    def __init__(self, posfunc, posfunc_der, net, normgradtraf, alpha_opt, init_weights, threshold):
        self.posfunc = posfunc
        self.posfunc_der = posfunc_der
        self.net = net
        self.normgradtraf = normgradtraf
        self.alpha_opt = alpha_opt
        self.threshold = threshold

        J = assemble((NN_der(0.05, self.normgradtraf, net)- exp(self.alpha_opt)) ** 2 * dx)
        Jhat = ReducedFunctional(J, net.weights_ctrls())
        self.Jhat = Jhat
        self.controls = Enlist(net.weights_ctrls())
        self.ctrls = tools.weights_to_list(init_weights)
        self.init_weights = init_weights

    def eval(self, x):
        x = tools.list_to_weights(x, self.init_weights)
        y = tools.trafo_weights(x, self.posfunc)
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
        # print(tools.list_to_array(self.controls))
        x = tools.list_to_weights(self.ctrls, self.init_weights)
        y = tools.trafo_weights(x, self.posfunc)
        self.net.set_weights(y)
        # self.Jhat(y)
        Jhat_der = self.Jhat.derivative()
        djx = tools.trafo_weights_chainrule(Jhat_der, x, self.posfunc_der)
        return self.controls.delist(djx)  # djx

    def first_order_test(self, init_weights):
        x0 = tools.weights_to_list(init_weights)
        ds = tools.weights_to_list(init_weights)

        print(tools.list_to_array(x0))

        j0 = self.__call__(x0)
        djx = self.derivative()  # x0)

        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001, 0.0000001]
        xlist = [tools.weights_list_add(x0, ds, eps) for eps in epslist]
        jlist = [self.__call__(x) for x in xlist]

        ds_ = tools.list_to_array(ds)
        djx_ = tools.list_to_array(djx)

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

def compute_machine_learning_new(mesh, V, Vs, params, boundaries, output_directory, threshold):
    alpha_opt = Function(Vs)
    normgradtraf = Function(Vs)

    # load data
    with XDMFFile(output_directory + "optimal_control_data.xdmf") as infile:
        infile.read_checkpoint(alpha_opt, "alpha_opt")
        infile.read_checkpoint(normgradtraf, "normgradtraf")

    set_working_tape(Tape())

    # neural net for coefficient
    layers = [1, 10, 1]
    bias = [True, True]
    x, y = SpatialCoordinate(mesh)
    net = ANN(layers, bias=bias, mesh=mesh)  # , init_method="fixed")

    parameters["form_compiler"]["quadrature_degree"] = 4

    # transform net.weights
    init_weights = generate_weights(layers, bias=bias)

    posfunc = lambda x: x ** 2
    posfunc_der = lambda x: 2 * x

    rf = Custom_Reduced_Functional(posfunc, posfunc_der, net, normgradtraf, alpha_opt, init_weights, threshold)
    rfn = ReducedFunctionalNumPy(rf)

    opt_theta = minimize(rfn, options={"disp": True, "gtol": 1e-12, "ftol": 1e-12,
                                       "maxiter": 100})  # minimize(Jhat, method= "L-BFGS-B") #

    transformed_opt_theta = tools.trafo_weights(tools.list_to_weights(opt_theta, init_weights), posfunc)
    net.set_weights(transformed_opt_theta)

    # net save
    net.save(output_directory + "trained_network.pkl")

def smoothmax(r, eps=1e-4):
    return conditional(gt(r, eps), r - eps / 2, conditional(lt(r, 0), 0, r ** 2 / (2 * eps)))

def NN_der(eta, s, net):
    return 1.0 + smoothmax(s-eta)*net(s)

def visualize(mesh, V, Vs, params, deformation, def_boundary_parts,
              zero_boundary_parts, boundaries, output_directory, threshold):
    net = ANN(output_directory + "trained_network.pkl")
    alpha_opt = Function(Vs)

    # load data
    with XDMFFile(output_directory + "optimal_control_data.xdmf") as infile:
        infile.read_checkpoint(alpha_opt, "alpha_opt")

        # boundary deformation
        g = deformation
        zero = Constant(("0.0", "0.0"))
        # boundary conditions
        bc = []
        for i in def_boundary_parts:
            bc.append(DirichletBC(V, g, boundaries, params[i]))
        for i in zero_boundary_parts:
            bc.append(DirichletBC(V, zero, boundaries, params[i]))

        ufile = File(output_directory + "comparison_ml_vs_harmonic.pvd")

        u = Function(V)
        v = TestFunction(V)
        E = inner(NN_der(threshold, inner(grad(u), grad(u)), net)
                  * grad(u), grad(v)) * dx(mesh)
        solve(E == 0, u, bc)

        up = project(u, V)
        ALE.move(mesh, up, annotate=False)
        ufile << up

        # comparison to harmonic extension
        upi = project(-1.0 * u, V)
        ALE.move(mesh, upi, annotate=False)
        u = Function(V)
        v = TestFunction(V)
        E = inner(grad(u), grad(v)) * dx(mesh)
        solve(E == 0, u, bc)

        up = project(u, V)
        ALE.move(mesh, up, annotate=False)
        ufile << up