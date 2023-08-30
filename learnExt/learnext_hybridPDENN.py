from dolfin import *
from dolfin_adjoint import *
from pathlib import Path

from .optimization_custom.ipopt_solver import IPOPTSolver_, IPOPTProblem_
from .optimization_custom.preprocessing import Preprocessing

from .NeuralNet.neural_network_custom import ANN, generate_weights
import numpy as np
from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy
import scipy.sparse as sps

import matplotlib.pyplot as plt

from .NeuralNet.tools import *

#import moola

class Custom_Reduced_Functional(object):
    def __init__(self, posfunc, posfunc_der, net, init_weights, threshold,
                 mesh, boundaries, data, params, order, output_path):
        self.posfunc = posfunc
        self.posfunc_der = posfunc_der
        self.net = net
        self.threshold = threshold
        self.ufile = File(output_path + "/custom_function_checks.pvd")

        V = VectorFunctionSpace(mesh, "CG", order)

        g = []
        uref = []
        try:
            output_flag = data["output"] != None
        except:
            output_flag = False

        # input data
        for i in range(len(data["input"])):
            g.append(data["input"][i])
            if output_flag:
                uref.append(data["output"][i])

        # objective
        J = 0
        # boundary deformation
        zero = Constant(("0.0", "0.0"))

        for j in range(len(g)):
            # boundary conditions
            bc = []
            gj = project(g[j], V, annotate=False)
            for i in params["def_boundary_parts"]:
                bc.append(DirichletBC(V, gj, boundaries, params[i]))
            for i in params["zero_boundary_parts"]:
                bc.append(DirichletBC(V, zero, boundaries, params[i]))

            # learned extension
            u = Function(V)
            v = TestFunction(V)
            E = inner(self.NN_der(threshold, inner(grad(u), grad(u)), net) * grad(u), grad(v)) * dx
            solve(E == 0, u, bc, solver_parameters={"newton_solver":{"relative_tolerance":1e-8}})

            self.ufile << u
            self.ufile << gj

            ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

            if output_flag:
                urefj = project(uref[j], u.function_space(), annotate=False)
                J += assemble(inner(u - urefj, u - urefj)*dx + inner(grad(u - urefj), grad(u - urefj))*dx)
                #              + inner(u - urefj, u - urefj)*ds)
            else:
                J += assemble(pow((1.0 / (det(Identity(2) + grad(u))) + det(Identity(2) + grad(u))), 2) * ds(2)
                             + inner(grad(det(Identity(2) + grad(u))), grad(det(Identity(2) + grad(u)))) * ds(2)
                             + pow((1.0 / (det(Identity(2) + grad(u))) + det(Identity(2) + grad(u))), 2) * dx
                             + inner(grad(det(Identity(2) + grad(u))), grad(det(Identity(2) + grad(u)))) * dx
                             )
        Jhat = ReducedFunctional(J, net.weights_ctrls())
        self.Jhat = Jhat
        self.controls = Enlist(net.weights_ctrls())
        self.ctrls = weights_to_list(init_weights)
        self.init_weights = init_weights

    def eval(self, x):
        x = list_to_weights(x, self.init_weights)
        y = trafo_weights(x, self.posfunc)
        z = list_to_weights(y, self.init_weights)
        self.net.weights = z
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

    @staticmethod
    def NN_der(eta, s, net):
        if eta != 0: # and net.plot_antider == False:
            return 1.0 + smoothmax(s - eta) * net(s)
        else:
            breakpoint()
            return net(s)

    def first_order_test(self, init_weights):
        init_weights = self.net.weights
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

        self.threshold = None

        self.fb = lambda x: x

        self.net_coeff = interpolate(Constant(0.0), self.Vs)

        self.ufile = File(self.output_path + "/comparison_ml_vs_harmonic_ref.pvd")


    def learn(self, data, threshold):
        self.machine_learning(data, threshold)
        self.visualize(data, threshold)

    def machine_learning(self, data, threshold=0):
        self.threshold = threshold

        set_working_tape(Tape())

        # neural net for coefficient
        layers = [1, 5, 5, 1]
        bias = [True, True, False]
        x, y = SpatialCoordinate(self.mesh)
        net = ANN(layers, bias=bias, mesh=mesh) #, init_method="fixed")

        parameters["form_compiler"]["quadrature_degree"] = 4

        # transform net.weights
        init_weights = generate_weights(layers, bias=bias)

        posfunc = lambda x: x**2 + 0.001*x**0 #x ** 2 #abs(x) #x ** 2
        posfunc2 = lambda x: x**2
        posfunc_der = lambda x: 2*x #2 * x # np.ones(len(x)) * (x >= 0) - np.ones(len(x))*(x < 0) #2 * x

        init_weights1 = trafo_weights(init_weights, posfunc)
        init_weights = list_to_weights(init_weights1, init_weights)
        net.weights = init_weights

        rf = Custom_Reduced_Functional(posfunc, posfunc_der, net, init_weights, threshold,
                                       self.mesh, self.boundaries, data, self.params, self.order, self.output_path)

        try:
            rf = Custom_Reduced_Functional(posfunc, posfunc_der, net, init_weights, threshold,
                                       self.mesh, self.boundaries, data, self.params, self.order, self.output_path)
        except Exception as e:
            print(e)
            breakpoint()

        taylortest = False
        if taylortest:
            rf.first_order_test(init_weights)
            exit(0)
        rfn = ReducedFunctionalNumPy(rf)

        opt_theta = minimize(rfn, method= "L-BFGS-B", options={"disp": True, "gtol": 1e-8, "ftol": 1e-8,
                                           "maxiter": 100})  # minimize(Jhat, method= "L-BFGS-B") #

        transformed_opt_theta = trafo_weights(list_to_weights(opt_theta, init_weights), posfunc)
        net.set_weights(transformed_opt_theta)

        # net save
        net.save(self.output_path + "/trained_network.pkl")
        self.net = net
        pass

    def visualize(self, data, threshold=None):
        if threshold == None and self.threshold != None:
            threshold = self.threshold
        elif self.threshold == None:
            threshold = 0

        if self.net == None:
            net = ANN(self.output_path + "/trained_network.pkl")
        else:
            net = self.net

        # boundary deformation
        for j in range(len(data["input"])):
            g = data["input"][j]
            zero = Constant(("0.0", "0.0"))
            # boundary conditions
            bc = []
            for i in self.params["def_boundary_parts"]:
                bc.append(DirichletBC(self.V, g, self.boundaries, self.params[i]))
            for i in self.params["zero_boundary_parts"]:
                bc.append(DirichletBC(self.V, zero, self.boundaries, self.params[i]))

            ufile = self.ufile

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


            # learned extension
            u = Function(self.V)
            cro = Custom_Reduced_Functional
            E = inner(cro.NN_der(threshold, inner(grad(u), grad(u)), net) * grad(u), grad(v)) * dx
            solve(E == 0, u, bc, solver_parameters={"nonlinear_solver": "newton", "newton_solver":
                {"maximum_iterations": 200}})

            up = project(u, self.V)
            upi = project(-1.0 * u, self.V)
            ALE.move(self.mesh, up, annotate=False)
            ufile << up
            ALE.move(self.mesh, upi, annotate=False)

            # reference - if available
            try:
                output_flag = data["output"] != None
            except:
                output_flag = False
            if output_flag:
                ui = project(-1.0 * data["output"][j], self.V, annotate=False)
                ALE.move(self.mesh, data["output"][j], annotate=False)
                ufile << data["output"][j]
                ALE.move(self.mesh, ui, annotate=False)

        pass

