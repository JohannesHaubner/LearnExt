from dolfin import *
from dolfin_adjoint import *
from copy import deepcopy
import numpy as np
import backend

from pyadjoint.optimization.optimization_solver import OptimizationSolver
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import diags


import cyipopt

import matplotlib.pyplot as plt

class IPOPTProblem_:
    def __init__(self, Jhat, scaling_Jhat, constraints, scaling_constraints, bounds,
                 preprocessing, inner_product_matrix, reg):
        """
        Jhat:                   list of different objective function summands
        scaling_Jhat:           list of floats with same length as Jhat that specifies how the sum of the different
                                objective function summands is taken
        constraints:            list of constraints
        scaling_constraints:    list with same length as constraints that specifies scaling of constraints
        bounds:                 list with same length as constraints, each element being a tuple of a lower and upper
                                bound
        preprocessing:          contains .dofs_to_control and its chainrule .dofs_to_control_chainrule
        inner_product_matrix:   inner product that Ipopt should use, discrete hack via sparse Cholesky
        reg:                    regularization parameter
        """
        self.Jhat = [ReducedFunctionalNumPy(Jhat[i]) for i in range(len(Jhat)) ]
        self.scaling_Jhat = scaling_Jhat
        self.constraints = [ReducedFunctionalNumPy(constraints[i]) for i in range(len(constraints))]
        self.scaling_constraints = scaling_constraints
        self.preprocessing = preprocessing
        self.inner_product_matrix = inner_product_matrix
        self.trafo_matrix = self.sparse_cholesky(inner_product_matrix)
        self.reg = reg
        self.bounds = bounds

    @staticmethod
    def sparse_cholesky(A):
        # input matrix A sparse symmetric positive-definite
        # https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d
        n = A.shape[0]
        LU = splu(A, permc_spec='NATURAL', diag_pivot_thresh=0)  # sparse LU decomposition

        if (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all():  # check the matrix A is positive definite
            return LU.L.dot(diags(LU.U.diagonal() ** 0.5)).transpose()
        else:
            sys.exit('The matrix is not positive definite')

    def transformation(self, x):
        """
        x --> (L^T)^(-1)*x
        """
        return spsolve(self.trafo_matrix, x)

    def transformation_chainrule(self, djy):
        """
        djy --> (L^T)^(-T)*djy
        """
        return spsolve(self.trafo_matrix.transpose(), djy)

    def initial_point_trafo(self, x0):
        """
        needs to be done since we apply the affine linear transformation afterwards
        """
        return self.trafo_matrix.dot(x0)


class IPOPTSolver_(OptimizationSolver):
    def __init__(self, problem, parameters=None):
        try:
            import cyipopt
        except ImportError:
            print("You need to install cyipopt. (It is recommended to install IPOPT with HSL support!)")
            raise
        self.problem = problem
        self.problem_obj = self.create_problem_obj(self)


        print('Initialization of IPOPTSolver finished')

    def create_problem_obj(self, outer):
        return IPOPTSolver_.opt_prob(outer)

    def test_objective(self, k):
        # check dof_to_deformation with first order derivative check
        print('Extension.test_dof_to_deformation started.......................')
        xl = k
        x0 = -0.5 * np.ones(xl) # 0.5 * np.ones(xl)
        ds = 1.0 * np.ones(xl)
        x0 = self.problem.initial_point_trafo(x0)
        ds = self.problem.initial_point_trafo(ds)
        # ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.problem_obj.objective(x0)
        djx = self.problem_obj.gradient(x0)
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 1e-5]
        jlist = [self.problem_obj.objective(x0 + eps * ds) for eps in epslist]
        order1, diff1 = self.perform_first_order_check(jlist, j0, djx, ds, epslist)
        return order1, diff1

    def test_constraints(self, k, ind, option=0):
        # check dof_to_deformation with first order derivative check
        print('Extension.test_dof_to_deformation started.......................')
        xl = k
        x0 = 0.24 * np.ones(xl)
        j0 = self.problem_obj.constraints(x0)[ind]
        djx = self.problem_obj.jacobian(x0)
        djx = djx[range(k*ind, k*(ind+1), 1)]
        print('j0', j0, 'djx', djx)
        if option == 1:
            ds = 1.0 * np.ones(xl)
            # ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
            epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 1e-5]
            jlist = [self.problem_obj.constraints(x0 + eps * ds)[ind] for eps in epslist]
            order1, diff1 = self.perform_first_order_check(jlist, j0, djx, ds, epslist)
        return order1, diff1

    @staticmethod
    def perform_first_order_check(jlist, j0, gradj0, ds, epslist):
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

        return order1, diff1


    class opt_prob(object):
        def __init__(self, outer):
            self.problem = outer.problem
            self.x = np.ones(self.problem.trafo_matrix.shape[0])
            self.y = self.problem.transformation(self.x)

        def trafo(self, x):
            if (abs(self.x -x) > 1e-14).any():
                print('recompute transformation')
                self.x = x
                self.y = self.problem.transformation(x)
            return self.y

        def objective(self, x):
            #
            # The callback for calculating the objective
            #
            # x to deformation
            print('evaluate objective')
            tx = self.trafo(x)
            rho = self.problem.preprocessing.dof_to_control(tx)
            j = 0
            for i in range(len(self.problem.Jhat)):
                j += self.problem.scaling_Jhat[i]*self.problem.Jhat[i](rho.vector()[:])
            j += 0.5 * self.problem.reg * np.dot(np.asarray(x), np.asarray(x))       # regularization
            return j

        def gradient(self, x):
            #
            # The callback for calculating the gradient
            #
            # print('evaluate derivative of objective funtion')
            print('evaluate gradient')
            tx = self.trafo(x)
            rho = self.problem.preprocessing.dof_to_control(tx)
            # savety feature:
            if (max(abs(self.problem.Jhat[0].get_controls() - rho.vector()[:])) > 1e-12):
                print('update control')
                [self.problem.Jhat[i](rho.vector()[:]) for i in range(len(self.problem.Jhat))]
            dJf = np.zeros(len(rho.vector()[:]))
            for i in range(len(self.problem.Jhat)):
                dJf += self.problem.scaling_Jhat[i]*self.problem.Jhat[i].derivative(forget=False, project=False)
            dJ = self.problem.preprocessing.dof_to_control_chainrule(dJf, 2)
            dJ = self.problem.transformation_chainrule(dJ)
            dJ += self.problem.reg * x
            return np.asarray(dJ, dtype=float)

        def constraints(self, x):
            #
            # The callback for calculating the constraints
            print('evaluate constraint')
            xt = self.trafo(x)
            rho = self.problem.preprocessing.dof_to_control(xt)

            constraints = []
            for i in range(len(self.problem.constraints)):
                constraints.append(self.problem.scaling_constraints[i]
                                   *self.problem.constraints[i](rho.vector()[:])
                                   )
            return np.array(constraints)

        def jacobian(self, x):
            #
            # The callback for calculating the Jacobian
            #
            print('evaluate jacobian')
            xt = self.trafo(x)
            rho = self.problem.preprocessing.dof_to_control(xt)

            dconstraints = []

            # savety feature:
            if (max(abs(self.problem.constraints[0].get_controls() - rho.vector()[:])) > 1e-12):
                print('update control')
                [self.problem.constraints[i](rho.vector()[:]) for i in range(len(self.problem.constraints))]

            for i in range(len(self.problem.constraints)):
                di = self.problem.constraints[i].derivative()
                di = self.problem.preprocessing.dof_to_control_chainrule(di,2)
                di = self.problem.transformation_chainrule(di)
                dconstraints.append(self.problem.scaling_constraints[i] *di)
            return np.asarray(np.concatenate([di for di in dconstraints]))

        def intermediate(
                self,
                alg_mod,
                iter_count,
                obj_value,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials
        ):

            #
            # Example for the use of the intermediate callback.
            #
            print("Objective value at iteration ", iter_count, " is ", obj_value)
            return


    def solve(self, x0):
        max_float = np.finfo(np.double).max
        min_float = np.finfo(np.double).min

        cl = []
        cu = []
        for i in range(len(self.problem.bounds)):
            if self.problem.bounds[i] != None:
                cl.append(self.problem.scaling_constraints[i] * self.problem.bounds[i][0])
                cu.append(self.problem.scaling_constraints[i] * self.problem.bounds[i][1])

        ub = np.array([max_float] * len(x0))
        lb = np.array([0.0] * len(x0))

        nlp = cyipopt.Problem(
            n=len(x0),
            m=len(cl),
            problem_obj=self.problem_obj,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
        )

        # initial point trafo
        x0 = self.problem.initial_point_trafo(x0)

        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('hessian_approximation', 'limited-memory')
        nlp.add_option('limited_memory_update_type', 'bfgs')
        nlp.add_option('limited_memory_max_history', 50)
        nlp.add_option('point_perturbation_radius', 0.0)

        # a benefitial option for starts close to solution:
        nlp.add_option('bound_mult_init_method', 'mu-based')

        nlp.add_option('max_iter', 15)
        nlp.add_option('tol', 1e-3)

        x, info = nlp.solve(x0)
        x = self.problem.transformation(x)
        return x