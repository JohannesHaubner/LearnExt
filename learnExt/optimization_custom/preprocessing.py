from dolfin import *
from dolfin_adjoint import *
import numpy as np


class Preprocessing:
    def __init__(self, FunctionSpace):
        self.F = FunctionSpace

    def dof_to_control(self, x):
        """
        map vector of degrees of freedom to function on triangular mesh
        degrees of freedom live on quadrilateral 2d mesh, whereas rho lives on uniform triangular 2d mesh
        """
        func = Function(self.F)
        func.vector()[:] = x
        return func

    def dof_to_control_chainrule(self, djy, option=1):
        """
        chainrule of dof_to_control
        """
        if option ==2:
            return djy
        else:
            return djy.vector()
