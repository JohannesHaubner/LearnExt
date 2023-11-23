import dolfin as df
import numpy as np

import os

from problem import Problem
from pathlib import Path

from typing import Literal

def get_F(u):
    return df.Identity(2) + df.grad(u)

def get_E(F):
    return 0.5 * ( F.T * F - df.Identity(2) )

def get_J(F):
    return df.det(F)

def get_S(E, mu: df.Constant, lambda_: df.Constant):
    return 2*mu * E + lambda_ * df.tr(E) * df.Identity(2)


class Solver:

    def __init__(self, problem: Problem, order: int = 2, dt: float = 1e-3, T: float = 0.1,
                 time_stepping: Literal["explicit_euler"] = "explicit_euler"):

        self.problem = problem

        self.order = order

        self.mesh = self.problem.mesh
        self.boundaries = self.problem.boundaries

        self.dt = dt
        self.t = 0.0
        self.T = T
        self.k = df.Constant(dt)

        self.time_stepping = time_stepping
        self.get_weak_form = {
            "explicit_euler": self.get_weak_form_explicit_euler
        }[self.time_stepping]

        V_el = df.VectorElement("CG", self.mesh.ufl_cell(), self.order)
        W_el = df.MixedElement([V_el, V_el])
        self.W = df.FunctionSpace(self.mesh, W_el)

        self.uv = df.TrialFunction(self.W)
        self.yz = df.TestFunction(self.W)

        self.u, self.v = df.split(self.uv)
        self.y, self.z = df.split(self.yz)

        w = df.Function(self.W)
        self.u_, self.v_ = w.split()


        return
    
    def solve(self, save_dir: os.PathLike):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if (save_dir / "solid.xdmf").exists():
            (save_dir / "solid.xdmf").unlink()
            (save_dir / "solid.h5").unlink()
        
        solid_file = df.XDMFFile(str(save_dir / "solid.xdmf"))
        solid_file.write(self.mesh)
        solid_file.write_checkpoint(self.u_, "uh", self.t, append=True)

        while self.t < self.T:
            self.step()
            self.t += self.dt
            solid_file.write_checkpoint(self.u_, "uh", self.t, append=True)

        solid_file.close()

        return
    
    def step(self):

        lhs, rhs, bcs = self.get_weak_form()
        w = df.Function(self.W)
        df.solve(lhs == rhs, w, bcs)
        self.u_, self.v_ = w.split()

        return
    
    def get_weak_form_explicit_euler(self):
        u, v, y, z = self.u, self.v, self.y, self.z
        u_, v_ = self.u_, self.v_

        rho = df.Constant(self.problem.rho)
        mu = df.Constant(self.problem.mu)
        lambda_ = df.Constant(self.problem.lambda_)
        g = df.Constant((0.0, self.problem.l))
        k = self.k
        
        F = df.Identity(2) + df.grad(u_)
        E = 0.5 * (F.T * F - df.Identity(2))
        S = 2*mu * E + lambda_ * df.tr(E) * df.Identity(2)
        P = F * S


        lhs = df.inner(u, y) * df.dx + \
              rho * df.inner(v, z) * df.dx

        rhs = df.inner(u_, y) * df.dx + k * df.inner(v_, y) * df.dx + \
              rho * df.inner(g, z) * df.dx  -k * df.inner(P, df.grad(z)) * df.dx
        
        bc = df.DirichletBC(self.W.sub(0), df.Constant((0.0, 0.0)), self.boundaries, 4)

        return lhs, rhs, [bc]
    
if __name__ == "__main__":
    problem = Problem(1.0)
    solver = Solver(problem, order=1, dt=0.001, T=0.1)
    solver.solve("gravity_driven_test/data/test")


