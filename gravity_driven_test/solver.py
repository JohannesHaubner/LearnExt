import dolfin as df
import numpy as np

import os
import tqdm

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
                 time_stepping: Literal["implicit_euler"] = "implicit_euler"):

        self.problem = problem

        self.order = order

        self.mesh = self.problem.mesh
        self.boundaries = self.problem.boundaries

        self.dt = dt
        self.t = 0.0
        self.T = T
        self.k = df.Constant(dt)
        self.it = 0

        self.time_stepping = time_stepping
        self.step = {
            "implicit_euler": self.step_implicit_euler
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
    
    def solve(self, save_dir: os.PathLike, fluid_order: int | None = None, log_active=False):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if (save_dir / "solid.xdmf").exists():
            (save_dir / "solid.xdmf").unlink()
            (save_dir / "solid.h5").unlink()
        if (save_dir / "fluid_harm.xdmf").exists():
            (save_dir / "fluid_harm.xdmf").unlink()
            (save_dir / "fluid_harm.h5").unlink()

        df.set_log_active(log_active)
        
        solid_file = df.XDMFFile(str(save_dir / "solid.xdmf"))
        solid_file.write(self.mesh)
        fluid_file = df.XDMFFile(str(save_dir / "fluid_harm.xdmf"))
        fluid_file.write(self.problem.fluid_mesh)
        
        solid_file.write_checkpoint(self.u_, "uh", self.it, append=True)

        solid_V = df.VectorFunctionSpace(self.mesh, "CG", self.order)
        uh_solid = df.Function(solid_V)
        uh_solid.interpolate(self.u_)

        fluid_order = self.order if fluid_order is None else fluid_order
        uh_fluid = self.extend_harmonic(uh_solid, order=fluid_order)
        fluid_file.write_checkpoint(uh_fluid, "uh", self.it, append=True)

        if log_active is False: 
            pbar = tqdm.tqdm(total=self.T)
            import warnings
            warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
        while self.t < self.T:
            self.step()
            self.t += self.dt
            self.it += 1
            if log_active is False: pbar.update(self.dt)

            solid_file.write_checkpoint(self.u_, "uh", self.it, append=True)

            uh_solid.interpolate(self.u_)
            uh_fluid = self.extend_harmonic(uh_solid, order=fluid_order)
            fluid_file.write_checkpoint(uh_fluid, "uh", self.it, append=True)

        solid_file.close()
        fluid_file.close()
        if log_active is False:
            pbar.close()
            warnings.filterwarnings("default", category=tqdm.TqdmWarning)

        return

    def step_implicit_euler(self):

        uv = df.Function(self.W)
        u, v = df.split(uv)
        y, z = self.y, self.z
        u_, v_ = self.u_, self.v_

        rho = df.Constant(self.problem.rho)
        mu = df.Constant(self.problem.mu)
        lambda_ = df.Constant(self.problem.lambda_)
        g = df.Constant((0.0, self.problem.l))
        k = self.k
        
        F = df.Identity(2) + df.grad(u)
        E = 0.5 * (F.T * F - df.Identity(2))
        S = 2*mu * E + lambda_ * df.tr(E) * df.Identity(2)
        P = F * S

        system = df.inner(u, y) * df.dx - k * df.inner(v, y) * df.dx + \
                 -df.inner(u_, y) * df.dx + \
                 rho * df.inner(v, z) * df.dx + k * df.inner(P, df.grad(z)) * df.dx + \
                 -rho * k * df.inner(g, z) * df.dx - rho * df.inner(v_, z) * df.dx

        bc = df.DirichletBC(self.W.sub(0), df.Constant((0.0, 0.0)), self.boundaries, 4)

        df.solve(system == 0, uv, [bc], solver_parameters={"nonlinear_solver": "newton", "newton_solver":
            {"maximum_iterations": 20}})
        self.u_, self.v_ = uv.split()
        
        return
    
    def extend_harmonic(self, uh: df.Function, order: int | None = None) -> df.Function:
        from meshing import translate_function
        if order is None:
            order = self.order

        uh_fluid = translate_function(from_u=uh,
                                    from_facet_f=self.boundaries,
                                    to_facet_f=self.problem.fluid_boundaries,
                                    shared_tags=self.problem.interface_tags)

        V = df.VectorFunctionSpace(self.problem.fluid_mesh, 'CG', order)
        u, v = df.TrialFunction(V), df.TestFunction(V)

        a = df.inner(df.grad(u), df.grad(v))*df.dx
        L = df.inner(df.Constant((0, )*len(u)), v)*df.dx
        # Those from solid
        bcs = [df.DirichletBC(V, uh_fluid, self.problem.fluid_boundaries, tag) for tag in self.problem.interface_tags]
        # The rest is fixed
        null = df.Constant((0, )*len(u))
        bcs.extend([df.DirichletBC(V, null, self.problem.fluid_boundaries, tag) for tag in self.problem.zero_displacement_tags])

        uh_f = df.Function(V)
        df.solve(a == L, uh_f, bcs)

        return uh_f
    

if __name__ == "__main__":
    problem = Problem(2.5)
    solver = Solver(problem, order=2, dt=0.02, T=1.0, time_stepping="implicit_euler")
    solver.solve("gravity_driven_test/data/test", fluid_order=2)


