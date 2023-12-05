import dolfin as df
import numpy as np
import os

import tqdm
from pathlib import Path

from typing import Sequence, Literal

from problem import Problem
from solver import Solver

from extensions import ExtensionOperator

def find_number_of_checkpoints(path_to_xdmf: os.PathLike) -> int:

    with open(path_to_xdmf, "r") as infile:
        lines = infile.readlines()

    for line in lines[:-20:-1]:
        if "Time" in line:
            num = line.split('"')[1]
            num = float(num)
            num = int(num)
            return num + 1
    raise RuntimeError

def find_cg_order(path_to_xdmf: os.PathLike) -> int:

    with open(path_to_xdmf, "r") as infile:
        for k in range(40):
            line = infile.readline()
            if "ElementDegree" in line:
                ind = next(filter(lambda i: line[i:i+len("ElementDegree")] == "ElementDegree",
                                  range(len(line))))
                order = int(line[ind + len("ElementDegree") + 2])
                return order
    raise RuntimeError

def run_extensions_from_file(extension: ExtensionOperator, path_to_file: os.PathLike, 
                             save_to_file: os.PathLike, save_order: int = 1, 
                             log_active: bool = False) -> None:
    path_to_file = Path(path_to_file)
    save_to_file = Path(save_to_file)

    df.set_log_active(log_active)

    num_checkpoints = find_number_of_checkpoints(path_to_file)
    order = find_cg_order(path_to_file)

    bc_file = df.XDMFFile(str(path_to_file))

    if save_to_file.exists():
        save_to_file.unlink()
        save_to_file.with_suffix(".h5").unlink()
    save_file = df.XDMFFile(str(save_to_file))

    mesh = df.Mesh()
    bc_file.read(mesh)
    save_file.write(mesh)

    V = df.VectorFunctionSpace(mesh, "CG", order)
    u_bc = df.Function(V)
    
    V_save = df.VectorFunctionSpace(mesh, "CG", save_order)
    u_save = df.Function(V_save)

    loop = range(num_checkpoints) if log_active else tqdm.tqdm(range(num_checkpoints))
    for k in loop:
        bc_file.read_checkpoint(u_bc, "uh", k)
        u_ext = extension.extend(u_bc)
        u_save.interpolate(u_ext)
        save_file.write_checkpoint(u_save, "uh", k, append=True)

    bc_file.close()
    save_file.close()
    return

def run_extensions_with_solver(extension: ExtensionOperator, solver: Solver, 
                               save_to_file: os.PathLike, save_order: int = 1, 
                               log_active: bool = False) -> None:
    save_to_file = Path(save_to_file)
    from meshing import translate_function

    problem = solver.problem

    if save_to_file.exists():
        save_to_file.unlink()
        save_to_file.with_suffix(".h5").unlink()
    save_file = df.XDMFFile(str(save_to_file))
    save_file.write(problem.fluid_mesh)    

    df.set_log_active(log_active)
    
    V_solid = df.VectorFunctionSpace(problem.mesh, "CG", solver.order)
    u_solid = df.Function(V_solid)
    V_save = df.VectorFunctionSpace(problem.fluid_mesh, "CG", save_order)
    u_save = df.Function(V_save)

    if log_active is False:
            pbar = tqdm.tqdm(total=solver.T)
            import warnings
            warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)

    u_solid.interpolate(solver.u_)
    u_bc = translate_function(from_u=u_solid,
                                    from_facet_f=solver.boundaries,
                                    to_facet_f=solver.problem.fluid_boundaries,
                                    shared_tags=solver.problem.interface_tags)
    u_ext = extension.extend(u_bc)
    u_save.interpolate(u_ext)
    save_file.write_checkpoint(u_save, "uh", solver.it, append=True)

    while solver.t < solver.T:
            solver.step()
            solver.t += solver.dt
            solver.it += 1
            if log_active is False: pbar.update(solver.dt)

            u_solid.interpolate(solver.u_)
            u_bc = translate_function(from_u=u_solid,
                                    from_facet_f=solver.boundaries,
                                    to_facet_f=solver.problem.fluid_boundaries,
                                    shared_tags=solver.problem.interface_tags)
            u_ext = extension.extend(u_bc)
            u_save.interpolate(u_ext)
            save_file.write_checkpoint(u_save, "uh", solver.it, append=True)

    if log_active is False:
        pbar.close()
        warnings.filterwarnings("default", category=tqdm.TqdmWarning)
    save_file.close()

    return


def get_maximum_deflection(solver: Solver, extension: ExtensionOperator, 
                               return_order: int = 2, log_active: bool = False) -> df.Function:
    """
    Return the function on the fluid domain corresponding to the extension of the time step of maximum deflection of the solid, solved by solver.

    Args:
        solver (Solver): The Solver object that the simulation is run with. This contains the Problem object deciding the value of l, 
                            which decides how much the solid is deformed
        extension (ExtensionOperator): The extension operator to apply to the maximal deformed time-step.
        return_order (int, optional): Which order Lagrange element to return function with. Defaults to 2.
        log_active (bool, optional): Whether or not to print out FEniCS log info, like convergence of nonlinear solvers. Defaults to False.

    Returns:
        df.Function: Function with extension to fluid domain of maximum solid deflection timestep.
    """
    from gravity_driven_test.meshing import translate_function
    df.set_log_active(log_active)

    problem = solver.problem
    V_solid = df.VectorFunctionSpace(problem.mesh, "CG", solver.order)
    u_solid = df.Function(V_solid)
    V_return = df.VectorFunctionSpace(problem.fluid_mesh, "CG", return_order)
    u_return = df.Function(V_return)

    sensor_loc = np.array([0.6, 0.21])

    x = V_solid.tabulate_dof_coordinates()
    cand_dofs = np.flatnonzero(np.linalg.norm(x - sensor_loc, axis=1) < 1e-3)
    dof = cand_dofs[1]

    y_last = -np.inf
    y_now = u_solid.vector()[dof]
    while solver.t < solver.T:
        print(f"\r\t\t\t\t\t\rit={solver.it:02d}, t={solver.t:.2f}", end="")
        solver.step()
        solver.t += solver.dt
        solver.it += 1
        u_solid.interpolate(solver.u_)
        y_last = np.copy(y_now)
        y_now = u_solid.vector()[dof]
        u_bc = translate_function(from_u=u_solid,
                                from_facet_f=solver.boundaries,
                                to_facet_f=solver.problem.fluid_boundaries,
                                shared_tags=solver.problem.interface_tags)
        u_ext = extension.extend(u_bc)

        if y_now - y_last < 0.0:
            print()
            return u_return
        
        u_return.interpolate(u_ext)

    raise RuntimeError

def save_maximum_deflections(ls: Sequence[float], save_path: os.PathLike, extension: ExtensionOperator,
                            solver_order: int = 2, save_order: int = 2, dt: float = 0.02, T: float = 1.0, 
                            time_stepping: Literal["implicit_euler"] = "implicit_euler") -> None:
    """
    Save to file the maximum deformation of the solid with load given by l in ls, represented on the fluid mesh as the result of extension.

    Args:
        ls (Sequence[float]): Gravitational loads for the solid
        save_path (os.PathLike): Where to save the extensions.
        extension (ExtensionOperator): The extension operator to translate onto the fluid mesh.
        solver_order (int, optional): Which order Lagrange elements to use in the solid solver. Defaults to 2.
        save_order (int, optional): Which order lagrange elements to save the extension on the fluid mesh. Defaults to 2.
        dt (float, optional): Time step for solid solver. Defaults to 0.02.
        T (float, optional): Maximum time for solid Solver. Defaults to 1.0.
        time_stepping (Literal['implicit_euler'], optional): Time stepping method for solid solver. Defaults to "implicit_euler".
    """
    
    problem_0 = Problem(0.0)
    fluid_mesh = problem_0.fluid_mesh

    save_path = Path(save_path)
    save_path.with_suffix(".xdmf").unlink(missing_ok=True)
    save_path.with_suffix(".h5").unlink(missing_ok=True)
    save_path.with_suffix(".ls.txt").unlink(missing_ok=True)

    save_file = df.XDMFFile(str(save_path.with_suffix(".xdmf")))
    save_file.write(fluid_mesh)

    save_path.with_suffix(".ls.txt").write_text("".join([f"{l}\n" for l in ls]))

    for k, l in enumerate(ls):
        print(f"{l = }")
        problem = Problem(l)
        solver = Solver(problem, order=solver_order, dt=dt, T=T, time_stepping=time_stepping)
        u_l = get_maximum_deflection(solver, extension, return_order=save_order, log_active=False)
        save_file.write_checkpoint(u_l, "uh", k, append=True)
        print()

    save_file.close()

    return


if __name__ == "__main__":
    problem_in = Problem(2.5)
    solver_in = Solver(problem_in, order=2, dt=0.02, T=1.0, time_stepping="implicit_euler")
    mesh_in = problem_in.fluid_mesh
    from extensions import BiharmonicExtension, HarmonicExtension, NNCorrectionExtension
    extension_in = BiharmonicExtension(mesh_in)
    # extension_in = HarmonicExtension(mesh_in)
    # extension_in = NNCorrectionExtension(mesh_in, "torch_extension/models/yankee")
    
    # run_extensions_from_file(extension, "gravity_driven_test/data/test/fluid_harm.xdmf", 
    #                          "gravity_driven_test/data/test/fluid_biharm.xdmf", save_order=1)
    # run_extensions_with_solver(extension_in, solver_in, "gravity_driven_test/data/test/fluid_yankee.xdmf",
    #                            save_order=1)

    # max_def = get_maximum_deflection(solver_in, extension_in, 2, False)

    ls = [1.0, 2.0, 2.5]
    # save_maximum_deflections(ls, "gravity_driven_test/data/max_deformations/max_deformations", extension_in, 
    #                          solver_order=2, save_order=2, dt=0.02, T=3.0, time_stepping="implicit_euler")
    save_maximum_deflections(ls, "gravity_driven_test/data/max_deformations/max_deformations_p1", extension_in, 
                             solver_order=2, save_order=1, dt=0.02, T=3.0, time_stepping="implicit_euler")
    