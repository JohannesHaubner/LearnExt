import dolfin as df
import numpy as np
import os

import tqdm
from pathlib import Path

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


if __name__ == "__main__":
    problem = Problem(1.0)
    solver = Solver(problem, order=2, dt=0.02, T=1.0, time_stepping="implicit_euler")
    mesh = problem.fluid_mesh
    from extensions import BiharmonicExtension, HarmonicExtension
    extension = BiharmonicExtension(mesh)
    # extension = HarmonicExtension(mesh)
    
    # run_extensions_from_file(extension, "gravity_driven_test/data/test/fluid_harm.xdmf", 
    #                          "gravity_driven_test/data/test/fluid_biharm.xdmf", save_order=1)
    run_extensions_with_solver(extension, solver, "gravity_driven_test/data/test/fluid_biharm.xdmf",
                               save_order=1)
    