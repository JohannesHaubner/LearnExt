import dolfin as df
import numpy as np
import os

from pathlib import Path


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


def get_flipped_cells(mesh: df.Mesh, u: df.Function) -> df.Function:
    assert u.function_space().ufl_element().degree() == 1

    mesh1 = df.Mesh(mesh)
    mesh1.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
    pre_orient1 = 2 * np.array(mesh1.cell_orientations()) - 1
    df.ALE.move(mesh1, u)
    mesh1.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
    post_orient1 = 2 * np.array(mesh1.cell_orientations()) - 1

    flipped_cells = pre_orient1 * post_orient1
    return flipped_cells.astype(np.float64)

def write_flipped_cells_from_file(path_to_files: os.PathLike, 
                                  save_to_path: os.PathLike, order: int) -> None:
    path_to_files = Path(path_to_files)
    assert path_to_files.with_suffix(".xdmf").exists()
    assert path_to_files.with_suffix(".h5").exists()
    assert path_to_files.with_suffix(".ls.txt").exists()
    
    save_to_path = Path(save_to_path)
    save_to_path.with_suffix(".signs.xdmf").unlink(missing_ok=True)

    print(str(path_to_files))

    mesh = df.Mesh()
    infile = df.XDMFFile(str(path_to_files.with_suffix(".xdmf")))
    infile.read(mesh)

    outfile = df.XDMFFile(str(save_to_path.with_suffix(".signs.xdmf")))
    outfile.write(mesh)

    V = df.VectorFunctionSpace(mesh, "CG", order)
    CG1 = df.VectorFunctionSpace(mesh, "CG", 1)
    DG0 = df.FunctionSpace(mesh, "DG", 0)
    signs = df.Function(DG0)

    u = df.Function(V)
    u_cg1 = df.Function(CG1)

    ks = range(find_number_of_checkpoints(path_to_files.with_suffix(".xdmf")))
    for k in ks:
        infile.read_checkpoint(u, "uh", k)
        u_cg1.interpolate(u)
        flipped_cells = get_flipped_cells(mesh, u_cg1)
        print(np.count_nonzero(flipped_cells == -1))
        signs.vector()[:] = flipped_cells
        outfile.write_checkpoint(signs, "sign_h", k, append=True)

    print()

    infile.close()
    outfile.close()

    return



def main():


    files_dir = Path("gravity_driven_test/data/max_deformations_redo")

    read_order = 1

    path = files_dir / "biharmonic"; write_flipped_cells_from_file(path, path, read_order)
    path = files_dir / "harmonic"; write_flipped_cells_from_file(path, path, read_order)
    path = files_dir / "hybrid_fsi"; write_flipped_cells_from_file(path, path, read_order)
    path = files_dir / "hybrid_art"; write_flipped_cells_from_file(path, path, read_order)
    path = files_dir / "nn_correct_fsi"; write_flipped_cells_from_file(path, path, read_order)
    path = files_dir / "nn_correct_art"; write_flipped_cells_from_file(path, path, read_order)

    return


if __name__ == "__main__":
    main()
