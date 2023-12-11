import dolfin as df
import numpy as np
import os

from pathlib import Path
from tqdm import tqdm

from FSIsolver.extension_operator.extension import ExtensionOperator


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
    try:
        df.ALE.move(mesh1, u, annotate=False)
    except:
        df.ALE.move(mesh1, u)
    mesh1.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
    post_orient1 = 2 * np.array(mesh1.cell_orientations()) - 1

    flipped_cells = pre_orient1 * post_orient1
    return flipped_cells.astype(np.float64)

def extend_from_file(path_to_files: os.PathLike, save_to_path: os.PathLike, 
                     extension: ExtensionOperator, order: int):
    print(str(path_to_files))
    print(str(save_to_path))
    path_to_files = Path(path_to_files)
    assert path_to_files.with_suffix(".xdmf").exists()
    assert path_to_files.with_suffix(".h5").exists()
    
    save_to_path = Path(save_to_path)
    save_to_path.with_suffix(".xdmf").unlink(missing_ok=True)
    save_to_path.with_suffix(".h5").unlink(missing_ok=True)
    save_to_path.with_suffix(".signs.xdmf").unlink(missing_ok=True)
    save_to_path.with_suffix(".signs.h5").unlink(missing_ok=True)

    mesh = df.Mesh()
    infile = df.XDMFFile(str(path_to_files.with_suffix(".xdmf")))
    infile.read(mesh)

    outfile = df.XDMFFile(str(save_to_path.with_suffix(".xdmf")))
    outfile.write(mesh)
    outfile_signs = df.XDMFFile(str(save_to_path.with_suffix(".signs.xdmf")))
    outfile_signs.write(mesh)


    V = df.VectorFunctionSpace(mesh, "CG", order)
    V_save = df.VectorFunctionSpace(mesh, "CG", 1)

    # interp_mat = df.PETScDMCollection.create_transfer_matrix(V, V_save)
    
    u_bc = df.Function(V)
    u_save = df.Function(V_save)

    DG0 = df.FunctionSpace(mesh, "DG", 0)
    signs = df.Function(DG0)


    ks = range(find_number_of_checkpoints(path_to_files.with_suffix(".xdmf")))
    ks = [0, 10, 20, 120, 272]
    for k in tqdm(ks):
        infile.read_checkpoint(u_bc, "uh", k)
        u_ext = extension.extend(u_bc)
        u_save.interpolate(u_ext)
        # interp_mat.mult(u_ext.vector(), u_save.vector()) # u_save.interpolate(u_ext)
        flipped_cells = get_flipped_cells(mesh, u_save)
        signs.vector()[:] = flipped_cells
        outfile.write_checkpoint(u_save, "uh", k, append=True)
        outfile_signs.write_checkpoint(signs, "sign_h", k, append=True)

    infile.close()
    outfile.close()
    outfile_signs.close()

    return



def main():

    path_to_files = Path("Data_MoF-2/membrane_test_p2.xdmf")
    save_to_dir = Path("membrane_test/data")

    mesh = df.Mesh()
    with df.XDMFFile(str(path_to_files)) as meshfile:
        meshfile.read(mesh)

    from FSIsolver.extension_operator.extension import Harmonic

    harmonic = Harmonic(mesh, save_extension=False)

    df.set_log_active(False)
    order = 2
    extend_from_file(path_to_files, save_to_dir / "harmonic", harmonic, order)

    return


if __name__ == "__main__":
    main()
