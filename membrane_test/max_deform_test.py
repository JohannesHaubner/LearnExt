import dolfin as df
import numpy as np
import os

from pathlib import Path
from tqdm import tqdm
from typing import Sequence

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
                     extension: ExtensionOperator, order: int, save_order: int = 1, checkpoints: Sequence[int] | None = None):
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
    CG1 = df.VectorFunctionSpace(mesh, "CG", 1)
    V_save = df.VectorFunctionSpace(mesh, "CG", save_order)

    # interp_mat = df.PETScDMCollection.create_transfer_matrix(V, V_save)
    
    u_bc = df.Function(V)
    u_cg1 = df.Function(CG1)
    u_save = df.Function(V_save)

    DG0 = df.FunctionSpace(mesh, "DG", 0)
    signs = df.Function(DG0)


    if checkpoints is None:
        ks = range(find_number_of_checkpoints(path_to_files.with_suffix(".xdmf")))
    else:
        ks = checkpoints
    for l, k in enumerate(tqdm(ks)):
        infile.read_checkpoint(u_bc, "uh", k)
        u_ext = extension.extend(u_bc)
        u_cg1.interpolate(u_ext)
        u_save.interpolate(u_ext)
        # interp_mat.mult(u_ext.vector(), u_save.vector()) # u_save.interpolate(u_ext)
        flipped_cells = get_flipped_cells(mesh, u_cg1)
        signs.vector()[:] = flipped_cells
        outfile.write_checkpoint(u_save, "uh", l, append=True)
        outfile_signs.write_checkpoint(signs, "sign_h", l, append=True)

    infile.close()
    outfile.close()
    outfile_signs.close()

    return



def main():

    path_to_files = Path("membrane_test/data/Data_MoF/membrane_test_p2.xdmf")
    # save_to_dir = Path("membrane_test/data/extended")
    # save_to_dir = Path("membrane_test/data/extended_p2")
    # save_to_dir = Path("membrane_test/data/cell_flip")
    save_to_dir = Path("membrane_test/data/histograms")

    mesh = df.Mesh()
    with df.XDMFFile(str(path_to_files)) as meshfile:
        meshfile.read(mesh)

    from FSIsolver.extension_operator.extension import Biharmonic, Harmonic, TorchExtension, LearnExtensionSimplified, LearnExtensionSimplifiedSNES

    biharmonic = Biharmonic(mesh)
    harmonic = Harmonic(mesh, save_extension=False)
    # hybrid_fsi = LearnExtensionSimplified(mesh, "example/learned_networks/trained_network.pkl")
    # hybrid_art = LearnExtensionSimplified(mesh, "example/learned_networks/artificial/trained_network.pkl")
    NN_correct_fsi = TorchExtension(mesh, "torch_extension/models/yankee", T_switch=0.0, silent=True)
    NN_correct_art = TorchExtension(mesh, "torch_extension/models/foxtrot", T_switch=0.0, silent=True)
    hybrid_fsi_snes = LearnExtensionSimplifiedSNES(mesh, "example/learned_networks/trained_network.pkl")
    hybrid_art_snes = LearnExtensionSimplifiedSNES(mesh, "example/learned_networks/artificial/trained_network.pkl")

    df.set_log_active(False)
    order = 2
    save_order = 1
    checkpoints = [0, 10, 20, 120, 272]
    # checkpoints = list(range(150, 211))
    checkpoints = [50, 150, 272]
    # checkpoints = None
    extend_from_file(path_to_files, save_to_dir / "biharmonic", biharmonic, order, save_order=save_order, checkpoints=checkpoints)
    extend_from_file(path_to_files, save_to_dir / "harmonic", harmonic, order, save_order=save_order, checkpoints=checkpoints)
    # extend_from_file(path_to_files, save_to_dir / "hybrid_fsi", hybrid_fsi, order, save_order=save_order, checkpoints=checkpoints)
    # extend_from_file(path_to_files, save_to_dir / "hybrid_art", hybrid_art, order, save_order=save_order, checkpoints=checkpoints)
    extend_from_file(path_to_files, save_to_dir / "nn_correct_fsi", NN_correct_fsi, order, save_order=save_order, checkpoints=checkpoints)
    extend_from_file(path_to_files, save_to_dir / "nn_correct_art", NN_correct_art, order, save_order=save_order, checkpoints=checkpoints)
    extend_from_file(path_to_files, save_to_dir / "hybrid_fsi_snes", hybrid_fsi_snes, order, save_order=save_order, checkpoints=checkpoints)
    extend_from_file(path_to_files, save_to_dir / "hybrid_art_snes", hybrid_art_snes, order, save_order=save_order, checkpoints=checkpoints)

    return


if __name__ == "__main__":
    main()
