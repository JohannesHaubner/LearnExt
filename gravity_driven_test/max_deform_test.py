import dolfin as df
import numpy as np
import os

from pathlib import Path

from FSIsolver.extension_operator.extension import ExtensionOperator

from run_extension import find_number_of_checkpoints


def get_flipped_cells(u: df.Function) -> np.ndarray:
    assert u.function_space().ufl_element().degree() == 1

    mesh1 = df.Mesh(u.function_space().mesh())
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


def get_degenerate_cells(u: df.Function, tensor_dg_order: int = 2) -> df.Function:

    msh = u.function_space().mesh()

    F = df.Identity(len(u)) + df.grad(u)

    V = df.TensorFunctionSpace(msh, 'DG', tensor_dg_order)
    ndofs_per_dim = V.sub(0).collapse().dolfin_element().space_dimension()

    v = df.TrialFunction(V)
    dv = df.TestFunction(V)

    a = df.inner(v, dv)*df.dx
    L = df.inner(F, dv)*df.dx

    cell_stats = []
    # dim = V.dolfin_element().space_dimension()
    for cell in df.cells(msh):
        A = df.assemble_local(a, cell=cell)
        b = df.assemble_local(L, cell=cell)
        T = np.linalg.solve(A, b)

        # Sample in dofs
        dets = []
        T_at_dofs = T.reshape((4, ndofs_per_dim)).T
        for T in T_at_dofs:
            T = T.reshape((2, 2))
            dets.append(np.linalg.det(T))
        cell_stats.append((min(dets), max(dets)))
    cell_stats = np.array(cell_stats)

    # bad_cells = np.where(cell_stats[:, 0] < 0)
    bad_cells = np.where(cell_stats[:, 0] < 0, -np.ones(cell_stats.shape[0], dtype=np.float64), np.ones(cell_stats.shape[0], dtype=np.float64))

    # cell_f = df.MeshFunction('size_t', msh, 2, 0)
    # cell_f.array()[bad_cells] = 1

    return bad_cells


def extend_from_file(path_to_files: os.PathLike, save_to_path: os.PathLike, extension: ExtensionOperator, 
                     order: int, save_order: int = 1, degen_test_order: int = 2):
    path_to_files = Path(path_to_files)
    assert path_to_files.with_suffix(".xdmf").exists()
    assert path_to_files.with_suffix(".h5").exists()
    assert path_to_files.with_suffix(".ls.txt").exists()
    
    save_to_path = Path(save_to_path)
    save_to_path.with_suffix(".xdmf").unlink(missing_ok=True)
    save_to_path.with_suffix(".h5").unlink(missing_ok=True)
    save_to_path.with_suffix(".ls.txt").unlink(missing_ok=True)
    save_to_path.with_suffix(".signs.xdmf").unlink(missing_ok=True)
    save_to_path.with_suffix(".signs.h5").unlink(missing_ok=True)

    mesh = df.Mesh()
    infile = df.XDMFFile(str(path_to_files.with_suffix(".xdmf")))
    infile.read(mesh)

    outfile = df.XDMFFile(str(save_to_path.with_suffix(".xdmf")))
    outfile.write(mesh)

    outfile_signs = df.XDMFFile(str(save_to_path.with_suffix(".signs.xdmf")))
    outfile_signs.write(mesh)

    save_to_path.with_suffix(".ls.txt").write_text(path_to_files.with_suffix(".ls.txt").read_text())

    V = df.VectorFunctionSpace(mesh, "CG", order)
    V_save = df.VectorFunctionSpace(mesh, "CG", save_order)
    DG0 = df.FunctionSpace(mesh, "DG", 0)
    
    u_bc = df.Function(V)
    u_save = df.Function(V_save)
    signs = df.Function(DG0)

    ks = range(find_number_of_checkpoints(path_to_files.with_suffix(".xdmf")))
    for k in ks:
        infile.read_checkpoint(u_bc, "uh", k)
        u_ext = extension.extend(u_bc)
        cell_signs = get_degenerate_cells(u_ext, tensor_dg_order=degen_test_order)
        signs.vector()[:] = cell_signs
        u_save.interpolate(u_ext)
        outfile.write_checkpoint(u_save, "uh", k, append=True)
        outfile_signs.write_checkpoint(signs, "sign_h", k, append=True)

    infile.close()
    outfile.close()
    outfile_signs.close()

    return



def main():

    from problem import Problem
    fluid_mesh = Problem(0.0).fluid_mesh

    path_to_files = "gravity_driven_test/data/max_deformations_redo/max_deformations_redo"
    save_to_dir = Path("gravity_driven_test/data/max_deformations_degen_dg6")

    from FSIsolver.extension_operator.extension import Biharmonic, Harmonic, TorchExtension, LearnExtensionSimplifiedSNES, LearnExtensionSimplified
    biharmonic_extension = Biharmonic(fluid_mesh)
    harmonic_extension = Harmonic(fluid_mesh)
    hybrid_fsi_extension = LearnExtensionSimplifiedSNES(fluid_mesh, "example/learned_networks/trained_network.pkl", 
                                                        snes_divergence_tolerance=1e18, snes_max_it=100,
                                                        snes_atol=1e-10, snes_rtol=1e-48, snes_stol=1e-48)
    hybrid_art_extension = LearnExtensionSimplifiedSNES(fluid_mesh, "example/learned_networks/artificial/trained_network.pkl", 
                                                        snes_divergence_tolerance=1e18, snes_max_it=100,
                                                        snes_atol=1e-10, snes_rtol=1e-48, snes_stol=1e-48)
    # hybrid_fsi_extension = LearnExtensionSimplified(fluid_mesh, "example/learned_networks/trained_network.pkl")
    # hybrid_art_extension = LearnExtensionSimplified(fluid_mesh, "example/learned_networks/artificial/trained_network.pkl")
    nn_correct_extension_fsi = TorchExtension(fluid_mesh, "torch_extension/models/yankee")
    nn_correct_extension_art = TorchExtension(fluid_mesh, "torch_extension/models/foxtrot")

    read_order = 2
    save_order = 2
    degen_test_order = 6
    extend_from_file(path_to_files, save_to_dir / "biharmonic", biharmonic_extension, read_order, save_order, degen_test_order)
    extend_from_file(path_to_files, save_to_dir / "harmonic", harmonic_extension, read_order, save_order, degen_test_order)
    extend_from_file(path_to_files, save_to_dir / "hybrid_fsi", hybrid_fsi_extension, read_order, save_order, degen_test_order)
    extend_from_file(path_to_files, save_to_dir / "hybrid_art", hybrid_art_extension, read_order, save_order, degen_test_order)
    extend_from_file(path_to_files, save_to_dir / "nn_correct_fsi", nn_correct_extension_fsi, read_order, save_order, degen_test_order)
    extend_from_file(path_to_files, save_to_dir / "nn_correct_art", nn_correct_extension_art, read_order, save_order, degen_test_order)

    return


if __name__ == "__main__":
    main()
