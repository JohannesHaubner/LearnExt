import dolfin as df
import numpy as np
import os

from pathlib import Path

from FSIsolver.extension_operator.extension import ExtensionOperator

from run_extension import find_number_of_checkpoints


def extend_from_file(path_to_files: os.PathLike, save_to_path: os.PathLike, extension: ExtensionOperator, 
                     order: int, save_order: int = 1):
    path_to_files = Path(path_to_files)
    assert path_to_files.with_suffix(".xdmf").exists()
    assert path_to_files.with_suffix(".h5").exists()
    assert path_to_files.with_suffix(".ls.txt").exists()
    
    save_to_path = Path(save_to_path)
    save_to_path.with_suffix(".xdmf").unlink(missing_ok=True)
    save_to_path.with_suffix(".h5").unlink(missing_ok=True)
    save_to_path.with_suffix(".ls.txt").unlink(missing_ok=True)

    mesh = df.Mesh()
    infile = df.XDMFFile(str(path_to_files.with_suffix(".xdmf")))
    infile.read(mesh)

    outfile = df.XDMFFile(str(save_to_path.with_suffix(".xdmf")))
    outfile.write(mesh)

    save_to_path.with_suffix(".ls.txt").write_text(path_to_files.with_suffix(".ls.txt").read_text())

    V = df.VectorFunctionSpace(mesh, "CG", order)
    V_save = df.VectorFunctionSpace(mesh, "CG", save_order)
    
    u_bc = df.Function(V)
    u_save = df.Function(V_save)

    ks = range(find_number_of_checkpoints(path_to_files.with_suffix(".xdmf")))
    for k in ks:
        infile.read_checkpoint(u_bc, "uh", k)
        u_ext = extension.extend(u_bc)
        u_save.interpolate(u_ext)
        outfile.write_checkpoint(u_save, "uh", k, append=True)

    infile.close()
    outfile.close()

    return



def main():

    from problem import Problem
    fluid_mesh = Problem(0.0).fluid_mesh

    path_to_files = "gravity_driven_test/data/max_deformations_redo/max_deformations_redo"
    save_to_dir = Path("gravity_driven_test/data/max_deformations_redo")

    from FSIsolver.extension_operator.extension import Biharmonic, Harmonic, TorchExtension, LearnExtensionSimplifiedSNES, LearnExtensionSimplified
    biharmonic_extension = Biharmonic(fluid_mesh)
    harmonic_extension = Harmonic(fluid_mesh)
    hybrid_fsi_extension = LearnExtensionSimplifiedSNES(fluid_mesh, "example/learned_networks/trained_network.pkl", snes_divergence_tolerance=1e18, snes_max_it=100)
    hybrid_art_extension = LearnExtensionSimplifiedSNES(fluid_mesh, "example/learned_networks/artificial/trained_network.pkl", snes_divergence_tolerance=1e18, snes_max_it=100)
    # hybrid_fsi_extension = LearnExtensionSimplified(fluid_mesh, "example/learned_networks/trained_network.pkl")
    # hybrid_art_extension = LearnExtensionSimplified(fluid_mesh, "example/learned_networks/artificial/trained_network.pkl")
    nn_correct_extension_fsi = TorchExtension(fluid_mesh, "torch_extension/models/yankee")
    nn_correct_extension_art = TorchExtension(fluid_mesh, "torch_extension/models/foxtrot")

    read_order = 2
    save_order = 1
    extend_from_file(path_to_files, save_to_dir / "biharmonic", biharmonic_extension, read_order, save_order)
    extend_from_file(path_to_files, save_to_dir / "harmonic", harmonic_extension, read_order, save_order)
    extend_from_file(path_to_files, save_to_dir / "hybrid_fsi", hybrid_fsi_extension, read_order, save_order)
    extend_from_file(path_to_files, save_to_dir / "hybrid_art", hybrid_art_extension, read_order, save_order)
    extend_from_file(path_to_files, save_to_dir / "nn_correct_fsi", nn_correct_extension_fsi, read_order, save_order)
    extend_from_file(path_to_files, save_to_dir / "nn_correct_art", nn_correct_extension_art, read_order, save_order)

    return


if __name__ == "__main__":
    main()
