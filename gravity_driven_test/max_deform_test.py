import dolfin as df
import numpy as np
import os

from pathlib import Path

from extensions import ExtensionOperator

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

    path_to_files = "gravity_driven_test/data/max_deformations/max_deformations"
    save_to_dir = Path("gravity_driven_test/data/max_deformations")

    from extensions import BiharmonicExtension, HarmonicExtension, NNCorrectionExtension, LearnExtension
    biharmonic_extension = BiharmonicExtension(fluid_mesh)
    harmonic_extension = HarmonicExtension(fluid_mesh)
    nn_correct_extension_yankee = NNCorrectionExtension(fluid_mesh, "torch_extension/models/yankee")
    nn_correct_extension_foxtrot = NNCorrectionExtension(fluid_mesh, "torch_extension/models/foxtrot")
    hybrid_extension = LearnExtension(fluid_mesh)

    read_order = 2
    save_order = 1
    extend_from_file(path_to_files, save_to_dir / "biharmonic", biharmonic_extension, read_order, save_order)
    extend_from_file(path_to_files, save_to_dir / "harmonic", harmonic_extension, read_order, save_order)
    extend_from_file(path_to_files, save_to_dir / "nn_correct_yankee", nn_correct_extension_yankee, read_order, save_order)
    extend_from_file(path_to_files, save_to_dir / "nn_correct_foxtrot", nn_correct_extension_foxtrot, read_order, save_order)
    extend_from_file(path_to_files, save_to_dir / "hybrid", hybrid_extension, read_order, save_order)

    return


if __name__ == "__main__":
    main()
