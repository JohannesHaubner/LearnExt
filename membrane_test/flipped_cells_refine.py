import dolfin as df
import numpy as np

from pathlib import Path

# test_path = Path("membrane_test/data/extended_p2/harmonic.xdmf")
test_path = Path("membrane_test/data/cell_flip/hybrid_fsi_snes.xdmf")
print(str(test_path))

mesh = df.Mesh()
test_file = df.XDMFFile(str(test_path))
test_file.read(mesh)

marker = df.MeshFunction("bool", mesh, 2)
marker.set_all(False)
class RefArea(df.SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-7
        first = 0.004 - tol <= x[0] <= 0.016 + tol
        second = 0.004 - tol <= x[1] <= 0.00525 + tol
        return first and second
ref_area = RefArea()
ref_area.mark(marker, True)
print(np.count_nonzero(marker.array()))

# ref_mesh = df.Mesh(mesh)
ref_mesh = df.refine(mesh, marker)

ref_signs_outfile_path = test_path.with_suffix(".refsigns.xdmf")
ref_signs_outfile_path.unlink(missing_ok=True)
ref_signs_outfile = df.XDMFFile(str(ref_signs_outfile_path))
ref_signs_outfile.write(ref_mesh)

ref_def_outfile_path = test_path.with_suffix(".refdef.xdmf")
ref_def_outfile_path.unlink(missing_ok=True)
ref_def_outfile = df.XDMFFile(str(ref_def_outfile_path))
ref_def_outfile.write(ref_mesh)

proc_outfile_path = test_path.with_suffix(".procsigns.xdmf")
proc_outfile_path.unlink(missing_ok=True)
proc_outfile = df.XDMFFile(str(proc_outfile_path))
proc_outfile.write(mesh)

j_outfile_path = test_path.with_suffix(".j.xdmf")
j_outfile_path.unlink(missing_ok=True)
j_outfile = df.XDMFFile(str(j_outfile_path))
j_outfile.write(mesh)

j_min_outfile_path = test_path.with_suffix(".j_min.xdmf")
j_min_outfile_path.unlink(missing_ok=True)
j_min_outfile = df.XDMFFile(str(j_min_outfile_path))
j_min_outfile.write(mesh)

print(mesh.num_cells())
print(ref_mesh.num_cells())


CG2 = df.VectorFunctionSpace(mesh, "CG", 2)
CG1 = df.VectorFunctionSpace(mesh, "CG", 1)
CG2_ref = df.VectorFunctionSpace(ref_mesh, "CG", 2)
CG1_ref = df.VectorFunctionSpace(ref_mesh, "CG", 1)
DG0 = df.FunctionSpace(mesh, "DG", 0)
DG0_ref = df.FunctionSpace(ref_mesh, "DG", 0)
DG1 = df.FunctionSpace(mesh, "DG", 1)

DG4 = df.FunctionSpace(mesh, "DG", 4)
DG3 = df.FunctionSpace(mesh, "DG", 3)
DG2 = df.FunctionSpace(mesh, "DG", 2)

u_cg2 = df.Function(CG2)
u_cg1 = df.Function(CG1)
u_cg2_ref = df.Function(CG2_ref)
u_cg1_ref = df.Function(CG1_ref)
signs = df.Function(DG0)
signs_ref = df.Function(DG0_ref)
j = df.Function(DG1)

from FSIsolver.extension_operator.extension import LearnExtension

# ks = [20, 120, 272]
# ks = [0, 10, 20, 120, 272]
ks = [0, 1, 2, 3, 4]
# ks = [2, 3, 4]
# ks = list(range(150, 211))
ks = list(range(150 - 150, 211 - 150))
# ks = [ks[0]]

for k in ks:

    test_file.read_checkpoint(u_cg2, "uh", k)
    u_cg1_ref.interpolate(u_cg2)

    ref_mesh_cp = df.Mesh(ref_mesh)
    ref_mesh_cp.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
    pre_orient_ref = 2 * np.array(ref_mesh_cp.cell_orientations()) - 1
    try:
        df.ALE.move(ref_mesh_cp, u_cg1_ref, annotate=False)
    except:
        df.ALE.move(ref_mesh_cp, u_cg1_ref)
    ref_mesh_cp.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
    post_orient_ref = 2 * np.array(ref_mesh_cp.cell_orientations()) - 1
    # print(np.count_nonzero(post_orient_ref * pre_orient_ref - 1), "flipped cells")
    signs_ref.vector()[:] = post_orient_ref * pre_orient_ref
    signs.interpolate(signs_ref)
    print(np.count_nonzero(signs.vector()[:] - 1))

    # j = df.project(df.det(df.Identity(2) + df.grad(u_cg2)), DG1)
    dg_space = DG4
    j = df.project(df.det(df.Identity(2) + df.grad(u_cg2)), dg_space)
    j_min = df.Function(DG0)
    dm = dg_space.dofmap()
    for i, cell in enumerate(df.cells(mesh)):
        cell_index = cell.index()
        cell_dofs = dm.cell_dofs(cell_index)
        j_min.vector()[[i]] = [np.min(j.vector()[cell_dofs])]


    ref_signs_outfile.write_checkpoint(signs_ref, "signs_ref", k, append=True)
    ref_def_outfile.write_checkpoint(u_cg1_ref, "uh_ref", k, append=True)
    proc_outfile.write_checkpoint(signs, "signs", k, append=True)
    j_outfile.write_checkpoint(j, "j", k, append=True)
    j_min_outfile.write_checkpoint(j_min, "j_min", k, append=True)

