import dolfin as df
import numpy as np

from pathlib import Path

if Path("membrane_test/data/extended/harmonic.xdmf").exists():
    test_path = Path("membrane_test/data/extended/harmonic.xdmf")
else:
    test_path = Path("membrane_test/data/Data_MoF/membrane_test_p1.xdmf")
print(str(test_path))

mesh = df.Mesh()
test_file = df.XDMFFile(str(test_path))
test_file.read(mesh)


print(f"{mesh.num_vertices() = }")
V = df.VectorFunctionSpace(mesh, "CG", 1)
u = df.Function(V)


from FSIsolver.extension_operator.extension import LearnExtension

ks = [20, 120, 272]

mesh0 = df.Mesh(mesh)
test_file.read_checkpoint(u, "uh", ks[0])
mesh0.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
pre_orient0 = np.array(mesh0.cell_orientations())
try:
    df.ALE.move(mesh0, u, annotate=False)
except:
    df.ALE.move(mesh0, u)
mesh0.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
post_orient0 = np.array(mesh0.cell_orientations())
print(np.count_nonzero(post_orient0 - pre_orient0), "flipped cells")


mesh1 = df.Mesh(mesh)
test_file.read_checkpoint(u, "uh", ks[1])
mesh1.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
pre_orient1 = np.array(mesh1.cell_orientations())
try:
    df.ALE.move(mesh1, u, annotate=False)
except:
    df.ALE.move(mesh1, u)
mesh1.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
post_orient1 = np.array(mesh1.cell_orientations())
print(np.count_nonzero(post_orient1 - pre_orient1), "flipped cells")


mesh2 = df.Mesh(mesh)
test_file.read_checkpoint(u, "uh", ks[2])
mesh2.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
pre_orient2 = np.array(mesh2.cell_orientations())
try:
    df.ALE.move(mesh2, u, annotate=False)
except:
    df.ALE.move(mesh2, u)
mesh2.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
post_orient2 = np.array(mesh2.cell_orientations())
print(np.count_nonzero(post_orient2 - pre_orient2), "flipped cells")


