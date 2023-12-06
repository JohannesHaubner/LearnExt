import dolfin as df
import numpy as np

from pathlib import Path

test_path = Path("gravity_driven_test/data/max_deformations/harmonic.xdmf")

mesh = df.Mesh()
test_file = df.XDMFFile(str(test_path))
test_file.read(mesh)

print(mesh.ordered())

print(mesh.num_vertices())
V = df.VectorFunctionSpace(mesh, "CG", 1)
u = df.Function(V)


mesh0 = df.Mesh(mesh)
test_file.read_checkpoint(u, "uh", 0)
mesh0.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
pre_orient0 = np.array(mesh0.cell_orientations())
print(type(mesh0))
print(type(u))
print(u.ufl_element())
df.ALE.move(mesh0, u)
mesh0.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
post_orient0 = np.array(mesh0.cell_orientations())
print(np.count_nonzero(post_orient0 - pre_orient0), "flipped cells")


mesh1 = df.Mesh(mesh)
test_file.read_checkpoint(u, "uh", 1)
mesh1.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
pre_orient1 = np.array(mesh1.cell_orientations())
df.ALE.move(mesh1, u)
mesh1.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
post_orient1 = np.array(mesh1.cell_orientations())
print(np.count_nonzero(post_orient1 - pre_orient1), "flipped cells")


mesh2 = df.Mesh(mesh)
test_file.read_checkpoint(u, "uh", 2)
mesh2.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
pre_orient2 = np.array(mesh2.cell_orientations())
df.ALE.move(mesh2, u)
mesh2.init_cell_orientations(df.Constant((0.0, 0.0, 1.0)))
post_orient2 = np.array(mesh2.cell_orientations())
print(np.count_nonzero(post_orient2 - pre_orient2), "flipped cells")

