import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import os


file_path = Path("membrane_test/data/Data_MoF/membrane_test.xdmf")

mesh = df.Mesh()
infile = df.XDMFFile(str(file_path))
infile.read(mesh)

x = mesh.coordinates()

V = df.VectorFunctionSpace(mesh, "CG", 2)
u = df.Function(V)
infile.read_checkpoint(u, "output_biharmonic_ext", 150)
uh_flat = u.compute_vertex_values()

uh = np.column_stack((uh_flat[:len(uh_flat)//2], uh_flat[len(uh_flat)//2:]))

plt.figure()
plt.plot((x+uh)[:,0], (x+uh)[:,1], 'ko', ms=1)
plt.savefig("membrane_test/data/loading_test_old.pdf")

infile.close()


file_path = Path("membrane_test/data/Data_MoF/membrane_test_p2.xdmf")

mesh = df.Mesh()
infile = df.XDMFFile(str(file_path))
infile.read(mesh)

x = mesh.coordinates()

V = df.VectorFunctionSpace(mesh, "CG", 2)
u = df.Function(V)
infile.read_checkpoint(u, "uh", 150)
uh_flat = u.compute_vertex_values()

uh = np.column_stack((uh_flat[:len(uh_flat)//2], uh_flat[len(uh_flat)//2:]))

plt.figure()
plt.plot((x+uh)[:,0], (x+uh)[:,1], 'ko', ms=1)
plt.savefig("membrane_test/data/loading_test_treated_p2.pdf")

infile.close()


file_path = Path("membrane_test/data/Data_MoF/membrane_test_p1.xdmf")

mesh = df.Mesh()
infile = df.XDMFFile(str(file_path))
infile.read(mesh)

x = mesh.coordinates()

V = df.VectorFunctionSpace(mesh, "CG", 1)
u = df.Function(V)
infile.read_checkpoint(u, "uh", 150)
uh_flat = u.compute_vertex_values()

uh = np.column_stack((uh_flat[:len(uh_flat)//2], uh_flat[len(uh_flat)//2:]))

plt.figure()
plt.plot((x+uh)[:,0], (x+uh)[:,1], 'ko', ms=1)
plt.savefig("membrane_test/data/loading_test_treated_p1.pdf")

infile.close()



