import dolfin as df
import torch
from timeit import default_timer as timer
from pathlib import Path

df.set_log_active(False)

N = 61
p = 2
d = 2
msh = df.UnitSquareMesh(N, N)
V = df.VectorFunctionSpace(msh, "CG", p, d)
u = df.Function(V)

M = 100
start = timer()
for _ in range(M):
    u_loc = u.vector().get_local()
end = timer()

print(f"T = {(end-start) / M:.2e}", end="\t")
print("u.vector().get_local(): ")


from torch_extension.loading import load_model

model = load_model("torch_extension/models/yankee")
x = torch.rand((3935, 8)).float().numpy()

M = 100
start = timer()
for _ in range(M):
    y = torch.tensor(x, dtype=torch.float32)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("torch.tensor(x, dtype=torch.float32): ")

M = 100
start = timer()
for _ in range(M):
    model(y)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("model(y): ")


msh = df.Mesh()
infile = df.XDMFFile("foxtrotRolloutP1.xdmf")
infile.read(msh)

fspace = df.VectorFunctionSpace(msh, "CG", 1)
Fspace = df.VectorFunctionSpace(msh, "CG", 2)
u = df.Function(fspace)
U = df.Function(Fspace)

M = 100
start = timer()
for k in range(M):
    infile.read_checkpoint(u, "uh", k)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("infile.read_checkpoint(), CG1: ")

Path("foop1.xmdf").unlink(missing_ok=True); Path("foop1.h5").unlink(missing_ok=True)
outfile = df.XDMFFile("foop1.xdmf")
outfile.write(msh)
M = 100
start = timer()
for k in range(M):
    outfile.write_checkpoint(u, "uh", k, append=True)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("outfile.write_checkpoint(), CG1: ")
Path("foop1.xdmf").unlink(); Path("foop1.h5").unlink(missing_ok=True)

infile = df.XDMFFile("foxtrotRolloutP2.xdmf")
M = 100
start = timer()
for _ in range(M):
    infile.read_checkpoint(U, "uh", k)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("infile.read_checkpoint(), CG2: ")

Path("foop2.xmdf").unlink(missing_ok=True); Path("foop2.h5").unlink(missing_ok=True)
outfile = df.XDMFFile("foop2.xdmf")
outfile.write(msh)
M = 100
start = timer()
for k in range(M):
    outfile.write_checkpoint(U, "uh", k, append=True)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("outfile.write_checkpoint(), CG2: ")
Path("foop2.xdmf").unlink(); Path("foop2.h5").unlink()

from torch_extension.extension import TorchExtension
extension = TorchExtension(msh, model, T_switch=0.5, silent=True)

M = 20
start = timer()
for k in range(M):
    extension.extend(U, params={"t": 0.0})
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("extension.extend(), t=0: ")

M = 10
start = timer()
for k in range(M):
    extension.extend(U, params={"t": 1.0})
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("extension.extend(), t=1: ")

from torch_extension.extension import TorchExtensionInplace
extension_inpl = TorchExtensionInplace(msh, model, T_switch=0.5, silent=True)
M = 10
start = timer()
for k in range(M):
    extension.extend(U, params={"t": 1.0})
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("extension.extend(), t=1, inplace: ")


M = 100
start = timer()
for k in range(M):
    clm_int = extension.clement_interpolater()
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("clement_interpolater(): ")


from torch_extension.extension import CG1_vector_plus_grad_to_array_w_coords
M = 100
start = timer()
for k in range(M):
    arr = CG1_vector_plus_grad_to_array_w_coords(u, clm_int)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("CG1_vector_plus_grad_to_array_w_coords: ")

# harmonic_cg1 = df.interpolate(uh, self.F_cg1)
M = 20
start = timer()
for k in range(M):
    xx = df.interpolate(U, fspace)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("df.interpolate, CG2->CG1: ")

M = 20
start = timer()
for k in range(M):
    xx = df.interpolate(u, Fspace)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("df.interpolate, CG1->CG2: ")

M = 20
start = timer()
for k in range(M):
    U.interpolate(u)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("U.interpolate(u), CG2->CG1: ")

M = 20
start = timer()
for k in range(M):
    u.interpolate(U)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("u.interpolate(U), CG1->CG2: ")


# self.u_cg1.vector().set_local(harmonic_cg1.vector().get_local())
M = 100
start = timer()
for k in range(M):
    u.vector().set_local(u.vector().get_local())
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("u.vector().set_local(u.vector().get_local()): ")


M = 100
start = timer()
for k in range(M):
    U = df.Function(Fspace)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("U = df.Function(Fspace), CG2: ")

M = 100
start = timer()
for k in range(M):
    u = df.Function(fspace)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t")
print("u = df.Function(fspace), CG1: ")


