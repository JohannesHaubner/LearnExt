import dolfin as df
import numpy as np
import torch
from timeit import default_timer as timer
from pathlib import Path

df.set_log_active(False)

T_tot = 0.0

msh = df.Mesh()
infile = df.XDMFFile("foxtrotRolloutP1.xdmf")
infile.read(msh)

fspace = df.VectorFunctionSpace(msh, "CG", 1)
Fspace = df.VectorFunctionSpace(msh, "CG", 2)
u = df.Function(fspace)
U = df.Function(Fspace)

M = 100
start = timer()
for _ in range(M):
    u_loc = u.vector().get_local()
end = timer()

print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("u.vector().get_local(), CG1: ")
T_tot += (end-start) / M


M = 100
start = timer()
for _ in range(M):
    U_loc = U.vector().get_local()
end = timer()

print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("U.vector().get_local(), CG2: ")


M = 100
start = timer()
for _ in range(M):
    u.vector().set_local(u_loc)
end = timer()

print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("u.vector().set_local(), CG1: ")
T_tot += (end-start) / M

M = 100
start = timer()
for _ in range(M):
    U.vector().set_local(U_loc)
end = timer()

print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("U.vector().set_local(), CG2: ")


from torch_extension.loading import load_model

model = load_model("torch_extension/models/yankee")
x = torch.rand((3935, 8)).float().numpy()

M = 100
start = timer()
for _ in range(M):
    y = torch.tensor(x, dtype=torch.float32)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("torch.tensor(x, dtype=torch.float32): ")
T_tot += (end-start) / M

M = 100
start = timer()
for _ in range(M):
    model(y)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("model(y): ")
T_tot += (end-start) / M

a = torch.rand((3935, 2), dtype=torch.float32).numpy()
b = torch.rand((3935, 1), dtype=torch.float32).numpy()
M = 100
start = timer()
for _ in range(M):
    c = a * b
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("corr_np = corr_np * self.mask_np: ")
T_tot += (end-start) / M

x = torch.rand((3935, 2), dtype=torch.float32)
M = 100
start = timer()
for _ in range(M):
    x.numpy()
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("x.numpy(): ")
T_tot += (end-start) / M

M = 100
start = timer()
for k in range(M):
    infile.read_checkpoint(u, "uh", k)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("infile.read_checkpoint(), CG1: ")

Path("foop1.xmdf").unlink(missing_ok=True); Path("foop1.h5").unlink(missing_ok=True)
outfile = df.XDMFFile("foop1.xdmf")
outfile.write(msh)
M = 100
start = timer()
for k in range(M):
    outfile.write_checkpoint(u, "uh", k, append=True)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("outfile.write_checkpoint(), CG1: ")
Path("foop1.xdmf").unlink(); Path("foop1.h5").unlink(missing_ok=True)

infile = df.XDMFFile("foxtrotRolloutP2.xdmf")
M = 100
start = timer()
for _ in range(M):
    infile.read_checkpoint(U, "uh", k)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("infile.read_checkpoint(), CG2: ")

Path("foop2.xmdf").unlink(missing_ok=True); Path("foop2.h5").unlink(missing_ok=True)
outfile = df.XDMFFile("foop2.xdmf")
outfile.write(msh)
M = 100
start = timer()
for k in range(M):
    outfile.write_checkpoint(U, "uh", k, append=True)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("outfile.write_checkpoint(), CG2: ")
Path("foop2.xdmf").unlink(); Path("foop2.h5").unlink()

from torch_extension.extension import TorchExtension
extension = TorchExtension(msh, model, T_switch=0.5, silent=True)

M = 20
start = timer()
for k in range(M):
    extension.extend(U, params={"t": 0.0})
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("extension.extend(), t=0: ")
T_tot += (end-start) / M

M = 10
start = timer()
for k in range(M):
    extension.extend(U, params={"t": 1.0})
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("extension.extend(), t=1: ")

from torch_extension.extension import TorchExtensionInplace
extension_inpl = TorchExtensionInplace(msh, model, T_switch=0.5, silent=True)
M = 10
start = timer()
for k in range(M):
    extension_inpl.extend(U, params={"t": 1.0})
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("extension.extend(), t=1, inplace: ")

from torch_extension.extension import TorchExtensionInplaceMat
extension_inpl_mat = TorchExtensionInplaceMat(msh, model, T_switch=0.5, silent=True)
extension_inpl_mat.extend(U, params={"t": 1.0})
M = 40
start = timer()
for k in range(M):
    extension_inpl_mat.extend(U, params={"t": 1.0})
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("extension.extend(), t=1, inplace, mat: ")
T_ext = (end - start) / M

M = 100
start = timer()
for k in range(M):
    clm_int = extension.clement_interpolater()
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("clement_interpolater(): ")
T_tot += (end-start) / M


from torch_extension.extension import CG1_vector_plus_grad_to_array_w_coords
M = 100
start = timer()
for k in range(M):
    arr = CG1_vector_plus_grad_to_array_w_coords(u, clm_int)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("CG1_vector_plus_grad_to_array_w_coords: ")
T_tot += (end-start) / M

# harmonic_cg1 = df.interpolate(uh, self.F_cg1)
M = 20
start = timer()
for k in range(M):
    xx = df.interpolate(U, fspace)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("df.interpolate, CG2->CG1: ")

M = 20
start = timer()
for k in range(M):
    xx = df.interpolate(u, Fspace)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("df.interpolate, CG1->CG2: ")

M = 20
start = timer()
for k in range(M):
    U.interpolate(u)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("U.interpolate(u), CG2->CG1: ")

M = 20
start = timer()
for k in range(M):
    u.interpolate(U)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("u.interpolate(U), CG1->CG2: ")


M = 40
start = timer()
for k in range(M):
    extension_inpl_mat.interp_mat_2_1.mult(U.vector(), u.vector())
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("matrix interpolate, CG2->CG1: ")
T_tot += (end-start) / M


M = 40
start = timer()
for k in range(M):
    extension_inpl_mat.interp_mat_1_2.mult(u.vector(), U.vector())
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("matrix interpolate, CG1->CG2: ")
T_tot += (end-start) / M


M = 100
start = timer()
for k in range(M):
    u.vector().set_local(u.vector().get_local())
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("u.vector().set_local(u.vector().get_local()): ")
T_tot += (end-start) / M


M = 100
start = timer()
for k in range(M):
    U = df.Function(Fspace)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("U = df.Function(Fspace), CG2: ")

M = 100
start = timer()
for k in range(M):
    u = df.Function(fspace)
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("u = df.Function(fspace), CG1: ")



bc = df.DirichletBC(Fspace, df.Constant((1.0, 1.0)), "on_boundary")
M = 100
start = timer()
for k in range(M):
    bc.apply(U.vector())
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("bc.apply(U.vector()), CG2")
T_tot += (end-start) / M

new_dofs = np.zeros_like(u.vector().get_local())
M = 100
start = timer()
for k in range(M):
    new_dofs[0::2] += c[:,0]
    new_dofs[1::2] += c[:,1]
end = timer()
print(f"T = {(end-start) / M:.2e}", end="\t# ")
print("new_dofs[0::2] += corr[:,0]; new_dofs[1::2] += corr[:,1], CG1")
T_tot += (end-start) / M

print("-----#-----#-----#-----#-----#-----#-----")

# T_tot = 0.0
# T_tot += 1.28e-02    # extension.extend(), t=0:
# T_tot += 6.09e-05    # matrix interpolate, CG2->CG1:
# T_tot += 4.32e-04    # u.vector().set_local(u.vector().get_local()):
# T_tot += 8.05e-03    # clement_interpolater():
# T_tot += 4.98e-04    # CG1_vector_plus_grad_to_array_w_coords:
# T_tot += 7.22e-06    # torch.tensor(x, dtype=torch.float32):
# T_tot += 1.09e-02    # model(y):
# T_tot += 1.65e-06    # x.numpy():
# T_tot += 3.05e-05    # corr_np = corr_np * self.mask_np: 
# T_tot += 6.94e-06    # u.vector().get_local(), CG1:
# T_tot += 8.99e-06    # new_dofs[0::2] += corr[:,0]; new_dofs[1::2] += corr[:,1], CG1
# T_tot += 4.18e-04    # u.vector().set_local(), CG1: 
# T_tot += 1.41e-04    # matrix interpolate, CG1->CG2:
# T_tot += 3.13e-04    # bc.apply(U.vector()), CG2

# print(f"{T_tot = :.2e}")
# print("T     = 2.73e-02    # extension.extend(), t=1, inplace, mat:")

print(f"{T_tot = :.2e}")
print(f"T_ext = {T_ext:.2e}    # extension.extend(), t=1, inplace, mat:")


