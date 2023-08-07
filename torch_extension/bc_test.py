import dolfin as df
import numpy as np



mesh = df.UnitSquareMesh(4, 4)

V_cg2 = df.VectorFunctionSpace(mesh, "CG", 2)
# V_cg1 = df.FunctionSpace(mesh, "CG", 1)

bc = df.DirichletBC(V_cg2, df.Constant((1.0,2.0)), 'on_boundary')

u = df.Function(V_cg2)

print(u.vector().get_local())

bc.apply(u.vector())

print(u.vector().get_local())

print(u.vector().get_local()[0::2])
print(u.vector().get_local()[1::2])

xy = V_cg2.tabulate_dof_coordinates()
x, y = xy[0::2], xy[1::2]

u_x = u.vector().get_local()[0::2]
u_y = u.vector().get_local()[1::2]

# for i in range(len(x)):
#     print(x[i], u_x[i])

# file = df.File("foo.pvd")
# file << u

inds = np.nonzero(u_x)[0]
print(x[inds])

inds = u_x == 0.0
print(x[inds])