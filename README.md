# TurtleFSIii
TurtleFSI.ii implements a fluid-structure interaction solver


Requires a recent master version of dolfin with MeshView support. It might require the changes propsed in https://bitbucket.org/fenics-project/dolfin/issues/1123/assemble-on-mixed-meshview-forms-returns.
Moreover, it also requires a dolfin-adjoint version which supports the changes of git+https://github.com/dolfin-adjoint/pyadjoint.git@constant-adjfloat.
