import dolfin as df
import numpy as np

from pathlib import Path
import os

from typing import Any
Mesh = Any

class Problem:


    def __init__(self, l: float, **kwargs):

        self.l = df.Constant(l)

        self.rho = kwargs["rho"] if "rho" in kwargs.keys() else 1e3 # kg m^-3
        
        if "mu" in kwargs.keys() and "lambda" in kwargs.keys():
            self.mu, self.lambda_ = kwargs["mu"], kwargs["lambda"]
        elif "E" in kwargs.keys() and "nu" in kwargs.keys():
            E, nu = kwargs["E"], kwargs["nu"]
            self.mu = 0.5 * E / (1 + nu)
            self.lambda_ = nu * E / ( (1 + nu)*(1 - 2*nu) )
        else:
            E, nu = 1.4e6, 0.4 # kg m^-1 s^-2, 1
            self.mu = 0.5 * E / (1 + nu)
            self.lambda_ = nu * E / ( (1 + nu)*(1 - 2*nu) )
        
        self.rho: float; self.mu: float; self.lambda_: float

        if "GRAV_TEST_MESH_PATH" in os.environ:
            self.mesh_path = Path(os.environ["GRAV_TEST_MESH_PATH"])
        else:
            self.mesh_path = Path("gravity_driven_test/data/mesh")

        self.mesh, self.boundaries = self.load_mesh()
        self.fluid_mesh, self.fluid_boundaries, \
            self.interface_tags, self.zero_displacement_tags = self.load_fluid_mesh()
        
        return
    
    def load_mesh(self):

        solid_mesh = df.Mesh()
        with df.HDF5File(solid_mesh.mpi_comm(), str(self.mesh_path / 'solid.h5'), 'r') as h5:
            h5.read(solid_mesh, 'mesh', False)

        tdim = solid_mesh.topology().dim()
        solid_boundaries = df.MeshFunction('size_t', solid_mesh, tdim-1, 0)
        with df.HDF5File(solid_mesh.mpi_comm(), str(self.mesh_path / 'solid.h5'), 'r') as h5:
            h5.read(solid_boundaries, 'boundaries')

            # ----6----
            # 4       | 9
            # ----6----

        return solid_mesh, solid_boundaries
    
    def load_fluid_mesh(self):


        fluid_mesh = df.Mesh()
        with df.HDF5File(fluid_mesh.mpi_comm(), str(self.mesh_path / 'fluid.h5'), 'r') as h5:
            h5.read(fluid_mesh, 'mesh', False)

        tdim = fluid_mesh.topology().dim()
        fluid_boundaries = df.MeshFunction('size_t', fluid_mesh, tdim-1, 0)
        with df.HDF5File(fluid_mesh.mpi_comm(), str(self.mesh_path / 'fluid.h5'), 'r') as h5:
            h5.read(fluid_boundaries, 'boundaries')

        fluid_tags = set(fluid_boundaries.array()) - set((0, ))
        iface_tags = {6, 9}
        zero_displacement_tags = fluid_tags - iface_tags

        return fluid_mesh, fluid_boundaries, iface_tags, zero_displacement_tags


    def elasticity_test(self, save_dir: os.PathLike | None = None):
        save_dir = Path(save_dir) if save_dir is not None else save_dir

        # ----6----
        # 4       | 9
        # ----6----

        displacement_bcs = {4: df.Constant((0, 0))}
        volume_load = df.Constant((0.0, 0.8))
        surface_load = {6: df.Constant((0.0, 0.0)), 9: df.Constant((0.0, 0.0))}

        mesh = self.boundaries.mesh()

        rho = df.Constant(self.rho)
        mu = df.Constant(self.mu)
        lambda_ = df.Constant(self.lambda_)

        V = df.VectorFunctionSpace(mesh, "CG", 2)
        u, v = df.TrialFunction(V), df.TestFunction(V)

        eps = lambda v: df.sym(df.grad(v))
        a = 2*mu*df.inner(eps(u), eps(v))*df.dx + lambda_*df.inner(df.div(u), df.div(v))*df.dx
        L = rho*df.inner(volume_load, v)*df.dx

        ds = df.Measure('ds', domain=mesh, subdomain_data=self.boundaries)
        for tag, surface_force in surface_load.items():
            L += df.inner(surface_force, v)*ds(tag)

        bcs = [df.DirichletBC(V, value, self.boundaries, tag)
            for tag, value in displacement_bcs.items()]
        
        uh = df.Function(V)
        df.solve(a == L, uh, bcs)



        # Represent the solid data on fluid mesh
        from meshing import translate_function

        uh_fluid = translate_function(from_u=uh,
                                    from_facet_f=self.boundaries,
                                    to_facet_f=self.fluid_boundaries,
                                    shared_tags=self.interface_tags)

        V = df.VectorFunctionSpace(self.fluid_mesh, 'CG', 2)
        u, v = df.TrialFunction(V), df.TestFunction(V)

        a = df.inner(df.grad(u), df.grad(v))*df.dx
        L = df.inner(df.Constant((0, )*len(u)), v)*df.dx
        # Those from solid
        bcs = [df.DirichletBC(V, uh_fluid, self.fluid_boundaries, tag) for tag in self.interface_tags]
        # The rest is fixed
        null = df.Constant((0, )*len(u))
        bcs.extend([df.DirichletBC(V, null, self.fluid_boundaries, tag) for tag in self.zero_displacement_tags])

        uh_f = df.Function(V)
        df.solve(a == L, uh_f, bcs)

        assert df.norm(uh) > 1e-6
        assert df.norm(uh_f) > 1e-6

        if save_dir is not None:
            (save_dir / "solid_elast.xdmf").unlink(missing_ok=True)
            with df.XDMFFile(str(save_dir / "solid_elast.xdmf")) as xd:
                xd.write(mesh)
                xd.write_checkpoint(uh, "uh", 0)

            (save_dir / "fluid_extend.xdmf").unlink(missing_ok=True)
            with df.XDMFFile(str(save_dir / "fluid_extend.xdmf")) as xd:
                xd.write(self.fluid_mesh)
                xd.write_checkpoint(uh_f, "uh_f", 0)

        return


def main():

    problem = Problem(3.0)
    problem.elasticity_test(save_dir="gravity_driven_test/data/elast_test")

    return

if __name__ == "__main__":
    main()
