# https://github.com/MiroK/gmshnics
from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
import dolfin as df
import numpy as np
import gmsh

def create_mesh():
    '''Mesh and entity function for the fluid & solid subdomain'''
    gmsh.initialize()
    # resolution
    resolution = 0.025  #0.05 #1 # 0.005 #0.1

    # geometric properties
    L = 2.5 #2.5 #20            # length of channel
    H = 0.41 #0.4 #6           # heigth of channel
    c = [0.2, 0.2, 0]  #[0.2, 0.2, 0] #[10, 3, 0]  # position of object
    r = 0.05 #0.05 #0.5 # radius of object

    # labels
    inflow = 1
    outflow = 2
    walls = 3
    noslipobstacle = 4
    obstacle = 5
    interface = 6
    fluid = 7
    solid = 8
    flag_tip = 9


    params = {"inflow" : inflow,
              "outflow": outflow,
              "noslip": walls,
              "obstacle_solid": noslipobstacle,
              "obstacle_fluid": obstacle,
              "interface": interface,
              'flag_tip': flag_tip,
              "mesh_parts": True,
              "fluid": fluid,
              "solid": solid
              }

    vol = L*H
    vol_D_minus_obs = vol - np.pi*r*r
    geom_prop = {"barycenter_hold_all_domain": [0.5*L, 0.5*H],
                 "volume_hold_all_domain": vol,
                 "volume_D_minus_obstacle": vol_D_minus_obs,
                 "volume_obstacle": np.pi*r*r,
                 "length_pipe": L,
                 "heigth_pipe": H,
                 "barycenter_obstacle": [c[0], c[1]],
                 }

    model = gmsh.model
    fac = model.occ

    # Add circle
    pc = fac.addPoint(*c)
    sin = 0.5 # sin(30°)
    cos = np.sqrt(3)/2 # cos(30°)
    pc0 = fac.addPoint(*c)
    pc1 = fac.addPoint(c[0]-r, c[1], 0, 0.2*resolution)
    pc2 = fac.addPoint(0.24898979485, 0.21, 0, 0.2*resolution)
    pc3 = fac.addPoint(0.24898979485, 0.19,0, 0.2*resolution)
    circle1 = fac.addCircleArc(pc2, pc0, pc1)
    circle2 = fac.addCircleArc(pc1, pc0, pc3)
    circle3 = fac.addCircleArc(pc2, pc0, pc3)

    # Add elastic flag
    pf1 = fac.addPoint(0.6, 0.21, 0, 0.2*resolution)
    pf2 = fac.addPoint(0.6, 0.19, 0, 0.2*resolution)
    fl1 = fac.addLine(pc3, pf2)
    fl2 = fac.addLine(pf2, pf1)
    fl3 = fac.addLine(pf1, pc2)

    # obstacle
    obstacle_curves = [fl1, fl2, fl3, circle1, circle2]
    obstacle_ = fac.addCurveLoop(obstacle_curves)

    flag_curves = [circle3, fl1, fl2, fl3]
    flag = fac.addCurveLoop(flag_curves)

    # Add points with finer resolution on left side
    points = [fac.addPoint(0, 0, 0, resolution),
              fac.addPoint(L, 0, 0, resolution), #5*resolution
              fac.addPoint(L, H, 0, resolution), #5*resolution
              fac.addPoint(0, H, 0, resolution)]

    # Add lines between all points creating the rectangle
    channel_lines = [fac.addLine(points[i], points[i+1])
                     for i in range(-1, len(points)-1)]

    # Create a line loop and plane surface for meshing
    channel_loop = fac.addCurveLoop(channel_lines)
    plane_surface = fac.addPlaneSurface([channel_loop, obstacle_])
    plane_surface2 = fac.addPlaneSurface([flag])

    fac.synchronize()

    model.mesh.generate(2)

    model.addPhysicalGroup(1, [channel_lines[0]], params['inflow']) 
    model.addPhysicalGroup(1, [channel_lines[2]], params['outflow']) 
    model.addPhysicalGroup(1, [channel_lines[1], channel_lines[3]], params['noslip'])
    model.addPhysicalGroup(1, [circle3], params['obstacle_solid'])
    model.addPhysicalGroup(1, [circle1, circle2], params['obstacle_fluid']) 
    model.addPhysicalGroup(1, [fl1, fl3], params['interface']) 
    model.addPhysicalGroup(1, [fl2], params['flag_tip']) 

    model.addPhysicalGroup(2, [plane_surface], params['fluid']) 
    model.addPhysicalGroup(2, [plane_surface2], params['solid']) 

    nodes, topologies = msh_gmsh_model(model, dim=2)
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    # Representation of physical volumes in terms of bounding surfaces
    mappings = {fluid: np.unique(np.hstack([model.getPhysicalGroupsForEntity(1, e)
                                            for e in channel_lines + obstacle_curves])),
                solid: np.unique(np.hstack([model.getPhysicalGroupsForEntity(1, e)
                                            for e in flag_curves]))}

    gmsh.finalize()
    
    return mesh, entity_functions, mappings, params, geom_prop


def translate_entity_f(parent_mesh, source_f, child_mesh, tags):
    '''Create entity function on child mesh'''
    child2parent = parent_mesh.data().array('parent_vertex_indices', 0)
    vertex_map = {v: k for k, v in enumerate(child2parent)}

    edim = source_f.dim()

    _, e2v = child_mesh.init(edim, 0), child_mesh.topology()(edim, 0)

    child_f = df.MeshFunction('size_t', child_mesh, edim, 0)
    child_entities = {tuple(sorted(e.entities(0))): e.index() for e in df.SubsetIterator(child_f, 0)}
    for tag in tags:
        for parent_entity in df.SubsetIterator(source_f, tag):
            parent_vertices = parent_entity.entities(0)
            # Now encode them in child
            child_vertices = tuple(sorted([vertex_map[pv] for pv in parent_vertices]))
            # Look for the matching child entity
            child_entity = child_entities[child_vertices]
            child_f[child_entity] = tag
    return child_f


def translate_function(from_u, to_facet_f, from_facet_f, shared_tags, to_u=None, tol_=1E-12):
    '''
    If tu_u and from_u are 2 functions on different domains that share 
    facets we transfer boundary data to to_u from from_u.
    '''
    if to_u is None:
        to_u = df.Function(df.FunctionSpace(to_facet_f.mesh(), from_u.ufl_element()))

    shape = to_u.ufl_shape    
    assert shape == from_u.ufl_shape
    assert len(shape) == 1  # We will only use this for vector spaces
    
    assert shared_tags
    assert to_facet_f.dim() == from_facet_f.dim()

    Vto = to_u.function_space()
    assert Vto.mesh().id() == to_facet_f.mesh().id()
    assert Vto.mesh().topology().dim() - 1 == to_facet_f.dim()

    Vfrom = from_u.function_space()
    assert Vfrom.mesh().id() == from_facet_f.mesh().id()
    assert Vfrom.mesh().topology().dim() - 1 == from_facet_f.dim()

    assert Vfrom.ufl_element() == Vto.ufl_element()

    to_dofs, from_values = [], []
    for dim in range(shape[0]):
        for tag in shared_tags:
            from_u_dim = df.Function(Vfrom.sub(dim).collapse())
            df.assign(from_u_dim, from_u.sub(dim))
            bc_from = df.DirichletBC(Vfrom.sub(dim), from_u_dim, from_facet_f, tag)
            
            this_from_dofs = list(bc_from.get_boundary_values().keys())
            from_values.extend(bc_from.get_boundary_values().values())
            
            # Bc dofs need to find the right permutation ...
            bc_to = df.DirichletBC(Vto.sub(dim), df.Constant(0), to_facet_f, tag)
            this_to_dofs = list(bc_to.get_boundary_values().keys())

            # ... based on position of dofs
            x = Vfrom.tabulate_dof_coordinates()
            y = Vto.tabulate_dof_coordinates()[this_to_dofs]

            for from_dof in this_from_dofs:
                dist = np.linalg.norm(y - x[from_dof], 2, axis=1)
                assert len(dist) == len(y)
                match_index = np.argmin(dist)
                assert dist[match_index] < tol_, (dist[match_index], dim, tag)

                to_dofs.append(this_to_dofs[match_index])

    to_values = to_u.vector().get_local()
    to_values[to_dofs] = from_values

    to_u.vector().set_local(to_values)

    return to_u

# ---------+---------+---------+---------+---------+---------+---------+---------+---------+---------

if __name__ == '__main__':
    from pathlib import Path

    mesh, entity_fs, mapping, params, _ = create_mesh()
    # Submeshes
    cell_f = entity_fs[mesh.topology().dim()]

    fluid_mesh = df.SubMesh(mesh, cell_f, params['fluid'])
    solid_mesh = df.SubMesh(mesh, cell_f, params['solid'])

    facet_f = entity_fs[mesh.topology().dim()-1]
    
    fluid_boundaries = translate_entity_f(fluid_mesh, facet_f, fluid_mesh, mapping[params['fluid']])
    solid_boundaries = translate_entity_f(solid_mesh, facet_f, solid_mesh, mapping[params['solid']])

    mesh_dir = Path("gravity_driven_test") / "data" / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    with df.HDF5File(mesh.mpi_comm(), str(mesh_dir / 'fluid.h5'), 'w') as h5:
        h5.write(fluid_mesh, 'mesh')
        h5.write(fluid_boundaries, 'boundaries')

    with df.HDF5File(mesh.mpi_comm(), str(mesh_dir / 'solid.h5'), 'w') as h5:
        h5.write(solid_mesh, 'mesh')
        h5.write(solid_boundaries, 'boundaries')

    f = df.Expression(('x[0]+2*x[1]', 'x[1]-2*x[0]'), degree=1)
    # Suppose we solve on solid and want to represent that function as data
    # for fluid
    VS = df.VectorFunctionSpace(solid_mesh, 'CG', 2)
    uS = df.interpolate(f, VS)

    interface_dofs = set(mapping[params['fluid']]) & set(mapping[params['solid']])
    uF = translate_function(from_u=uS,
                            to_facet_f=fluid_boundaries,
                            from_facet_f=solid_boundaries,
                            shared_tags=interface_dofs)

    VF = uF.function_space()
    x = VF.tabulate_dof_coordinates()
    # So now on iface_dofs
    for tag in interface_dofs:
        bc = df.DirichletBC(VF, uF, fluid_boundaries, tag)

        bc_dofs = bc.get_boundary_values().keys()
        error = max(np.linalg.norm(uF(x[dof]) - f(x[dof])) for dof in bc_dofs)
        assert error < 1E-13

