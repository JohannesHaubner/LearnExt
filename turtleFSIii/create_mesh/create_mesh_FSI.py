import numpy as np
from ogs5py import OGS
import pygmsh, meshio
import h5py

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

params = {"inflow" : inflow,
          "outflow": outflow,
          "noslip": walls,
          "obstacle_solid": noslipobstacle,
          "obstacle_fluid": obstacle,
          "interface": interface,
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
             "barycenter_obstacle": [ c[0], c[1]],
             }
np.save('../../Output/Mesh_Generation/params.npy', params)
np.save('../../Output/Mesh_Generation/geom_prop.npy', geom_prop)

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()
# Add circle
pc = model.add_point(c)
sin = 0.5 # sin(30°)
cos = np.sqrt(3)/2 # cos(30°)
pc0 = model.add_point(c)
pc1 = model.add_point((c[0]-r, c[1], 0), mesh_size=0.2*resolution)
pc2 = model.add_point((0.24898979485, 0.21, 0), mesh_size=0.2*resolution)
pc3 = model.add_point((0.24898979485, 0.19,0), mesh_size=0.2*resolution)
circle1 = model.add_circle_arc(pc2, pc0, pc1)
circle2 = model.add_circle_arc(pc1, pc0, pc3)
circle3 = model.add_circle_arc(pc2, pc0, pc3)

# Add elastic flag
pf1 = model.add_point((0.6, 0.21, 0), mesh_size=0.2*resolution)
pf2 = model.add_point((0.6, 0.19, 0), mesh_size=0.2*resolution)
fl1 = model.add_line(pc3, pf2)
fl2 = model.add_line(pf2, pf1)
fl3 = model.add_line(pf1, pc2)

# obstacle
obstacle = model.add_curve_loop([fl1, fl2, fl3, circle1, circle2])
flag = model.add_curve_loop([circle3, fl1, fl2, fl3])

# Add points with finer resolution on left side
points = [model.add_point((0, 0, 0), mesh_size=resolution),
          model.add_point((L, 0, 0), mesh_size=resolution), #5*resolution
          model.add_point((L, H, 0), mesh_size=resolution), #5*resolution
          model.add_point((0, H, 0), mesh_size=resolution)]

# Add lines between all points creating the rectangle
channel_lines = [model.add_line(points[i], points[i+1])
                 for i in range(-1, len(points)-1)]

# Create a line loop and plane surface for meshing
channel_loop = model.add_curve_loop(channel_lines)
plane_surface = model.add_plane_surface(
    channel_loop, holes=[obstacle])
plane_surface2 = model.add_plane_surface(
    flag)

# Call gmsh kernel before add physical entities
model.synchronize()

volume_marker = 6
model.add_physical([channel_lines[0]], "inflow") # mark inflow boundary with 1
model.add_physical([channel_lines[2]], "outflow") # mark outflow boundary with 2
model.add_physical([channel_lines[1], channel_lines[3]], "walls") # mark walls with 3
model.add_physical([circle3], "noslip_obstacle")
model.add_physical([circle1, circle2], "obstacle") # mark obstacle with 4
model.add_physical([fl1, fl2, fl3], "interface") # mark interface with 5
model.add_physical([plane_surface], "fluid") # mark fluid domain with 6
model.add_physical([plane_surface2], "solid") # mark solid domain with 7

geometry.generate_mesh(dim=2)
import gmsh
gmsh.write("mesh.msh")
gmsh.clear()
geometry.__exit__()

import meshio
mesh_from_file = meshio.read("mesh.msh")

import numpy
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh

line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write("../../Output/Mesh_Generation/facet_mesh.xdmf", line_mesh)

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write("../../Output/Mesh_Generation/mesh_triangles.xdmf", triangle_mesh)

#mesh = line_mesh
#mesh_boundary = meshio.Mesh(points=mesh.points,
#                                               cells={"line": mesh.get_cells_type("line")},
#                                               cell_data={"name_to_read": [mesh.get_cell_data("gmsh:physical", "line")]})
#meshio.write("../Output/Mesh_Generation/mesh_boundary.xdmf", mesh_boundary)


#model = OGS()
#model.msh.generate("gmsh", geo_object=geom)
#model.msh.show()
