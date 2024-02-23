import numpy as np
from ogs5py import OGS
import pygmsh, meshio
import h5py

from pathlib import Path
here = Path(__file__).parent.resolve()

# resolution
resolution = 0.0005  #0.05 #1 # 0.005 #0.1
resolution2 = 0.000125
resolution3 = 0.000125

# geometric properties
L = 2.5 #2.5 #20            # length of channel
H = 0.41 #0.4 #6           # heigth of channel
c = [0.2, 0.2, 0]  #[0.2, 0.2, 0] #[10, 3, 0]  # position of object
r = 0.05 #0.05 #0.5 # radius of object

d = {}
d["c_1"] = [0.0, 0.005, 0.]
d["c_2"] = [0.005, 0.005, 0.]
d["c_3"] = [0.005, 0.0045, 0.]
d["c_4"] = [0.005, 0.0, 0.0]
d["c_5"] = [0.015, 0., 0.]
d["c_6"] = [0.015, 0.0045, 0.]
d["c_7"] = [0.015, 0.005, 0.]
d["c_8"] = [0.02, 0.005, 0.]
d["c_9"] = [0.02, 0.021, 0.]
d["c_10"] = [0.0, 0.021, 0.]
c = [0.01, 0.005, 0.]
c2 = [0.01, 0.0045, 0.]
c3 = [0.01, 0.0, 0.]

# labels
inflow = 1
outflow = 2
walls = 3
solid_left = 4
solid_right = 5
interface = 6
fluid = 7
solid = 8

params = {"inflow" : inflow,
          "outflow": outflow,
          "noslip": walls,
          "solid_left": solid_left,
          "solid_right": solid_right,
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
np.save(str(here.parent.parent) + '/Output/Mesh_Generation/params_fsi2.npy', params)
np.save(str(here.parent.parent) + '/Output/Mesh_Generation/geom_prop_fsi2.npy', geom_prop)

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

# Add points with finer resolution on left side
points =  []
for i in range(10):
    if i == 1 or i == 2 or i == 5 or i == 6: # or i == 3 or i == 4: 
        points.append(model.add_point(d["c_"+ str(i+1)], mesh_size = resolution2))
    else:
        points.append(model.add_point(d["c_"+ str(i+1)], mesh_size = resolution))
points2 = [model.add_point(c, mesh_size = resolution3), model.add_point(c2, mesh_size=resolution3), model.add_point(c3, mesh_size=resolution)]

# Add lines between all points creating the rectangle
channel_lines = []
for i in range(-1, len(points)-1):
    if i == 1 or i == 5:
        channel_lines.append(model.add_line(points[i+1], points[i]))
        print(points[i])
    elif i == 3:
        channel_lines.append(model.add_line(points[i], points2[2]))
        channel_lines.append(model.add_line(points2[2], points[i+1]))
    else:
        channel_lines.append(model.add_line(points[i], points[i+1]))
        print(points[i])
                 
interface_lines = []
interface_lines.append(model.add_line(points[1], points2[0]))
interface_lines.append(model.add_line(points2[0], points[6]))
interface_lines.append(model.add_line(points[5], points2[1]))
interface_lines.append(model.add_line(points2[1], points[2]))

elastic = model.add_curve_loop([interface_lines[0], interface_lines[1], channel_lines[7], interface_lines[2], interface_lines[3], channel_lines[2]])

# Create a line loop and plane surface for meshing
channel_loop = model.add_curve_loop([channel_lines[0], channel_lines[1], interface_lines[0], interface_lines[1], channel_lines[8], channel_lines[9], channel_lines[10]])
channel_loop3 = model.add_curve_loop([interface_lines[2], interface_lines[3], channel_lines[3], channel_lines[4], channel_lines[5], channel_lines[6]])
plane_surface = model.add_plane_surface(
    channel_loop)
plane_surface2 = model.add_plane_surface(
    elastic)
plane_surface3 = model.add_plane_surface(
    channel_loop3
)

# Call gmsh kernel before add physical entities
model.synchronize()

#volume_marker = 6
model.add_physical([channel_lines[4], channel_lines[5]], "inflow") # mark inflow boundary with 1
model.add_physical([channel_lines[-1]], "outflow") # mark outflow boundary with 2
wall_lines = []
for i in [0, 1, 3, 6, 8, 9]:
    wall_lines.append(channel_lines[i])
model.add_physical(wall_lines, "walls") # mark walls with 3
model.add_physical(channel_lines[2], "solid_left")
model.add_physical(channel_lines[7], "solid_right")
model.add_physical(interface_lines, "interface") # mark interface with 6
model.add_physical([plane_surface, plane_surface3], "fluid") # mark fluid domain with 7
model.add_physical([plane_surface2], "solid") # mark solid domain with 8

geometry.generate_mesh(dim=2)
import gmsh
gmsh.write(str(here.parent.parent) + "/Output/Mesh_Generation/mesh_fsi2.msh")
gmsh.clear()
geometry.__exit__()

import meshio
mesh_from_file = meshio.read(str(here.parent.parent) + "/Output/Mesh_Generation/mesh_fsi2.msh")

import numpy
def create_mesh(mesh: meshio.Mesh, cell_type: str, data_name: str = "name_to_read",
                prune_z: bool = False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
                           data_name: [cell_data]})
    return out_mesh #https://fenicsproject.discourse.group/t/what-is-wrong-with-my-mesh/7504/8

line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(str(here.parent.parent) + "/Output/Mesh_Generation/facet_mesh_fsi2.xdmf", line_mesh)

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(str(here.parent.parent) + "/Output/Mesh_Generation/mesh_triangles_fsi2.xdmf", triangle_mesh)

#mesh = line_mesh
#mesh_boundary = meshio.Mesh(points=mesh.points,
#                                               cells={"line": mesh.get_cells_type("line")},
#                                               cell_data={"name_to_read": [mesh.get_cell_data("gmsh:physical", "line")]})
#meshio.write("../Output/Mesh_Generation/mesh_boundary.xdmf", mesh_boundary)


#model = OGS()
#model.msh.generate("gmsh", geo_object=geom)
#model.msh.show()
