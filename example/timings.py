import dolfin as df
import numpy as np
import tqdm
import pickle 
from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))
import FSIsolver.extension_operator.extension as extension

#msh = df.Mesh()
#infile = df.XDMFFile("Output/Extension/Data/FSIbenchmarkII_data_.xdmf") #TODO: change input data file
#infile.read(msh)
#
#msh_r = df.Mesh(msh)

# load mesh
mesh = df.Mesh()
with df.XDMFFile(str(here.parent) + "/Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mesh)
mvc = df.MeshValueCollection("size_t", mesh, 2)
mvc2 = df.MeshValueCollection("size_t", mesh, 2)
with df.XDMFFile(str(here.parent) + "/Output/Mesh_Generation/facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
with df.XDMFFile(str(here.parent) + "/Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mvc2, "name_to_read")
boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)
domains = df.cpp.mesh.MeshFunctionSizet(mesh,mvc2)

# boundary parts
params = np.load(str(here.parent) + '/Output/Mesh_Generation/params.npy', allow_pickle='TRUE').item()

params["no_slip_ids"] = ["noslip", "obstacle_fluid", "obstacle_solid"]

# subdomains
msh = df.MeshView.create(domains, params["fluid"])
msh_r = df.Mesh(msh)

threshold = 0.001

timings = {}

refinement_levels = 1
datapoints = range(40) #TODO: adapt 5 to number of snapshots you want to average over

df.parameters['allow_extrapolation'] = True
u_bc = df.Function(df.VectorFunctionSpace(msh, "CG", 2))
df.parameters['allow_extrapolation'] = False

file = df.File('./mesh_levels.pvd')

i = 0
infile = df.XDMFFile("Output/Extension/Data/FSIbenchmarkII_data_.xdmf") #TODO: change input data file

while i <= refinement_levels:
    if i != 0:
        msh_r = df.refine(msh_r)

    file << msh_r
    
    ext_ops = {}
    ext_ops["harmonic"] = extension.Harmonic(msh_r)
    ext_ops["biharmonic"] = extension.Biharmonic(msh_r)
    ##ext_ops["learned"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold)# learned
    ext_ops["learned artificial"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold)# learned artificial dataset
    ##ext_ops["learned incremental"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=False)# learned linearized
    ext_ops["learned incremental artificial"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=False)# learned linearized artificial dataset
    ##ext_ops["learned incremental corrected"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=True)# learned linearized corrected
    ext_ops["learned incremental corrected artificial"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=True)# learned linearized corrected artificial dataset
    ##ext_ops["NN corrected"] = extension.TorchExtension(msh_r, "torch_extension/models/yankee", T_switch=0.0, silent=True)
    ext_ops["nncor_art"] = extension.TorchExtension(msh_r, "torch_extension/models/foxtrot", T_switch=0.0, silent=True)

    timings_r = {}
    V = df.VectorFunctionSpace(msh_r, "CG", 2)
    u_bc_r = df.Function(V)
    for j in tqdm.tqdm(ext_ops.keys()):
        for k in tqdm.tqdm(datapoints):
            infile.read_checkpoint(u_bc, "output_biharmonic_ext", k)
            u_bc_r.assign(df.project(u_bc, V))
            if j == "nncor" or "nncor_art":
                print(j)
                u_ext = ext_ops[j].extend(u_bc_r, {"t": 1.0})
            else:
                print(j)
                u_ext = ext_ops[j].extend(u_bc_r)
        timings_r[j] = ext_ops[j].get_timings()
        
    timings["refinment " + str(i)] = timings_r

    i += 1

    with open('Output/Extension/Data/timings.pickle', 'wb') as handle:
        pickle.dump(timings, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load with:
#with open('Output/Extension/Data/timings.pickle', 'rb') as handle:
#    b = pickle.load(handle)