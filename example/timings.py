import dolfin as df
import tqdm
import pickle 
from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))
import FSIsolver.extension_operator.extension as extension

msh = df.Mesh()
infile = df.XDMFFile("Output/Extension/Data/membrane_data.xdmf") #TODO: change input data file
infile.read(msh)

msh_r = df.Mesh(msh)


threshold = 0.001

timings = {}

refinement_levels = 1
datapoints = range(5) #TODO: adapt 5 to number of snapshots you want to average over

df.parameters['allow_extrapolation'] = True
u_bc = df.Function(df.VectorFunctionSpace(msh, "CG", 2))
df.parameters['allow_extrapolation'] = False

file = df.File('./mesh_levels.pvd')

i = 0

while i <= refinement_levels:
    if i != 0:
        msh_r = df.refine(msh_r)

    file << msh_r
    
    ext_ops = {}
    ext_ops["harmonic"] = extension.Harmonic(msh_r)
    ext_ops["biharmonic"] = extension.Biharmonic(msh_r)
    #ext_ops["learned"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold)# learned
    ext_ops["learned artificial"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold)# learned artificial dataset
    #ext_ops["learned incremental"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=False)# learned linearized
    ext_ops["learned incremental artificial"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=False)# learned linearized artificial dataset
    #ext_ops["learned incremental corrected"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=True)# learned linearized corrected
    ext_ops["learned incremental corrected artificial"] = extension.LearnExtension(msh_r, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=True)# learned linearized corrected artificial dataset
    #ext_ops["NN corrected"] = extension.TorchExtension(msh_r, "torch_extension/models/yankee", T_switch=0.0, silent=True)
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