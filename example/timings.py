import dolfin as df
import tqdm
from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))
import FSIsolver.extension_operator.extension as extension

msh = df.Mesh()
infile = df.XDMFFile("Output/Extension/Data/membrane_data.xdmf")
infile.read(msh)


threshold = 0.001

ext_ops = {}

#ext_ops["harmonic"] = extension.Harmonic(msh)
#ext_ops["biharmonic"] = extension.Biharmonic(msh)
#ext_ops["learned"] = extension.LearnExtension(msh, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold)# learned
#ext_ops["learned_art"] = extension.LearnExtension(msh, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold)# learned artificial dataset
#ext_ops["learned_inc"] = extension.LearnExtension(msh, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=False)# learned linearized
#ext_ops["learned_inc_art"] = extension.LearnExtension(msh, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=False)# learned linearized artificial dataset
#ext_ops["learned_inc_cor"] = extension.LearnExtension(msh, NN_path=str(str(here.parent) + "/example/learned_networks/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=True)# learned linearized corrected
#ext_ops["learned_inc_cor_art"] = extension.LearnExtension(msh, NN_path=str(str(here.parent) + "/example/learned_networks/artificial/trained_network.pkl"), threshold=threshold, incremental=True, incremental_corrected=True)# learned linearized corrected artificial dataset
ext_ops["nncor"] = extension.TorchExtension(msh, "torch_extension/models/yankee", T_switch=0.0, silent=True)
ext_ops["nncor_art"] = extension.TorchExtension(msh, "torch_extension/models/foxtrot", T_switch=0.0, silent=True)

for j in tqdm.tqdm(ext_ops.keys()):
    u_bc = df.Function(df.VectorFunctionSpace(msh, "CG", 2))
    for k in tqdm.tqdm(range(5)):
        infile.read_checkpoint(u_bc, "output_biharmonic_ext", k)
        if j == "nncor" or "nncor_art":
            print(j)
            u_ext = ext_ops[j].extend(u_bc, {"t": 1.0})
        else:
            print(j)
            u_ext = ext_ops[j].extend(u_bc)
    ext_ops[j].get_timings()