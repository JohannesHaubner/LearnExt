import matplotlib.pyplot as pl
import numpy as np
from pathlib import Path

#times = np.asarray(range(len(displacement)))*deltat

#pl.plot(times, displacement)
#pl.axis([0,15, -0.1, 0.1])
#pl.savefig('./displacement_plot.png')

tmax = 15.0

colors3 = [np.asarray([218, 215, 213])*1./255,
          np.asarray([0, 101, 189])*1./255,
          np.asarray([227, 114, 34])*1./255
          ]

colors4 = [np.asarray([218, 215, 213])*1./255,
          np.asarray([0, 101, 189])*1./255,
          np.asarray([0, 0, 0])*1./255,
          np.asarray([227, 114, 34])*1./255
          ]

def plot_displacement(list, times, str, colors, foldernames):

    for i in range(len(list)):
        #pl.plot(times[i], list[i], linewidth=0.6, label=foldernames[i])
        pl.plot(times[i], list[i], color = colors[i], linewidth=0.6, label=foldernames[i])
    pl.axis([0, tmax, -0.1, 0.1])
    pl.legend(loc='lower left')
    pl.xlabel("time")
    pl.ylabel("y-displacement of tip of the flap")
    pl.savefig(str)
    pl.close()

def plot_determinant(list, times, str, colors, foldernames):

    for i in range(len(list)):
        #pl.plot(times[i], list[i], linewidth=0.6, label=foldernames[i])
        pl.plot(times[i], list[i], color = colors[i], linewidth=0.6, label=foldernames[i])
    pl.axis([0, tmax, -0.1, 1.1])
    pl.legend(loc='lower left')
    pl.xlabel("time")
    pl.ylabel("minimal determinant of deformation gradient")
    pl.axhline(y=0.0, color="black", lw=0.6, alpha=0.8)
    pl.savefig(str)
    pl.close()

def plot_timestep(times, str, colors, foldernames):
    
    times_max = times[0][0] # assume that first time step of first simulation is the maximal timestep
    for i in range(len(times)):
        times_diff = [times[i][k+1] - times[i][k] for k in range(len(times[i])-1)]
        times_mid = [0.5*(times[i][k+1] + times[i][k]) for k in range(len(times[i])-1)]
        #pl.plot(times_mid, times_diff, linewidth=0.6, label=foldernames[i])
        pl.plot(times_mid, times_diff, color=colors[i], linewidth=0.6, label=foldernames[i])
        pl.axis([0, tmax, 0.0, 0.011])
    pl.legend(loc='lower left')
    pl.xlabel("time")
    pl.ylabel("time-step size")
    pl.savefig(str)
    pl.close()

if __name__ == "__main__":


    configs = [
        (["Output/files/biharmonic", "Output/files/harmonic_trafo", "Output/files/harmonic_notrafo"],
         ["biharmonic", "harmonic incremental", "harmonic"]),
        (["Output/FSIbenchmarkII_biharmonic_new", "Output/files/supervised_learn/standard", "Output/files/supervised_learn/incremental", "Output/files/supervised_learn/lintraf_correct"],
         ["biharmonic", "hybrid", "hybrid linear incremental", "hybrid linear incremental corrected"]),
        (["Output/FSIbenchmarkII_biharmonic_new", "Output/files/supervised_learn/artificial_standard", "Output/files/supervised_learn/artificial_incremental", "Output/files/supervised_learn/artificial_lintraf_correct"],
         ["biharmonic", "hybrid", "hybrid linear incremental", "hybrid linear incremental corrected"]),
        (["Output/FSIbenchmarkII_biharmonic_new", "Output/files/supervised_learn/standard", "Output/files/supervised_learn/nncorrect-fsi"],
         ["biharmonic", "hybrid", "NN-corrected"]),
        (["Output/FSIbenchmarkII_biharmonic_new", "Output/files/supervised_learn/artificial_standard", "Output/files/supervised_learn/nncorrect-artificial"],
         ["biharmonic", "hybrid", "NN-corrected"])
    ]

    for iter, (foldernames, names) in enumerate(configs):

        times_list = []
        displacement_list = []
        determinant_list = []

        for i, f in enumerate(foldernames):
            times_list.append(np.loadtxt(f + "/times.txt"))
            displacement_list.append(np.loadtxt(f + "/displacementy.txt"))
            determinant_list.append(np.loadtxt(f + "/determinant.txt"))


        save_dir = Path(f"newfigs/flat")
        save_dir.mkdir(exist_ok=True, parents=True)

        colors = colors3 if len(names) == 3 else colors4

        plot_displacement(displacement_list, times_list, save_dir / f"fsiresults_{iter}_displacement_plot.pdf", colors, names)

        plot_determinant(determinant_list, times_list, save_dir / f"fsiresults_{iter}_determinant_plot.pdf", colors, names)

        # plot_timestep(times_list, save_dir / f"fsiresults_{iter}_timestepsize_plot.pdf", colors, names)
