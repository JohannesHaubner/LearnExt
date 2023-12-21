import matplotlib.pyplot as pl
import numpy as np

#times = np.asarray(range(len(displacement)))*deltat

#pl.plot(times, displacement)
#pl.axis([0,15, -0.1, 0.1])
#pl.savefig('./displacement_plot.png')

tmax = 0.8 # 15

colors = [np.asarray([218, 215, 213])*1./255,
          np.asarray([0, 101, 189])*1./255,
        #   np.asarray([0, 0, 0])*1./255,
          np.asarray([227, 114, 34])*1./255
          ]

def plot_displacement(list, times, str, colors, foldernames):
    if False: #len(list)>2:
        colors = [(len(list)-i)/(len(list))*colors[0] + i/(len(list))*colors[1] for i in range(len(list)+1)]
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
    if False: #len(list) > 2:
        colors = [(len(list)-i)/(len(list))*colors[0] + i/(len(list))*colors[1] for i in range(len(list)+1)]
    for i in range(len(list)):
        #pl.plot(times[i], list[i], linewidth=0.6, label=foldernames[i])
        pl.plot(times[i], list[i], color = colors[i], linewidth=0.6, label=foldernames[i])
    pl.axis([0, tmax, -0.1, 1.1])
    pl.legend(loc='lower left')
    pl.xlabel("time")
    pl.ylabel("minimal determinant of deformation gradient")
    # pl.axhline(y=0.0, color="black", lw=0.6, alpha=0.8)
    pl.savefig(str)
    pl.close()

def plot_timestep(times, str, colors, foldernames):
    if False: #len(times) > 1:
        colors = [(len(times) - i) / (len(times)) * colors[0] + i / (len(times)) * colors[1] for i in
                  range(len(times)+1)]
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
    foldernames = ["biharmonic", "supervised_learn/standard", "supervised_learn/incremental", "supervised_learn/lintraf_correct"] # ["biharmonic", "harmonic_trafo", "harmonic_notrafo"] #
    names =  ["biharmonic", "learned", "learned linear incremental", "learned linear incremental corrected"] #["biharmonic", "harmonic incremental", "harmonic"]
    # foldernames = ["biharmonic", "supervised_learn/standard", "../../TorchOutput/dataanalysis/learnext_dataset/yankee"]
    foldernames = ["biharmonic", "supervised_learn/standard", "../../TorchOutput/dataanalysis/artificial/foxtrot"]
    names = ["biharmonic", "hybrid PDE-NN", "NN-corrected harmonic"]

    times_list = []
    displacement_list = []
    determinant_list = []

    for i in foldernames:
        str = "../Output/files/" + i
        str = "Output/files/" + i
        times_list.append(np.loadtxt(str + "/times.txt"))
        displacement_list.append(np.loadtxt(str + "/displacementy.txt"))
        determinant_list.append(np.loadtxt(str + "/determinant.txt"))

    # plot_displacement(displacement_list, times_list,
    #                   '../Output/visualizations/displacement_plot.pdf', colors, names)
    plot_displacement(displacement_list, times_list,
                     'Output/visualizations/artificial/foxtrot_2/displacement_plot.pdf', colors, names)

    # plot_determinant(determinant_list, times_list,
    #                  '../Output/visualizations/determinant_plot.pdf', colors, names)
    plot_determinant(determinant_list, times_list,
                     'Output/visualizations/artificial/foxtrot_2/determinant_plot.pdf', colors, names)

    # plot_timestep(times_list,
    #                  '../Output/visualizations/timestepsize_plot.pdf', colors, names)
    plot_timestep(times_list,
                     'Output/visualizations/artificial/foxtrot_2/timestepsize_plot.pdf', colors, names)
