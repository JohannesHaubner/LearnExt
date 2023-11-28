import matplotlib.pyplot as plt
import numpy as np

t = np.loadtxt("TorchOutput/dataanalysis/times.txt")
disp = np.loadtxt("TorchOutput/dataanalysis/displacementy.txt")
det = np.loadtxt("TorchOutput/dataanalysis/determinant.txt")


print(f"{t = }")
print(f"{disp = }")
print(f"{det = }")


# fig, axs = plt.subplots(1, 2)
# axs[0].plot(t, disp)
# axs[0].set_title("Displacement")
# axs[1].plot(t, det)
# axs[1].set_title("Determinant")

# fig.savefig("foo.png", dpi=200)

from visualization.plot_displacement import plot_displacement, plot_determinant, plot_timestep, colors

foldernames = ["biharmonic", "supervised_learn/standard", "supervised_learn/incremental", "supervised_learn/lintraf_correct"] # ["biharmonic", "harmonic_trafo", "harmonic_notrafo"] #
names=  ["biharmonic", "learned", "learned linear incremental", "learned linear incremental corrected"] #["biharmonic", "harmonic incremental", "harmonic"]
names = ["Torch-corrected"]

times_list = [t]
displacement_list = [disp]
determinant_list = [det]

# for i in foldernames:
#     str = "../Output/files/" + i
#     times_list.append(np.loadtxt(str + "/times.txt"))
#     displacement_list.append(np.loadtxt(str + "/displacementy.txt"))
#     determinant_list.append(np.loadtxt(str + "/determinant.txt"))

# plt.figure()

plot_displacement(displacement_list, times_list,
                    'TorchOutput/visualizations/displacement_plot.pdf', [colors[2]], names)

plot_determinant(determinant_list, times_list,
                    'TorchOutput/visualizations/determinant_plot.pdf', [colors[2]], names)

plot_timestep(times_list,
                    'TorchOutput/visualizations/timestepsize_plot.pdf', [colors[2]], names)

