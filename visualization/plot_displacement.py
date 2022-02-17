import matplotlib.pyplot as pl
import numpy as np

#times = np.asarray(range(len(displacement)))*deltat

#pl.plot(times, displacement)
#pl.axis([0,15, -0.1, 0.1])
#pl.savefig('./displacement_plot.png')

colors = [np.asarray([0, 101, 189])*1./255,
          np.asarray([218, 215, 213])*1./255,
          np.asarray([227, 114, 34])*1./255,
          np.asarray([0, 0, 0])*1./255
         ]

def plot_displacement(list, list_dt, str):
    for i in range(len(list)):
        times = np.asarray(range(len(list[i])))*list_dt[i]
        pl.plot(times, list[i], color = colors[i])
        pl.axis([0, 15, -0.1, 0.1])
        pl.savefig(str)
    pl.close()

def plot_determinant(list, list_dt, str):
    for i in range(len(list)):
        times = np.asarray(range(len(list[i])))*list_dt[i]
        pl.plot(times, list[i], color = colors[i])
        pl.axis([0, 15, -0.1, 1.1])
        pl.savefig(str)
    pl.close()

if __name__ == "__main__":
    displacement_b = np.loadtxt('../Output/files/biharmonic/displacementy.txt')
    displacement_h = np.loadtxt('../Output/files/harmonic_learned_0025/displacementy.txt')
    displacement_n = np.loadtxt('../Output/files/harmonic_new/displacementy.txt')
    displacement_nn = np.loadtxt('../Output/files/learned_new/displacementy.txt')
    deltat = 0.0025
    plot_displacement([displacement_b, displacement_n, displacement_h, displacement_nn], [deltat, 0.01, deltat, 0.01],
                      '../Output/visualizations/displacement_plot.png')

    determinant_b = np.loadtxt('../Output/files/biharmonic/determinant.txt')
    determinant_h = np.loadtxt('../Output/files/harmonic_learned_0025/determinant.txt')
    determinant_n = np.loadtxt('../Output/files/harmonic_new/determinant.txt')
    determinant_nn = np.loadtxt('../Output/files/learned_new/determinant.txt')
    deltat = 0.0025
    plot_determinant([determinant_b, determinant_n, determinant_h, determinant_nn], [deltat, 0.01, deltat, 0.01],
                     '../Output/visualizations/determinant_plot.png')
