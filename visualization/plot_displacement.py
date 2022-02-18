import matplotlib.pyplot as pl
import numpy as np

#times = np.asarray(range(len(displacement)))*deltat

#pl.plot(times, displacement)
#pl.axis([0,15, -0.1, 0.1])
#pl.savefig('./displacement_plot.png')

colors = [np.asarray([0, 101, 189])*1./255,
          np.asarray([218, 215, 213])*1./255]#,
         # np.asarray([227, 114, 34])*1./255,
         # np.asarray([0, 0, 0])*1./255
         #]

def plot_displacement(list, times, str):
    colors = [np.asarray([0, 101, 189])*1./255,
          np.asarray([218, 215, 213])*1./255]
    if len(list)>2:
        colors = [(len(list)-i)/(len(list)-1)*colors[0] + i/(len(list)-1)*colors[1] for i in range(len(list))]
    for i in range(len(list)):
        pl.plot(times[i], list[i], color = colors[i])
        pl.axis([0, 15, -0.1, 0.1])
        pl.savefig(str)
    pl.close()

def plot_determinant(list, times, str):
    colors = [np.asarray([0, 101, 189]) * 1. / 255,
              np.asarray([218, 215, 213]) * 1. / 255]
    if len(list) > 2:
        colors = [(len(list)-i)/(len(list)-1)*colors[0] + i/(len(list)-1)*colors[1] for i in range(len(list))]
    for i in range(len(list)):
        pl.plot(times[i], list[i], color = colors[i])
        pl.axis([0, 15, -0.1, 1.1])
        pl.savefig(str)
    pl.close()

if __name__ == "__main__":
    times = np.loadtxt('../Output/files/learned/times.txt')

    displacement = np.loadtxt('../Output/files/learned/displacementy.txt')

    plot_displacement([displacement], [times],
                      '../Output/visualizations/displacement_plot.png')

    determinant = np.loadtxt('../Output/files/learned/determinant.txt')
    deltat = 0.0025
    plot_determinant([determinant], [times],
                     '../Output/visualizations/determinant_plot.png')
