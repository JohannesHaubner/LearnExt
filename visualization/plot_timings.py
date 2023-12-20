import matplotlib.pyplot as plt
import numpy as np

import pickle 
from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent))

color_ = [np.asarray([0, 51, 89])*1./255,
          np.asarray([227, 114, 34]) * 1/255,
          np.asarray([218, 215, 213])*1./255
          ]

colors_blue = []
colors_orange = []
for i in range(3):
    colors_blue.append( (4 - i)/4*color_[0] + i/4*np.asarray([1,1,1]))
    colors_orange.append( (8 - 2*i)/8*color_[1] + i/8*np.asarray([1,1,1]))

colors = np.concatenate( (np.asarray(colors_blue), np.asarray(colors_orange), np.asarray([color_[2]])), axis=0)

#colors.append(colors_orange)
#colors.append(color_[2])

with open('Output/Extension/Data/timings.pickle', 'rb') as handle:
    b = pickle.load(handle)

#from IPython import embed; embed()

for k in b.keys():
    c = list(b[k].keys())
    d = {}
    j1 = []
    j2 = []
    j3 = []
    j4 = []
    j5 = []
    j6 = []
    j7 = []
    for j in b[k].keys():
        j1.append(b[k][j]['linear solves'])
        j6.append(b[k][j]['torch'])
        j7.append(b[k][j]['clement'])
        j2.append(b[k][j]['correct'] - j7[-1] - j6[-1])
        j3.append(b[k][j]['assemble_snes'])
        j4.append(b[k][j]['snes_total'] - j3[-1])
        j5.append(b[k][j]['total'] - j1[-1] - j2[-1] - j3[-1] - j4[-1])
        d['linear solves'] = np.array(j1)
        d['nonlinear solve: assembly'] = np.array(j3)
        d['nonlinear solve: rest'] = np.array(j4)
        d['NN correction'] = np.array(j6)
        d['Clement interpolation'] = np.array(j7)
        d['torch rest'] = np.array(j2)
        d['rest'] = np.array(j5)
    
    # plot
    fig, ax = plt.subplots()
    bottom = np.zeros(len(c))
    i = 0
    for type, times in d.items():
        p = ax.bar(c, times, 0.5, label=type, bottom=bottom, color=colors[i])
        i += 1
        bottom += times
    ax.legend()
    ax.set_ylabel('avg wall tot time per iteration')
    ax.tick_params(axis='x', labelrotation = 90)
    #ax.tick_params(axis='y', labelrotation = 90)
    plt.savefig('./timings_plot' + str(k) + '.png')