import matplotlib.pyplot as plt
import numpy as np

from operator import itemgetter 

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
for i in range(2):
    colors_blue.append( (4 - i)/4*color_[0] + i/4*np.asarray([1,1,1]))
for i in range(3):
    colors_orange.append( (4 - i)/4*color_[1] + i/4*np.asarray([1,1,1]))

colors =  [np.asarray([0, 51, 89])*1./255,
          np.asarray([0, 101, 189])*1./255,
          np.asarray([227, 114, 34])*1./255,
          np.asarray([218, 215, 213])*1./255]
#colors.append(colors_orange)
#colors.append(color_[2])

colors = np.concatenate( (np.asarray(colors_blue), np.asarray(colors_orange), np.asarray([color_[2]])), axis=0)

with open('Output/Extension/Data/timings.pickle', 'rb') as handle:
    b = pickle.load(handle)

#from IPython import embed; embed()
js = 0

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
        j6.append(b[k][j]['torch'])
        j7.append(b[k][j]['clement'])
        j2.append(b[k][j]['correct'] - j7[-1] - j6[-1])
        j3.append(b[k][j]['assemble'])
        j1.append(b[k][j]['solve'] - j3[-1])
        j5.append(b[k][j]['total'] - j1[-1] - j2[-1] - j3[-1] - j6[-1] - j7[-1])
        d['linear solves'] = np.array(j1)
        d['assemble'] = np.array(j3)
        d['torch'] = np.array(j6)
        d['Clement interpolation'] = np.array(j7)
        d['NN correction rest'] = np.array(j2)
        d['rest'] = np.array(j5)
    
    # plot
    fig, ax = plt.subplots()
    i = 0
    c = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    indices = [0, 1, 5]
    bottom = np.zeros(len(indices))
    titles = ['refinement 0', 'refinement 1', 'refinement 2']
    for type, times in d.items():
        p = ax.bar(itemgetter(*indices)(c), itemgetter(*indices)(times), 0.5, label=type, bottom=bottom, color=colors[i])
        i += 1
        bottom += itemgetter(*indices)(times)
    ax.legend()
    ax.set_ylabel('avg wall tot time per iteration')
    ax.tick_params(axis='x')
    plt.title(titles[js])
    js += 1
    #ax.tick_params(axis='y', labelrotation = 90)
    #plt.gca().invert_yaxis()
    plt.savefig('./timings_plot_part_' + str(k) + '.png')