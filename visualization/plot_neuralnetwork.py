import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent))

from learnExt.NeuralNet.neural_network_custom import ANN, generate_weights, NN
#from NeuralNet.neural_network_orig import NN as NN_orig
from learnExt.learnext_hybridPDENN import LearnExt
from learnExt.learnext_hybridPDENN import Custom_Reduced_Functional as crf

sys.path.insert(1, '../example')

threshold = 0.001


# load neural network
output_directory = str("../example/learned_networks/")
net2 = ANN(output_directory + "trained_network.pkl")
net = ANN(output_directory + "trained_network_supervised.pkl")
net4 = ANN(output_directory + "trained_network_1103.pkl")
output_directory = str("../example/learned_networks/artificial/")
net3 = ANN(output_directory + "trained_network.pkl")

mesh = UnitSquareMesh(5, 5)
Vs = FunctionSpace(mesh, "CG", 1)

x = np.linspace(0, 1.0, 50)
y = [project(crf.NN_der(threshold, i, net), Vs).vector().get_local()[0] for i in x]
#y2 = [project(crf.NN_der(threshold, i, net2), Vs).vector().get_local()[0] for i in x]
#y4 = [project(crf.NN_der(threshold, i, net4), Vs).vector().get_local()[0] for i in x]

#net3.plot_antider = True
#y3 = [project(crf.NN_der(threshold, i, net3), Vs).vector().get_local()[0] for i in x]
#y5 = [(1e2*i+1)**3 for i in x]
plt.figure()
plt.plot(x,y)
#plt.plot(x,y2)
#plt.plot(x,y3)
#plt.plot(x,y4)
#plt.plot(x,y5)
plt.xlabel(r"$s$")
plt.ylabel(r"$\alpha(\theta_{opt}, s)$")
#plt.show()
plt.savefig('../Output/visualizations/neuralnet_plot.pdf')
plt.close()