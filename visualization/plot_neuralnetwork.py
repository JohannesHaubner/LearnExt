import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent))
sys.path.insert(1, '../learnExt')

from NeuralNet.neural_network_custom import ANN, generate_weights, NN
from NeuralNet.neural_network_orig import NN as NN_orig
from learnext import LearnExt

sys.path.insert(1, '../example')

threshold = 0.001


# load neural network
output_directory = str("../example/learned_networks/")
net = ANN(output_directory + "trained_network_758_step1.pkl")
net2 = ANN(output_directory + "trained_network_0903_1.pkl")
output_directory = str("../Output/learnExt/results/")
net3 = ANN(output_directory + "trained_network_hybrid.pkl")

mesh = UnitSquareMesh(5, 5)
Vs = FunctionSpace(mesh, "CG", 1)

x = np.linspace(0, 1.0, 50)
#y = [project(LearnExt.NN_der(threshold, i, net), Vs).vector().get_local()[0] for i in x]
y2 = [project(LearnExt.NN_der(threshold, i, net2), Vs).vector().get_local()[0] for i in x]
#y3 = [project(LearnExt.NN_der(threshold, i, net3), Vs).vector().get_local()[0] for i in x]

#net3.plot_antider = True
#y4 = [project(LearnExt.NN_der(threshold, i, net3), Vs).vector().get_local()[0] for i in x]
y5 = [(1e2*i+1)**3 for i in x]
plt.figure()
#plt.plot(x,y)
plt.plot(x,y2)
#plt.plot(x,y3)
#plt.plot(x,y4)
plt.plot(x,y5)
plt.show()
