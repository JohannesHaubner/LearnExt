import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent))
sys.path.insert(1, '../learnExt')

from NeuralNet.neural_network_custom import ANN, generate_weights
from learnext import LearnExt

sys.path.insert(1, '../example')

threshold = 0.0


# load neural network
output_directory = str("../example/learned_networks/")
net = ANN(output_directory + "trained_network_758_step1.pkl")
net2 = ANN(output_directory + "trained_network_758_step3.pkl")
output_directory = str("../Output/learnExt/results/")
net3 = ANN(output_directory + "trained_network.pkl")

mesh = IntervalMesh(2, 0, 1)
Vs = FunctionSpace(mesh, "CG", 1)

x = np.linspace(0, 20, 100)
#y = [project(LearnExt.NN_der(threshold, i, net), Vs).vector().get_local()[0] for i in x]
#y2 = [project(LearnExt.NN_der(threshold, i, net2), Vs).vector().get_local()[0] for i in x]
y3 = [project(LearnExt.NN_der(threshold, i, net3), Vs).vector().get_local()[0] for i in x]

plt.figure()
#plt.plot(x,y)
#plt.plot(x,y2)
plt.plot(x,y3)
plt.show()

breakpoint()