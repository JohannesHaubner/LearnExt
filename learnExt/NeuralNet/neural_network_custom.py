from fenics import *
from fenics_adjoint import *
import ufl
import numpy as np
from IPython import embed
from numpy.random import randn, random, seed
#seed(10)
import dill as pickle


class ANN(object):
    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            # Load weights.
            return cls.load(args[0])
        else:
            return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], str):
            # Pickle has loaded the weights.
            pass
        else:
            layers = args[0]
            bias = kwargs.get("bias")
            sig = lambda x: sigmoid(x)
            som = lambda x: softmax(x)
            sigma = kwargs.get("sigma", som)
            sigma_derivative = kwargs.get("sigma_derivative", sig)
            init_method = kwargs.get("init_method", "normal")
            self.weights = generate_weights(layers, bias, init_method=init_method)
            self.layers = layers
            self.bias = bias
            self.sigma = sigma
            self.sigma_derivative = sigma_derivative
            self.ctrls = None
            self.backup_weights_flat = None

    def save(self, path):
        with open(f"{path}", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(f"{path}", "rb") as f:
            return pickle.load(f)

    def __setstate__(self, state):
        self.__dict__ = state
        weights = []
        for w in self.weights:
            weight = {}
            weight["weight"] = Constant(np.reshape(w["weight"], w["weight_shape"]))
            if "bias" in w:
                weight["bias"] = Constant(np.reshape(w["bias"], w["bias_shape"]))
            weights.append(weight)
        self.weights = weights

    def __getstate__(self):
        state = self.__dict__.copy()
        weights = []
        for weight in self.weights:
            app = {}
            w = weight["weight"]
            app["weight"] = w.values()
            app["weight_shape"] = w.ufl_shape
            if "bias" in weight:
                bias = weight["bias"]
                app["bias"] = bias.values()
                app["bias_shape"] = bias.ufl_shape
            weights.append(app)
        state["weights"] = weights
        state["ctrls"] = None
        state["backup_weights_flat"] = None
        return state

    def __call__(self, *args):
        return NN(args, self.weights, self.sigma, self.sigma_derivative)

    def weights_flat(self):
        ctrls = self.weights_ctrls()
        r = []
        for ctrl in ctrls:
            r.append(ctrl.tape_value())
        return r

    def weights_ctrls(self):
        if self.ctrls is None:
            r = []
            for weight in self.weights:
                r.append(Control(weight["weight"]))
                if "bias" in weight:
                    r.append(Control(weight["bias"]))
            self.ctrls = r
        return self.ctrls

    def opt_callback(self, *args, **kwargs):
        r = []
        for ctrl in self.weights_ctrls():
            r.append(ctrl.tape_value()._ad_create_checkpoint())
        self.backup_weights_flat = r

    def set_weights(self, weights):
        i = 0
        for weight in self.weights:
            w = weight["weight"]
            w.assign(weights[i])
            w.block_variable.save_output()
            i += 1
            if "bias" in weight:
                weight["bias"].assign(weights[i])
                weight["bias"].block_variable.save_output()
                i += 1


def generate_weights(layers, bias, init_method="normal"):
    init_method = init_method.lower()
    assert init_method in ["normal", "uniform", "fixed"]

    weights = []
    for i in range(len(layers)-1):
        weight = {}

        dim = np.prod(layers[i] * layers[i+1])
        if init_method == "uniform":
            value = random(dim)
        elif init_method == "normal":
            value = np.sqrt(2 / layers[i]) * randn(dim)
        elif init_method == "fixed":
            value = 0*randn(dim)
        weight["weight"] = Constant(value.reshape(layers[i+1], layers[i]))

        if bias[i]:
            b = Constant(np.zeros(layers[i+1]))
            weight["bias"] = b
        weights.append(weight)
    return weights


def NN(inputs, weights, sigma, sigma_derivative):
    #weights = weight_transformation(weights) #seems to be the correct spot for weight_transform (get it running or include it somewhere else)
    r = as_vector(inputs)
    output = as_vector(inputs)
    depth = len(weights)
    for i, weight in enumerate(weights):
        #print('iteration', i, depth)
        term = weight["weight"] * r
        if "bias" in weight:
            term += weight["bias"]
        if i  >= depth:
            r = output
        else:
            r = apply_activation(term, func=sigma)
            r2 = apply_activation(term, func=sigma_derivative) # r2 has the correct form
            weight_tensor = weight_as_tensor(weight["weight"]) #weight_tensor has the correct form
            (k, j, i, l) = ufl.indices(4)
            delta = Identity(weight_tensor.ufl_shape[0])
            Akj = ufl.as_tensor(r2[k]*delta[k,l]*delta[i,k]*weight_tensor[i,j],(l,j))
            if i == 0:
                output = Akj
            else:
                output = Akj*output
            if i == depth -1:
                r = output

    if r.ufl_shape[0] == 1:
        return r[0]
    return r

def weight_as_tensor(weight):
    app = {}
    w = weight
    app["weight"] = w.values()
    app["weight_shape"] = w.ufl_shape
    #expressions = [ufl.as_ufl(Constant(e)) for e in w.values()]
    expressions = [Constant(e) for e in w.values()]
    exp_rs = np.reshape(expressions, w.ufl_shape)
    return ufl.as_tensor(exp_rs)



def identity(x):
    return x


class ELU(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, *args, **kwargs):
        return conditional(ufl.gt(x, 0), x, self.alpha * (ufl.exp(x) - 1))


def sigmoid(x):
    return 1/(1 + ufl.exp(-x))

def softmax(x):
    return ufl.ln(1 + ufl.exp(x))


def apply_activation(vec, func=ufl.tanh):
    """Applies the activation function `func` element-wise to the UFL expression `vec`.
    """
    v = [func(vec[i]) for i in range(vec.ufl_shape[0])]
    return ufl.as_vector(v)
