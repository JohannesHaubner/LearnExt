from fenics import *
from dolfin_adjoint import *
import numpy as np

def trafo_weights(init_weights, posfunc):
    weights_trafo = []
    #i = -1
    for weight in init_weights:
        #i += 1
        w = weight["weight"]
        weights_trafo.append(Constant(posfunc(w.values()).reshape(w.ufl_shape)))
        if "bias" in weight:
            #i += 1
            bias = weight["bias"]
            weights_trafo.append(Constant(bias.values().reshape(bias.ufl_shape)))
    return weights_trafo


def weights_to_list(init_weights):
    list = []
    for weight in init_weights:
        w = weight["weight"]
        list.append(w)
        if "bias" in weight:
            bias = weight["bias"]
            list.append(bias)
    return list

def list_to_array(list):
    array = []
    for y in list:
        array.append(y.values())
    return np.concatenate(array, axis=0)

def list_to_weights(list_weights, init_weights):
    weights = []
    i = -1
    for weight in init_weights:
        i+=1
        app2 = {}
        w = weight["weight"]
        app2["weight"] = list_weights[i]
        if "bias" in weight:
            i +=1
            bias = weight["bias"]
            app2["bias"] = list_weights[i]
        weights.append(app2)
    return weights


def weights_list_add(y1, y2, eps):
    weights_add = []
    i = -1
    while i < len(y1) -1:
        i +=1
        weights_add.append(Constant((y1[i].values() + eps*y2[i].values()).reshape(y1[i].ufl_shape)))
    return weights_add

def weights_add(y1, y2, eps):
    y1list = weights_to_list(y1)
    y2list = weights_to_list(y2)
    ylist = weights_list_add(y1list, y2list, eps)
    ylist_weights = list_to_weights(ylist, y1)
    return ylist_weights

def trafo_weights_chainrule(df, init_weights, posfunc_der):
    # implement later via overloading
    df_cr = []
    i = -1
    for weight in init_weights:
        i = i+1
        w = weight["weight"]
        a = posfunc_der(w.values())
        b = df[i].values()
        df_cr.append(Constant((a*b).reshape(w.ufl_shape)))
        if "bias" in weight:
            i = i+1
            df_cr.append(df[i])
    return df_cr

def flatten(list):
    ctrls = list
    r = []
    for ctrl in ctrls:
        r.append(ctrl)
    return r