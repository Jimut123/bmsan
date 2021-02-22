#Import Modules

#GPyOpt - Cases are important, for some reason
import GPyOpt
from GPyOpt.methods import BayesianOptimization

#numpy
import numpy as np
from numpy.random import multivariate_normal #For later example

import pandas as pd

#Plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy.random import multivariate_normal

# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
#%pylab inline
import GPyOpt
import GPy
import numpy as np
import pickle
import numpy


import drrmsan_multilosses

def f(x):
    # x is a 4D vector.
    # Function which will send alpha_1, alpha_2, alpha_3 and alpha_4
    # to the actual model and will get the dice coefficient in return.
    
    alpha_1 = x[:, 0][0]
    alpha_2 = x[:, 1][0]
    alpha_3 = x[:, 2][0]
    alpha_4 = x[:, 3][0]
    print(alpha_1, " ", alpha_2," ",alpha_3," ",alpha_4)
    dice = drrmsan_multilosses.get_dice_from_alphas(float(alpha_1), float(alpha_2), float(alpha_3), float(alpha_4))
    # Here we will send the alphas to the actual model and in return
    # we will recieve the dice coefficient to optimise, since this is
    # a maximization problem, we return the -ve of objective function
    # to be maximized
    # dice_coef = drrmsan_multilosses.get_dice_from_alphas(float(alpha_1[0]), float(alpha_2[0]), float(alpha_3[0]), float(alpha_4[0]))
    # dice_coef =  float(alpha_1)+ float(alpha_2)+ float(alpha_3)+ float(alpha_4)
    print(dice)
    return dice

    

domain = [{'name': 'alpha_1', 'type': 'continuous', 'domain': (0,1), 'dimensionality':1},
          {'name': 'alpha_2', 'type': 'continuous', 'domain': (0,1), 'dimensionality':1},
          {'name': 'alpha_3', 'type': 'continuous', 'domain': (0,1), 'dimensionality':1},
          {'name': 'alpha_4', 'type': 'continuous', 'domain': (0,1), 'dimensionality':1}]

constraints = [{'name': 'constr_1', 'constraint': '0.9999 - x[:,0] - x[:,1] - x[:,2] - x[:,3]'},
               {'name': 'constr_1', 'constraint': '-1.00001 + x[:,0] + x[:,1] + x[:,2] + x[:,3]'}]


def load_entire_file_into_memory_and_then_convert(filename):
    with open(filename, 'r') as input_file:
        full_file_contents = input_file.read()
        lines_of_file = full_file_contents.split('\n')
        return numpy.array(lines_of_file)

dump = load_entire_file_into_memory_and_then_convert('store_alphas_dice.txt')
print(dump)

X = []
Y = []
for item in dump[:-1]:
    all_items = item.split(' ')
    X.append([float(i) for i in all_items[:4]]) 
    Y.append(float(all_items[-2]))

X = numpy.array(X)
print(X)

Y = -numpy.array(Y)
print(Y)

Y = np.expand_dims(Y, axis=1)
Y

maxiter = 1

kernel = GPy.kern.Matern52(input_dim=4, ARD=True, variance=1, lengthscale=[1,1,1,1]);

myBopt_4d = GPyOpt.methods.BayesianOptimization(f=f, X=X, Y=Y, domain=domain, constraints = constraints,
                                                kernel=kernel, acquisition_type ='EI', model_type='GP', 
                                                verbosity=True, acquisition_optimizer_type='lbfgs', 
                                                cost_withGradients=None, exact_feval=True)

myBopt_4d.run_optimization(max_iter = maxiter, verbosity=True)
print("="*20)
print("Value of (x,y) that minimises the objective:"+str(myBopt_4d.x_opt))    
print("Minimum value of the objective: "+str(myBopt_4d.fx_opt))     
print("="*20)
#myBopt_4d.plot_acquisition()

f = open("./bayesian_opt.txt", "a+")
dump_str = "Value of (x,y) that minimises the objective:"+str(myBopt_4d.x_opt)+"\n"
f.write(dump_str)
dump_str = "Minimum value of the objective: "+str(myBopt_4d.fx_opt)+"\n"
f.write(dump_str)
f.close()
