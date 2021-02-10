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


import drrmsan_multilosses


def f(x):
    """
    x is a 4D vector.
    Function which will send alpha_1, alpha_2, alpha_3 and alpha_4
    to the actual model and will get the dice coefficient in return.
    """
    alpha_1 = x[:, 0]
    alpha_2 = x[:, 1]
    alpha_3 = x[:, 2]
    alpha_4 = x[:, 3]
    print(alpha_1, " ", alpha_2," ",alpha_3," ",alpha_4)
    # Here we will send the alphas to the actual model and in return
    # we will recieve the dice coefficient to optimise, since this is
    # a maximization problem, we return the -ve of objective function
    # to be maximized
    dice_coef = drrmsan_multilosses.get_dice_from_alphas(float(alpha_1[0]), float(alpha_2[0]), float(alpha_3[0]), float(alpha_4[0]))
    return -dice_coef

domain = [{'name': 'alpha_1', 'type': 'continuous', 'domain': (0,1)},
          {'name': 'alpha_2', 'type': 'continuous', 'domain': (0,1)},
          {'name': 'alpha_3', 'type': 'continuous', 'domain': (0,1)},
          {'name': 'alpha_4', 'type': 'continuous', 'domain': (0,1)}]

constraints = [{'name': 'constr_1', 'constraint': '0.9999 - x[:,0] - x[:,1] - x[:,2] - x[:,3]'},
               {'name': 'constr_1', 'constraint': '-1.00001 + x[:,0] + x[:,1] + x[:,2] + x[:,3]'}]


alpha_1_list = []
alpha_2_list = []
alpha_3_list = []
alpha_4_list = []
for i in range(20):
    alphas = np.random.dirichlet(np.ones(4),size=1)
    #print("case = ", i+1, " ", alphas[0][0], alphas[0][1], alphas[0][2], alphas[0][3])
    alpha_1_list.append(alphas[0][0])
    alpha_2_list.append(alphas[0][1])
    alpha_3_list.append(alphas[0][2])
    alpha_4_list.append(alphas[0][3])
alpha_1_np = np.array(alpha_1_list)
alpha_2_np = np.array(alpha_2_list)
alpha_3_np = np.array(alpha_3_list)
alpha_4_np = np.array(alpha_4_list)
alpha_1, alpha_2, alpha_3, alpha_4 = np.meshgrid(alpha_1_np, alpha_2_np, alpha_3_np, alpha_4_np)


maxiter = 20

myBopt_4d = GPyOpt.methods.BayesianOptimization(f, domain=domain)
myBopt_4d.run_optimization(max_iter = maxiter, verbosity=True)
print("="*20)
print("Value of (x,y) that minimises the objective:"+str(myBopt_4d.x_opt))
print("Minimum value of the objective: "+str(myBopt_4d.fx_opt))
print("="*20)
#myBopt_4d.plot_acquisition()


