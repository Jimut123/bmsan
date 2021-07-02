
import GPy
import numpy as np
import scipy.io 
import os
import matplotlib.pyplot as plt
import cmath
from numpy import linalg as LA

from FV_initial_file import FV_2D
from GPyOpt.methods import BayesianOptimization


import variables_file
xc = variables_file.xc;
h0 = variables_file.h0; 
N = variables_file.N;
x_bathy = variables_file.x_bathy;
h_lower = variables_file.h_lower;
h_upper = variables_file.h_upper;
#-------------------------------------------------

import pickle    
f1 = open('Input_data.pickle', 'rb');
[inputs_all, runup_mat, time_all_list, runup_all_list] = pickle.load(f1);
f1.close();


domain =  [{'name': 'sigma', 'type': 'continuous', 'domain':(0.4,2), 'dimensionality':1}, 
           {'name': 'lengthscale', 'type': 'continuous', 'domain':(5,15), 'dimensionality':1}, 
           {'name': 'h_brk_1', 'type': 'continuous', 'domain':(0.001,7), 'dimensionality':1},
           {'name': 'h_brk_2', 'type': 'continuous', 'domain':(0.001,7), 'dimensionality':1},
           {'name': 'h_brk_3', 'type': 'continuous', 'domain':(0.001,7), 'dimensionality':1},
           {'name': 'h_brk_4', 'type': 'continuous', 'domain':(0.001,7), 'dimensionality':1}, 
           {'name': 'h_brk_5', 'type': 'continuous', 'domain':(0.001,7), 'dimensionality':1},
           {'name': 'h_brk_6', 'type': 'continuous', 'domain':(0.001,7), 'dimensionality':1},
           {'name': 'h_brk_7', 'type': 'continuous', 'domain':(0.001,7), 'dimensionality':1}];

   
xc_list = xc[:,0].tolist(); 
h0_list = h0[:,0].tolist();
h_lower_list = h_lower[:,0].tolist();
h_upper_list = h_upper[:,0].tolist();

constrains =  [ {'name': 'constr_1', 'constraint': 'x[:,2] - np.interp({},  {}, {})'.format(x_bathy[0,0], xc_list, h_lower_list) },
                {'name': 'constr_2', 'constraint': 'x[:,3] - np.interp({},  {}, {})'.format(x_bathy[1,0], xc_list, h_lower_list) },
                {'name': 'constr_3', 'constraint': 'x[:,4] - np.interp({},  {}, {})'.format(x_bathy[2,0], xc_list, h_lower_list) },
                {'name': 'constr_4', 'constraint': 'x[:,5] - np.interp({},  {}, {})'.format(x_bathy[3,0], xc_list, h_lower_list) },
                {'name': 'constr_5', 'constraint': 'x[:,6] - np.interp({},  {}, {})'.format(x_bathy[4,0], xc_list, h_lower_list) },
                {'name': 'constr_6', 'constraint': 'x[:,7] - np.interp({},  {}, {})'.format(x_bathy[5,0], xc_list, h_lower_list) },
                {'name': 'constr_7', 'constraint': 'x[:,8] - np.interp({},  {}, {})'.format(x_bathy[6,0], xc_list, h_lower_list) },
                {'name': 'constr_11', 'constraint': 'np.interp({},  {}, {}) - x[:,2]'.format(x_bathy[0,0], xc_list, h_upper_list) },
                {'name': 'constr_12', 'constraint': 'np.interp({},  {}, {}) - x[:,3]'.format(x_bathy[1,0], xc_list, h_upper_list) },
                {'name': 'constr_13', 'constraint': 'np.interp({},  {}, {}) - x[:,4]'.format(x_bathy[2,0], xc_list, h_upper_list) },
                {'name': 'constr_14', 'constraint': 'np.interp({},  {}, {}) - x[:,5]'.format(x_bathy[3,0], xc_list, h_upper_list) },
                {'name': 'constr_15', 'constraint': 'np.interp({},  {}, {}) - x[:,6]'.format(x_bathy[4,0], xc_list, h_upper_list) }, 
                {'name': 'constr_16', 'constraint': 'np.interp({},  {}, {}) - x[:,7]'.format(x_bathy[5,0], xc_list, h_upper_list) },
                {'name': 'constr_17', 'constraint': 'np.interp({},  {}, {}) - x[:,8]'.format(x_bathy[6,0], xc_list, h_upper_list) }];

 
runup_mat = -runup_mat;
kernel = GPy.kern.Matern52(input_dim=9, ARD=True, variance=1, lengthscale=[1,1,1,1,1,1,1,1,1]);
myBopt = BayesianOptimization(f=FV_2D, X= inputs_all[:,0,:], Y=runup_mat, domain=domain, constraints=constrains, 
                              kernel=kernel,
                              acquisition_type ='EI', initial_design_numdata = 0, 
                              exact_feval=True,
                              verbosity=True,
                              cost_withGradients=None,
                              model_type='GP', 
                              acquisition_optimizer_type='lbfgs') 
                             
myBopt.run_optimization(max_iter=100)
myBopt.plot_convergence()


Bopt_out = open('Bopt_results.pickle', 'wb');
pickle.dump([myBopt.X, myBopt.Y)], Bopt_out);     
Bopt_out.close();
    

min_runup = np.minimum.accumulate(myBopt.Y).ravel()
iteration = np.arange(1,min_runup.shape[0]+1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(iteration, min_runup, 'r-')
ax2.set_xlabel('Iteration no.', fontsize=13)
ax2.set_ylabel('Optimized variable', fontsize=13)
plt.show()