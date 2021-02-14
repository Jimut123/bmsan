# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np

# --- Define your problem
def f(x): 
    x1 = x[:,0]
    x2 = x[:,1]
    return (6*x1-2)**2*np.sin(12*x2-4)
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)},
        {'name': 'var_2', 'type': 'continuous', 'domain': (0,1)}]

# --- Solve your problem
myBopt = BayesianOptimization(f=f, domain=domain)
myBopt.run_optimization(max_iter=15)
myBopt.plot_acquisition()


