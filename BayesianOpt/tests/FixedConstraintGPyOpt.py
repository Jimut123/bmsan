import GPyOpt
import GPy
import numpy as np

func = GPyOpt.objective_examples.experiments2d.sixhumpcamel()


space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)},
        {'name': 'var_2', 'type': 'continuous', 'domain': (-1.5,1.5)}]




