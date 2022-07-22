'''
    python demonstration file for NMFLibrary.

    This file illustrates how to use this library. 
    This demonstrates Frobenius-norm based 
	- multiplicative updates (MU) algorithm,
     	- hierarchical alternative least squares (HALS) algorithm, and
     	- accelerated HALS algorithm.
	
    This file is part of NMFLibrary.

    Created by H.Kasai on July 21, 2022


    To execute this script, the following setup is required. 

    See more details at 
    https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

    1. Find the path to the MATLAB folder. Start MATLAB and type matlabroot in the command window. Copy the path returned by matlabroot.

    2. Install the Engine API

       To install the engine API, choose one of the following. You must call this python install command in the specified folder.
	
       - For Windows
		cd "matlabroot\extern\engines\python"
		python setup.py install

       - macOS or Linux
		cd "matlabroot/extern/engines/python"
		python setup.py instal

	- Examples (R2022 case):

	    For Windows 
 		cd "c:\Program Files\MATLAB\R2022a\extern\engines\python" 
 		python setup.py install

            For Linux
		cd "/usr/local/MATLAB/R2022a/bin/matlab/extern/engines/python"
		python setup.py install

	    For macOS
		cd "/Applications/MATLAB_R2022a.app/extern/engines/python"
 		python setup.py install 
'''

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


# generate matrix
m = 500
n = 100
V = np.random.random_sample((m,n))

# define rank to be factorized
rank = 5

# set options for solvers
options = dict()
options['verbose'] = '1'
options['max_epoch'] = '100'

# import MATLAB module
import matlab.engine
eng = matlab.engine.start_matlab()

# add MATLAB path
eng.run_me_first(0, nargout=0)

# convert numpy array to matlab.double
V_m = matlab.double(V.tolist())

# convert dictioary to matlab.struct
options_m = eng.struct(options);

# perform solvers in MATLAB 
# Fro-MU
[w_mu, infos_mu] = eng.fro_mu_nmf(V_m, rank, options_m, nargout=2)
# HALS
[w_hals, infos_hals] = eng.als_nmf(V_m, rank, options_m, nargout=2)
# ACC-HALS
options['alg'] = 'acc_hals'
options_m = eng.struct(options);
[w_acc_hals, infos_acc_hals] = eng.als_nmf(V_m, rank, options_m, nargout=2)

# convert matlab.struct to list
iter_mu = list(infos_mu['iter'][0])
time_mu = list(infos_mu['time'][0])
cost_mu = list(infos_mu['cost'][0])
iter_hals = list(infos_hals['iter'][0])
time_hals = list(infos_hals['time'][0])
cost_hals = list(infos_hals['cost'][0])
iter_acc_hals = list(infos_acc_hals['iter'][0])
time_acc_hals = list(infos_acc_hals['time'][0])
cost_acc_hals = list(infos_acc_hals['cost'][0])

# plotting
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.plot(iter_mu, cost_mu, label ="Fro-MU")
plt.plot(iter_hals, cost_hals, label ="HALS")
plt.plot(iter_acc_hals, cost_acc_hals, label ="ACC-HALS")
plt.legend() 
plt.title('Epoch vs. Cost')

plt.figure()
plt.xlabel('Time [sec]')
plt.ylabel('Cost')
plt.plot(time_mu, cost_mu, label ="Fro-MU")
plt.plot(time_hals, cost_hals, label ="HALS")
plt.plot(time_acc_hals, cost_acc_hals, label ="ACC-HALS")
plt.legend() 
plt.title('Time vs. Cost')

plt.show()

eng.quit()