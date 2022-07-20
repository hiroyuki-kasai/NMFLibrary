'''
    To execute this script, the following setup is required. 

    See more details at 
    https://jp.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

	
    For Windos
		cd "matlabroot\extern\engines\python"
		python setup.py install

	macOS or Linux
		cd "matlabroot/extern/engines/python"
		python setup.py instal

	Examples (R2022 case):

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
rank = 5

# set options for solvers
options = dict()
options['verbose'] = '1'
options['max_epoch'] = '100'

# add path of matlabAPI
#sys.path.append('matlabroot/extern/engines/python/build/lib/')

# import matlab module
import matlab.engine
eng = matlab.engine.start_matlab()

# convert numpy array to matlab.double
V_matlab = matlab.double(V.tolist())

# call nmf module in matlab
ret = eng.demo_python(V_matlab, rank, options)

# store results
mu_infos = ret['mu']
mu_iter = list(mu_infos['iter'][0])
mu_time = list(mu_infos['time'][0])
mu_cost = list(mu_infos['cost'][0])

hals_infos = ret['hals']
hals_iter = list(hals_infos['iter'][0])
hals_time = list(hals_infos['time'][0])
hals_cost = list(hals_infos['cost'][0])

acchals_infos = ret['acchals']
acchals_iter = list(acchals_infos['iter'][0])
acchals_time = list(acchals_infos['time'][0])
acchals_cost = list(acchals_infos['cost'][0])

# plotting
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.plot(mu_iter, mu_cost, label ="Fro-MU")
plt.plot(hals_iter, hals_cost, label ="HALS")
plt.plot(acchals_iter, acchals_cost, label ="Acc-HALS")
plt.legend() 
plt.title('Epoch vs. Cost')

plt.figure()
plt.xlabel('Time [sec]')
plt.ylabel('Cost')
plt.plot(mu_time, mu_cost, label ="Fro-MU")
plt.plot(hals_time, hals_cost, label ="HALS")
plt.plot(acchals_time, acchals_cost, label ="Acc-HALS")
plt.legend() 
plt.title('Time vs. Cost')

plt.show()

eng.quit()
