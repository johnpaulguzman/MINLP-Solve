########################################################################
#
#     This is an example call of MIDACO 5.0
#     -------------------------------------
#
#     MIDACO solves Multi-Objective Mixed-Integer Non-Linear Problems:
#
#
#      Minimize     F_1(X),... F_O(X)  where X(1,...N-NI)   is CONTINUOUS
#                                      and   X(N-NI+1,...N) is DISCRETE
#
#      subject to   G_j(X)  =  0   (j=1,...ME)      equality constraints
#                   G_j(X) >=  0   (j=ME+1,...M)  inequality constraints
#
#      and bounds   XL <= X <= XU
#
#
#     The problem statement of this example is given below. You can use 
#     this example as template to run your own problem. To do so: Replace 
#     the objective functions 'F' (and in case the constraints 'G') given 
#     here with your own problem and follow the below instruction steps.
#
########################################################################
######################   OPTIMIZATION PROBLEM   ########################
########################################################################
def problem_function(x):
    print("tweaked x:", x)
    f = [0.0]*1 # Initialize array for objectives F(X)
    g = [0.0]*3 # Initialize array for constraints G(X)

    # Objective functions F(X)
    f[0] = (x[0] - 1)*(x[0] - 1) + \
           (x[1] - 2)*(x[1] - 2) + \
           (x[2] - 3)*(x[2] - 3) + \
           (x[3] - 4)*(x[3] - 4)**2 + \
           1.23456789  
    
    #  Equality constraints G(X) = 0 MUST COME FIRST in g[0:me-1]
    g[0] = x[0] - 1.0
    # Inequality constraints G(X) >= 0 MUST COME SECOND in g[me:m-1] 
    g[1] = x[1] - 1.333333333       
    g[2] = x[2] - 2.666666666 
    
    return f,g

########################################################################
#########################   MAIN PROGRAM   #############################
########################################################################
import random
import numpy

def random_float(min=0.5, max=4.5):
    return round(random.uniform(min, max))

def init_ndim_array(dimensions, default=None):
    if type(dimensions) is not list or len(dimensions) <= 0: return default
    else: return init_ndim_array(dimensions, default=[default]*dimensions.pop())

max_j = 3
max_r = 2
max_t = 5
SP_jt = init_ndim_array([max_j, max_t])
ORDER_jrt = init_ndim_array([max_j, max_r, max_t])
LOST_jrt = init_ndim_array([max_j, max_r, max_t])

for j in range(max_j):
    for t in range(max_t):
        SP_jt[j][t] = random_float()
        for r in range(max_r):
            ORDER_jrt[j][r][t] = random_float()
            LOST_jrt[j][r][t] = random_float()

flat = numpy.array(SP_jt).reshape(max_j * max_t)
unflat = flat.reshape(max_j, max_t)

Mx = list(flat)
Mxl = [0] * len(Mx)
Mxu = [i*2.0 for i in Mx]

key = b'MIDACO_LIMITED_VERSION___[CREATIVE_COMMONS_BY-NC-ND_LICENSE]'

problem = {} # Initialize dictionary containing problem specifications
option  = {} # Initialize dictionary containing MIDACO options

problem['@'] = problem_function # Handle for problem function name

########################################################################
### Step 1: Problem definition     #####################################
########################################################################

# STEP 1.A: Problem dimensions
##############################
problem['o']  = 1  # Number of objectives 
problem['n']  = 4 #+ len(flat)  # Number of variables (in total) 
problem['ni'] = 2  # Number of integer variables (0 <= ni <= n) 
problem['m']  = 3  # Number of constraints (in total) 
problem['me'] = 1  # Number of equality constraints (0 <= me <= m) 

# STEP 1.B: Lower and upper bounds 'xl' & 'xu'  
##############################################  
problem['xl'] = [ 1, 1, 1, 1 ] #+ [1] * len(flat)
problem['xu'] = [ 4, 4, 4, 4 ] #+ [4] * len(flat)

# STEP 1.C: Starting point 'x'  
##############################  
problem['x'] = problem['xl'] # Here for example: starting point = lower bounds
    
########################################################################
### Step 2: Choose stopping criteria and printing options    ###########
########################################################################
   
# STEP 2.A: Stopping criteria 
#############################
option['maxeval'] = 10000     # Maximum number of function evaluation (e.g. 1000000) 
option['maxtime'] = 60*60*24  # Maximum time limit in Seconds (e.g. 1 Day = 60*60*24) 

# STEP 2.B: Printing options  
############################ 
option['printeval'] = 1000  # Print-Frequency for current best solution (e.g. 1000) 
option['save2file'] = 1     # Save SCREEN and SOLUTION to TXT-files [0=NO/1=YES]

########################################################################
### Step 3: Choose MIDACO parameters (FOR ADVANCED USERS)    ###########
########################################################################

option['param1']  = 0.0  # ACCURACY  
option['param2']  = 0.0  # SEED  
option['param3']  = 0.0  # FSTOP  
option['param4']  = 0.0  # ALGOSTOP  
option['param5']  = 0.0  # EVALSTOP  
option['param6']  = 0.0  # FOCUS  
option['param7']  = 0.0  # ANTS  
option['param8']  = 0.0  # KERNEL  
option['param9']  = 0.0  # ORACLE  
option['param10'] = 0.0  # PARETOMAX
option['param11'] = 0.0  # EPSILON  
option['param12'] = 0.0  # CHARACTER

########################################################################
### Step 4: Choose Parallelization Factor   ############################
########################################################################

option['parallel'] = 0 # Serial: 0 or 1, Parallel: 2,3,4,5,6,7,8...

########################################################################
############################ Run MIDACO ################################
########################################################################

import midaco

if __name__ == '__main__': 

  solution = midaco.run( problem, option, key )

# print(solution['f'])
# print(solution['g'])
# print(solution['x'])

########################################################################
############################ END OF FILE ###############################
########################################################################
         
