import os
import itertools
from pyomo.environ import *
irange = lambda start, end: range(start, end+1)

current_dir = os.path.split(os.path.abspath(__file__))[0]
solver_name = "bonmin"
solver_path = current_dir + r"\..\solvers\CoinAll-1.6.0-win64-intel11.1\bin\bonmin.exe"


index_max = {
    "i" : 3,
    "k" : 4,
    "j" : 7,
    "t" : 7,
    "f" : 2,
    "w" : 3,
    "r" : 4,
}
import random
def random_float(min=0.5, max=9.5):
    return random.uniform(min, max)

def init_ndim_array(dimensions, default=None):
    if type(dimensions) is not list or len(dimensions) <= 0: return default
    else: return init_ndim_array(dimensions, default=[default]*dimensions.pop())

SP_jt = init_ndim_array([index_max["j"], index_max["t"]])
ORDER_jrt = init_ndim_array([index_max["j"], index_max["r"], index_max["t"]])
LOST_jrt = init_ndim_array([index_max["j"], index_max["r"], index_max["t"]])
for j in range(index_max["j"]):
    for t in range(index_max["t"]):
        SP_jt[j][t] = random_float()
        for r in range(index_max["r"]):
            ORDER_jrt[j][r][t] = random_float()
            LOST_jrt[j][r][t] = random_float()


## ===
a, b, c = 370, 420, 1
model             = ConcreteModel()
model.x           = Var([1,2], domain=Binary)
model.y           = Var([1,2], domain=Binary)
model.Objective   = Objective(expr = a * model.x[1] + b * model.x[2] + (a-b)*model.y[1] + (a+b)*model.y[2], sense=maximize)
model.Constraint1 = Constraint(expr = model.x[1] + model.x[2] + model.y[1] + model.y[2] <= c)
## ===

print("Using the solver {NAME} in filepath {PATH}".format(NAME=solver_name, PATH=solver_path))
opt = SolverFactory(solver_name, executable=solver_path)
results = opt.solve(model)

print("Print values for all variables")
for v in model.component_data_objects(Var):
  print(str(v), v.value)