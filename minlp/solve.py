import os
from pyomo.environ import *
from pyomo.opt import SolverFactory

current_dir = os.path.split(os.path.abspath(__file__))[0]
solver_name = "bonmin"
solver_path = current_dir + r"\..\solvers\CoinAll-1.6.0-win64-intel11.1\bin\bonmin.exe"

a, b, c = 370, 420, 4
model             = ConcreteModel()
model.x           = Var([1,2], domain=Binary)
model.y           = Var([1,2], domain=Binary)
model.Objective   = Objective(expr = a * model.x[1] + b * model.x[2] + (a-b)*model.y[1] + (a+b)*model.y[2], sense=maximize)
model.Constraint1 = Constraint(expr = model.x[1] + model.x[2] + model.y[1] + model.y[2] <= c)

print("Using the solver {NAME} in filepath {PATH}".format(NAME=solver_name, PATH=solver_path))
opt = SolverFactory(solver_name, executable=solver_path)
results = opt.solve(model)

print("Print values for all variables")
for v in model.component_data_objects(Var):
  print(str(v), v.value)