from xlsx_reader import XLSXReader
from pyomo.environ import *
from pyomo.dae import *
import os
import pprint; indented_print = pprint.PrettyPrinter(indent=4).pprint
#import code; code.interact(local=locals())

def PyomoMin(a, b):
    return base.expr.Expr_if(IF=a > b, THEN=a, ELSE=b)

def SafeIdx(item, *index_set, default=0):
    #if index_set not in item.index_set(): print("<Did guard {} with index {}".format(item.name, index_set))
    return item[index_set] if index_set in item.index_set() else default

def SafeIdx2(item, index_set):
    if index_set in item.index_set(): print("<Did guard2 {} with index {}".format(item.name, index_set))
    return base.expr.Expr_if(IF=index_set in item.index_set(), THEN=item[index_set], ELSE=0)

current_dir = os.path.split(os.path.abspath(__file__))[0]
solver_name = "bonmin"
solver_path = "{}\\..\\solvers\\CoinAll-1.6.0-win64-intel11.1\\bin\\bonmin.exe".format(current_dir)

model = ConcreteModel()
model.x1 = Var(domain=PositiveIntegers)
model.x2 = Var(domain=PositiveIntegers)
#model.x = Var([1,2], domain=PositiveIntegers)

model.c = Param(initialize=10)
model.supply = Param(initialize=100)
model.max_price = Param(initialize=20)

model.Objective = Objective(expr=model.x1*model.x2 - model.c*model.x1, sense=maximize)


model.time = ContinuousSet(bounds=(0,10))
model.xx = Var(model.time)
def _intX(m,i):
    return model.xx[i]**2
model.intX = Integral(model.time,wrt=model.time,rule=_intX)

def C1(model): 
    return model.x1 <= model.supply
model.C1 = Constraint(rule=C1)

def C2(model): 
    return model.x2 <= model.intX
model.C2 = Constraint(rule=C2)

print(">>Using the solver {NAME} in filepath {PATH}".format(NAME=solver_name, PATH=solver_path))
opt = SolverFactory(solver_name, executable=solver_path)
opt.options["print_level"] = 12
opt.options["wantsol"] = 1
results = opt.solve(model, logfile="{}\\solver.log".format(current_dir), keepfiles=True, tee=True, symbolic_solver_labels=True)

def print_value_s(array, padding=32, line_size=5):  # TODO generalize to all model components
    for index, item in enumerate(array):
        if hasattr(item, "value"): value_s = item.value
        elif hasattr(item, "valeus"): value_s = item.values
        else: value_s = "No value_s"
        entry = "{} = {}".format(str(item), value_s)
        end = "\n" if index % line_size == 0 else ""
        print(("{:<"+str(padding)+"}").format(entry), end=end)
    print()

def print_value_sx(array, padding=32, line_size=5):  # TODO generalize to all model components
    for index, item in enumerate(array):
        end = "\n" if index % line_size == 0 else ""
        print(("{:<"+str(padding)+"}").format(item), end=end)
    print()

print("Printing values for all variables")
print_value_s(model.component_data_objects(pyomo.environ.Var))
results.write()

model.intX.display() # CONTINUE

import code; code.interact(local=locals())