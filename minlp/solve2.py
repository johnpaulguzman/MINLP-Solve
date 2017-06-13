import os
from pyomo.environ import *
irange = lambda start, end: list(range(start, end+1))

current_dir = os.path.split(os.path.abspath(__file__))[0]
solver_name = "bonmin"
solver_path = current_dir + r"\..\solvers\CoinAll-1.6.0-win64-intel11.1\bin\bonmin.exe"

idx = {
    "t" : irange(1, 7),
    "f" : irange(1, 2),
    "w" : irange(1, 3),
    "r" : irange(1, 4),
}
idx["i"] = irange(1, 3)
idx["j"] = irange(1, 2 ** idx["i"][-1] - 1)
idx["k"] = irange(idx["i"][-1] + 1, 2 ** idx["i"][-1] - 1)

model = ConcreteModel()
## >> VARIABLES 
#  DECISION VARIABLES
model.P_ift = Var(idx["i"], idx["f"], idx["t"], domain=PositiveIntegers)
model.UTFW_jfwt = Var(idx["j"], idx["f"], idx["w"], idx["t"], domain=PositiveIntegers)
model.UTFR_jfrt = Var(idx["j"], idx["f"], idx["r"], idx["t"], domain=PositiveIntegers)
model.UTWR_jwrt = Var(idx["j"], idx["w"], idx["r"], idx["t"], domain=PositiveIntegers)
model.BUW_kwt = Var(idx["k"], idx["w"], idx["t"], domain=PositiveIntegers)
model.BUF_kft = Var(idx["k"], idx["f"], idx["t"], domain=PositiveIntegers)
model.ALLOC_ijt = Var(idx["i"], idx["j"], idx["t"], domain=PositiveIntegers)
model.SP_jt = Var(idx["j"], idx["t"], domain=PositiveReals)
#  SYSTEM VARIABLES
model.BEGINVF_jft = Var(idx["j"], idx["f"], idx["t"], domain=PositiveIntegers)
model.BEGINVW_jwt = Var(idx["j"], idx["w"], idx["t"], domain=PositiveIntegers)
model.ENDVF_jft = Var(idx["j"], idx["f"], idx["t"], domain=PositiveIntegers)
model.ENDVW_jwt = Var(idx["j"], idx["w"], idx["t"], domain=PositiveIntegers)
model.SPR_jt = Var(idx["j"], idx["t"], domain=PositiveReals)
model.ORDER_jrt = Var(idx["j"], idx["r"], idx["t"], domain=PositiveIntegers)
model.CUMORDER_jt = Var(idx["j"], idx["t"], domain=PositiveIntegers)
model.D_jrt = Var(idx["j"], idx["r"], idx["t"], domain=PositiveReals)
#  BINARY VARIABLES
model.BSUW_kwt = Var(idx["k"], idx["w"], idx["t"], domain=Boolean)
model.BSUF_kft = Var(idx["k"], idx["f"], idx["t"], domain=Boolean)
model.SETUPW_kwt = Var(idx["k"], idx["w"], idx["t"], domain=Boolean)
model.SETUPF_kft = Var(idx["k"], idx["f"], idx["t"], domain=Boolean)
model.INTROBW_kwt = Var(idx["k"], idx["w"], idx["t"], domain=Boolean)
model.INTROBF_kft = Var(idx["k"], idx["f"], idx["t"], domain=Boolean)
model.INTROBINCURW_kw = Var(idx["k"], idx["w"], domain=Boolean)
model.INTROBINCURF_kf = Var(idx["k"], idx["f"], domain=Boolean)
model.LOST_jrt = Var(idx["j"], idx["r"], idx["t"], domain=Boolean)
model.OP_jrt = Var(idx["j"], idx["r"], idx["t"], domain=Boolean)
model.QW_kwt = Var(idx["k"], idx["w"], idx["t"], domain=Boolean)
model.QF_kft = Var(idx["k"], idx["f"], idx["t"], domain=Boolean)
model.OFFER_jt = Var(idx["j"], idx["t"], domain=Boolean)
## << VARIABLES  # Refer to as model.var[i1, i2, i3, ...]

""" Domain options:
    Any: all possible values
    Reals : floating point values
    PositiveReals: strictly positive floating point values
    NonPositiveReals: non-positive floating point values
    NegativeReals: strictly negative floating point value
    NonNegativeReals: non-negative floating point values
    UnitInterval: floating point values in the interval [0,1]
    Integers: integer values
    PositiveIntegers: positive integer values
    NonPositiveIntegers: non-positive integer values
    NegativeIntegers: negative integer values
    NonNegativeIntegers: non-negative integer values
    Boolean: boolean values, which can be represented as False/True, 0/1, ’False’/’True’ and ’F’/’T’
"""

print("Using the solver {NAME} in filepath {PATH}".format(NAME=solver_name, PATH=solver_path))
opt = SolverFactory(solver_name, executable=solver_path)
results = opt.solve(model)

print("Print values for all variables")
for v in model.component_data_objects(Var):
  print(str(v), v.value)