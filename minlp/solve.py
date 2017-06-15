import pyomo.environ
import os

model = pyomo.environ.ConcreteModel()

model.market = pyomo.environ.Set(initialize=['market'])

model.ask_price = pyomo.environ.Param(model.market, initialize={'market' : 12})
model.bid_price = pyomo.environ.Param(model.market, initialize={'market' : 10})
model.ask_liquidity = pyomo.environ.Param(model.market, initialize={'market' : 100})
model.bid_liquidity = pyomo.environ.Param(model.market, initialize={'market' : 100})

model.VOLUME_BUY = pyomo.environ.Var(model.market, within = pyomo.environ.NonNegativeReals)
model.VOLUME_SELL = pyomo.environ.Var(model.market, within = pyomo.environ.NonNegativeReals)

def max_buy(model, market):
    return model.VOLUME_BUY[market] <= model.ask_liquidity[market]

model.max_buy_equation = pyomo.environ.Constraint(model.market, rule=max_buy)

def max_sell(model, market):
    return model.VOLUME_SELL[market] <= model.bid_liquidity[market]

model.max_sell_equation = pyomo.environ.Constraint(model.market, rule=max_sell)

def objective_component1(model):
    return sum(model.VOLUME_BUY[market] * model.ask_price[market] for market in model.market)

model.obj_component1 = pyomo.environ.Expression(rule=objective_component1)

def objective_component2(model):
    return - sum(model.VOLUME_SELL[market] * model.bid_price[market] for market in model.market)

model.obj_component2 = pyomo.environ.Expression(rule=objective_component2)
model.objective = pyomo.environ.Objective(expr=model.obj_component1 + model.obj_component2, sense=-1)

solver_name = "bonmin"
solver_path = os.path.split(os.path.abspath(__file__))[0] + r"\..\solvers\CoinAll-1.6.0-win64-intel11.1\bin\bonmin.exe"
print("Using the solver {NAME} in filepath {PATH}".format(NAME=solver_name, PATH=solver_path))
opt = pyomo.environ.SolverFactory(solver_name, executable=solver_path)
results = opt.solve(model)
print("Print values for all variables")
for v in model.component_data_objects(pyomo.environ.Var):
    print(str(v), v.value)
for v in model.component_data_objects(pyomo.environ.Param):
    print(str(v), v.items())