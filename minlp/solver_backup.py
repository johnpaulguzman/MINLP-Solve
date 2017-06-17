from xlsx_reader import XLSXReader
from pyomo.environ import *
from pyomo.dae import *
import os
import pprint; indented_print = pprint.PrettyPrinter(indent=4).pprint
#import code; code.interact(local=locals())

def PyomoMin(a, b):
    return base.expr.Expr_if(IF=(a > b), THEN=(a), ELSE=(b))

def SafeIdx(item, *index_set, default=0):
    #if index_set not in item.index_set(): print("<Did guard {} with index {}".format(item.name, index_set))
    return item[index_set] if index_set in item.index_set() else default

current_dir = os.path.split(os.path.abspath(__file__))[0]
solver_name = "bonmin"
solver_path = "{}\\..\\solvers\\CoinAll-1.6.0-win64-intel11.1\\bin\\bonmin.exe".format(current_dir)
#solver_io = 'nl'
input_path = "{}\\Parameters.xlsx".format(current_dir)
input_reader = XLSXReader(input_path)
idx, params = input_reader.extract_idxParams()
model = ConcreteModel()
print(">>Using the solver {} running in {}".format(solver_name, solver_path))
print(">>Parsing .xlsx file in {}".format(input_path))
print(">>Loaded index values:")
for idx_name, idx_range in idx.items():
    print("{} = {}".format(idx_name, idx_range))
print(">>Loaded parameter values:")
for name, value in params.items():
    setattr(model, name, Param(*input_reader.get_idx(name, idx) , initialize=value, default=-1))
    stored_variable = getattr(model, name)  # TODO USE hasattr
    stored_value = list(stored_variable.values()) if type(stored_variable) == pyomo.core.base.param.IndexedParam else stored_variable.value
    print("model.{} = {}".format(name, stored_value))

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
model.ENDINVF_jft = Var(idx["j"], idx["f"], idx["t"], domain=PositiveIntegers)
model.ENDINVW_jwt = Var(idx["j"], idx["w"], idx["t"], domain=PositiveIntegers)
model.SPR_jt = Var(idx["j"], idx["t"], domain=PositiveReals)
model.ORDER_jrt = Var(idx["j"], idx["r"], idx["t"], domain=PositiveIntegers)
model.CUMORDER_jt = Var(idx["j"], idx["t"], domain=PositiveIntegers)
model.D_jrt = Var(idx["j"], idx["r"], idx["t"], domain=PositiveReals)
model.PC_ift = Var(idx["i"], idx["f"], idx["t"], domain=PositiveIntegers)
model.BACKORDER_jrt = Var(idx["j"], idx["r"], idx["t"], domain=PositiveReals)

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


## >> OBJECTIVE
model.Objective = Objective(expr=sum(model.SP_jt[j,t] * model.ORDER_jrt[j,r,t] * (1 - model.LOST_jrt[j,r,t]) for t in idx["t"] for r in idx["r"] for j in idx["j"])\
    -sum(model.P_ift[i,f,t] * model.PC_ift[i,f,t] for t in idx["t"] for f in idx["f"] for i in idx["i"])\
    -sum(model.ENDINVF_jft[j,f,t] * model.ICF_jf[j,f] for t in idx["t"] for f in idx["f"] for j in idx["j"])\
    -sum(model.ENDINVW_jwt[j,w,t] * model.ICW_jw[j,w] for t in idx["t"] for w in idx["w"] for j in idx["j"])\
    -sum(model.UTFW_jfwt[j,f,w,t] * model.TCFW_jfw[j,f,w] for t in idx["t"] for j in idx["j"] for f in idx["f"] for w in idx["w"])\
    -sum(model.UTFR_jfrt[j,f,r,t] * model.TCFR_jfr[j,f,r] for t in idx["t"] for j in idx["j"] for f in idx["f"] for r in idx["r"])\
    -sum(model.UTWR_jwrt[j,w,r,t] * model.TCWR_jwr[j,w,r] for t in idx["t"] for j in idx["j"] for w in idx["w"] for r in idx["r"])\
    -sum(model.BACKORDER_jrt[j,r,t] * model.BOC_r[r] for t in idx["t"] for r in idx["r"] for j in idx["j"])\
    -sum(model.INTROBINCURW_kw[k,w] * model.INTCOSTW_kw[k,w] for w in idx["w"] for k in idx["k"])\
    -sum(model.INTROBINCURF_kf[k,f] * model.INTCOSTF_kf[k,f] for f in idx["f"] for k in idx["k"])\
    -sum(model.SETUPW_kwt[k,w,t] * model.SETCOSTW_kw[k,w] for t in idx["t"] for w in idx["w"] for k in idx["k"])\
    -sum(model.SETUPF_kft[k,f,t] * model.SETCOSTF_kf[k,f] for t in idx["t"] for f in idx["f"] for k in idx["k"])\
    -sum(model.BUW_kwt[k,w,t] * model.BCW_kw[k,w] for t in idx["t"] for w in idx["w"] for k in idx["k"])\
    -sum(model.BUF_kft[k,f,t] * model.BCF_kf[k,f] for t in idx["t"] for f in idx["f"] for k in idx["k"]), 
    sense=maximize)

## >> CONSTRAINTS
def C451a(model, i, f, t): 
    return model.BEGINVF_jft[i,f,t] == model.ENDINVF_jft[i,f,t] 
model.C451a = Constraint(idx["i"], idx["f"], idx["t"], rule=C451a)

def C451b(model, i, f): 
    return model.BEGINVF_jft[i,f,1] == model.INITIALINVF_if[i,f]
model.C451b = Constraint(idx["i"], idx["f"], rule=C451b)

def C452a(model, i, w, t): return model.BEGINVW_jwt[i,w,t] == SafeIdx(model.ENDINVW_jwt, i,w,t-1)
model.C452a = Constraint(idx["i"], idx["w"], idx["t"], rule=C452a)

def C452b(model, i, w): 
    return model.BEGINVW_jwt[i,w,1] == model.INITIALINVW_iw[i,w]
model.C452b = Constraint(idx["i"], idx["w"], rule=C452b)

def C453(model, k, f, t): 
    return model.BEGINVF_jft[k,f,t] == SafeIdx(model.ENDINVF_jft, k,f,t-1)
model.C453 = Constraint(idx["k"], idx["f"], idx["t"], rule=C453)

def C454(model, k, w, t): 
    return model.BEGINVW_jwt[k,w,t] == SafeIdx(model.ENDINVW_jwt, k,w,t-1) + sum(SafeIdx(model.UTFW_jfwt, k,f,w,t-model.x_fw[f,w]) for f in idx["f"]) 
model.C454 = Constraint(idx["k"], idx["w"], idx["t"], rule=C454)

def C455(model, i, f, t): 
    return model.P_ift[i,f,t] <= model.PCAP_if[i,f]
model.C455 = Constraint(idx["i"], idx["f"], idx["t"], rule=C455)

def C456(model, f, t): 
    return sum(model.BEGINVF_jft[j,f,t] / model.ST_j[j] for j in idx["j"]) <= model.ICAP_f[f]
model.C456 = Constraint(idx["f"], idx["t"], rule=C456)

def C457(model, w, t): 
    return sum(model.BEGINVW_jwt[j,w,t] / model.ST_j[j] for j in idx["j"]) <= model.WCAP_w[w]
model.C457 = Constraint(idx["w"], idx["t"], rule=C457)

def C458(model, j, f, t): 
    return sum(model.UTFW_jfwt[j,f,w,t] for w in idx["w"]) + sum(model.UTFR_jfrt[j,f,r,t] for r in idx["r"]) <= model.BEGINVF_jft[j,f,t]
model.C458 = Constraint(idx["j"], idx["f"], idx["t"], rule=C458)

def C459(model, j, w, t): 
    return sum(model.UTWR_jwrt[j,w,r,t] for r in idx["r"]) <= model.BEGINVW_jwt[j,w,t]
model.C459 = Constraint(idx["j"], idx["w"], idx["t"], rule=C459)

def C4510(model, i, k, f, t): 
    return model.BUF_kft[k,f,t] <= model.BEGINVF_jft[i,f,t] - sum(model.UTFW_jfwt[i,f,w,t] for w in idx["w"]) - sum(model.UTFR_jfrt[i,f,r,t] for r in idx["r"]) + model.M * (1 - model.Y_ij[i,k])
model.C4510 = Constraint(idx["i"], idx["k"], idx["f"], idx["t"], rule=C4510)

def C4511(model, i, k, w, t): 
    return model.BUW_kwt[k,w,t] <= model.BEGINVW_jwt[i,w,t] - sum(model.UTWR_jwrt[i,w,r,t] for r in idx["r"]) + model.M * (1 - model.Y_ij[i,k])
model.C4511 = Constraint(idx["i"], idx["k"], idx["w"], idx["t"], rule=C4511)

def C4512(model, k, f, t): 
    return model.BUF_kft[k,f,t] <= model.BCAPF_kf[k,f]
model.C4512 = Constraint(idx["k"], idx["f"], idx["t"], rule=C4512)

def C4513(model, k, w, t): 
    return model.BUW_kwt[k,w,t] <= model.BCAPW_kw[k,w]
model.C4513 = Constraint(idx["k"], idx["w"], idx["t"], rule=C4513)

def C4514a(model, k, f, t): 
    return model.BUF_kft[k,f,t] <= model.M * model.BSUF_kft[k,f,t]
model.C4514a = Constraint(idx["k"], idx["f"], idx["t"], rule=C4514a)

def C4514b(model, k, f, t): 
    return model.BUF_kft[k,f,t] >= model.M * (1 - model.BSUF_kft[k,f,t])
model.C4514b = Constraint(idx["k"], idx["f"], idx["t"], rule=C4514b)

def C4515a(model, k, w, t): 
    return model.BUW_kwt[k,w,t] <= model.M * model.BSUW_kwt[k,w,t]
model.C4515a = Constraint(idx["k"], idx["w"], idx["t"], rule=C4515a)

def C4515b(model, k, w, t): 
    return model.BUW_kwt[k,w,t] >= model.M * (1 - model.BSUW_kwt[k,w,t])
model.C4515b = Constraint(idx["k"], idx["w"], idx["t"], rule=C4515b)

def CS4516a(model, k, f, t): 
    return sum(model.BSUF_kft[k,f,t_i] for t_i in range(1,t+1)) <= 1 + model.M * (1 - model.INTROBF_kft[k,f,t])
model.CS4516a = Constraint(idx["k"], idx["f"], idx["t"], rule=CS4516a)

def CS4516b(model, k, f, t): 
    return sum(model.BSUF_kft[k,f,t_i] for t_i in range(1,t+1)) >= 2 * model.QF_kft[k,f,t] - model.M * (1 - model.INTROBF_kft[k,f,t])
model.CS4516b = Constraint(idx["k"], idx["f"], idx["t"], rule=CS4516b)

def CS4516c(model, k, f, t): 
    return sum(model.BSUF_kft[k,f,t_i] for t_i in range(1,t+1)) >= model.QF_kft[k,f,t]
model.CS4516c = Constraint(idx["k"], idx["f"], idx["t"], rule=CS4516c)

def CS4516d(model, k, f, t): 
    return sum(model.BSUF_kft[k,f,t_i] for t_i in range(1,t+1)) <= model.M * model.QF_kft[k,f,t]
model.CS4516d = Constraint(idx["k"], idx["f"], idx["t"], rule=CS4516d)

def CS4516e(model, k, f, t): 
    return model.QF_kft[k,f,t] >= model.INTROBF_kft[k,f,t]
model.CS4516e = Constraint(idx["k"], idx["f"], idx["t"], rule=CS4516e)

def CS4517a(model, k, w, t): 
    return sum(model.BSUW_kwt[k,w,t_i] for t_i in range(1,t+1)) <= 1 + model.M * (1 - model.INTROBW_kwt[k,w,t])
model.CS4517a = Constraint(idx["k"], idx["w"], idx["t"], rule=CS4517a)

def CS4517b(model, k, w, t): 
    return sum(model.BSUW_kwt[k,w,t_i] for t_i in range(1,t+1)) >= 2 * model.QW_kwt[k,w,t] - model.M * (1 - model.INTROBW_kwt[k,w,t])
model.CS4517b = Constraint(idx["k"], idx["w"], idx["t"], rule=CS4517b)

def CS4517c(model, k, w, t): 
    return sum(model.BSUW_kwt[k,w,t_i] for t_i in range(1,t+1)) >= model.QW_kwt[k,w,t]
model.CS4517c = Constraint(idx["k"], idx["w"], idx["t"], rule=CS4517c)

def CS4517d(model, k, w, t): 
    return sum(model.BSUW_kwt[k,w,t_i] for t_i in range(1,t+1)) <= model.M * model.QW_kwt[k,w,t]
model.CS4517d = Constraint(idx["k"], idx["w"], idx["t"], rule=CS4517d)

def CS4517e(model, k, w, t): 
    return model.QW_kwt[k,w,t] >= model.INTROBW_kwt[k,w,t]
model.CS4517e = Constraint(idx["k"], idx["w"], idx["t"], rule=CS4517e)

def CS4518(model, k, f): 
    return model.INTROBINCURF_kf[k,f] == PyomoMin(sum(model.INTROBF_kft[k,f,t] for t in idx["t"]), 1)
model.CS4518 = Constraint(idx["k"], idx["f"], rule=CS4518)

## >> SOLVE
print(">>Using the solver {NAME} in filepath {PATH}".format(NAME=solver_name, PATH=solver_path))
opt = SolverFactory(solver_name, executable=solver_path)  # solver_io=solver_io)
opt.options["print_level"] = 12
opt.options["wantsol"] = 1
results = opt.solve(model, logfile="{}\\solver.log".format(current_dir), keepfiles=True, tee=True)  # , symbolic_solver_labels=True)


def row_print(array, padding=32, line_size=5):
    for index, item in enumerate(array):
        end = "\n" if index % line_size == 0 else ""
        print(("{:<"+str(padding)+"}").format(item), end=end)
    print()

def print_vars(model):  # TODO generalize to all model components
    var_list = model.component_data_objects(pyomo.environ.Var)
    entries = []
    for item in var_list:
        if hasattr(item, "value"): value_s = item.value
        elif hasattr(item, "valeus"): value_s = item.values
        else: value_s = "No value_s"
        entries += ["{} = {}".format(str(item), value_s)]
    row_print(entries)

print("Printing values for all variables")
print_vars(model)

#results.write()
#import sys; sys.stdout = open('model.txt', 'w'); model.display()