import subprocess
from pyomo.environ import *
from pyomo.dae import *
from time import time

import config
from xlsx_reader import XLSXReader

import winsound
def beep(): winsound.Beep(300,2000)

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

def PyomoMinMax(a, b, do_min=True):
    if do_min: return base.expr.Expr_if(IF=a < b, THEN=a, ELSE=b)
    else: return base.expr.Expr_if(IF=a > b, THEN=a, ELSE=b)

def MultiMinMax(a, *bs, do_min=True):
    if bs: return MultiMinMax(PyomoMinMax(a, bs.pop(), do_min=do_min), *bs, do_min=do_min)
    else: return a

def SafeIdx(item, *index_set, default=0):
    return item[index_set] if index_set in item.index_set() else default

def PreSafeIdx(array, remove_leq=1):
    return [item for item in array if item > remove_leq]

def float_or_str(item):
    try: return float(item)
    except: return str(item).strip()

def dict_replace(string, dict):
    for key, value in dict.items():
        string = string.replace(key, value)
    return string

def RunMathematica(query, values, j, t, get_last_only=True):
    start_time = time()
    script_dir = "{}\\alpha_jt[{},{}].m".format(config.math_script_dir, j, t)
    run_script = dict_replace(query, values)
    print(">>Running script =====\n{}\n<<End of script ======".format(run_script))
    with open(script_dir, 'w') as file: file.write(run_script)
    run_script = r'"{}" -script "{}"'.format(config.math_exe, script_dir)
    process = subprocess.Popen(run_script, stdout=subprocess.PIPE)
    script_return = [float_or_str(line) for line in process.stdout.readlines()]
    print(">>Returned: {} | Time elapsed (sec): {}".format(script_return, time() - start_time))
    return script_return[-1] if get_last_only else script_return

input_reader = XLSXReader(config.input_path)
idx, params = input_reader.extract_idxParams()
model = ConcreteModel()
print(">>Using the solver {} running in {}".format(config.solver_name, config.solver_path))
print(">>Parsing .xlsx file in {}".format(config.input_path))
print(">>Loaded index values:")
for idx_name, idx_range in idx.items():
    print("{} = {}".format(idx_name, idx_range))
print(">>Loaded parameter values:")
for name, value in params.items():
    setattr(model, name, Param(*input_reader.get_idx(name, idx) , initialize=value, default=-1))
    stored_variable = getattr(model, name)  # OR use hasattr
    stored_value = list(stored_variable.values()) if type(stored_variable) == pyomo.core.base.param.IndexedParam else stored_variable.value
    print("model.{} = {}".format(name, stored_value))

##############################################################################################################################################################################################
def apply_alpha_contingency(model, alphas):
    idx = {"k": [4,5,6,7], "t": [1,2,3,4,5,6,7]}
    are_zeroes = [(bool(model.Contingency_k[k]), k) for k in idx["k"]]
    are_zeroes += [(True, 1)]
    case_number = sum([is_zero[0] for is_zero in are_zeroes])
    print(">>Contingency Case: {}".format(case_number))
    import code;code.interact(local=locals())
    if case_number == 0:  # Case when all thetas are 0
        return # Update nothing
    elif case_number == 1:  # Case when exactly 1 theta is 0
        update_k = [is_zero[1] for is_zero in are_zeroes if is_zero[0]][0]
        print(update_k)  # TODO
    else: # Case when more 1 thetas are 0
        update_ks = [is_zero[1] for is_zero in are_zeroes if is_zero[0]]
        print(update_ks)  # TODO
    import code;code.interact(local=locals())

def alpha_integrals(model, j, t):
    S1, S2, S3, S4, S5, S6, S7 = [model.SPR_jt[j,t] for j in idx["j"]]
    T4, T5, T6, T7 = [model.Contingency_k[k] for k in idx["k"]] 
    if j == 0:
        replace_dict = {
            "SPR_var": f"{S1}", 
            "B_var": f"Min[ {S2}, {S4}/(1+{T4}) - x ]",
            "A_var": f"Min[ {S3}, {S5}/(1+{T5}) - x, {S6}/(1+{T6}) - y, {S7}/(1+{T7}) - x - y ]",
            "mu_var": "{{ {{ {}, {}, {} }} }}".format(*model.RPMean_i.values()),
            "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*model.Covariance_iI.values()) }
        math_query = ''' ("ALPHA [0,%d] COMPUTATION")
            SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {x, -intlimit, SPR}, {y, -intlimit, B}, {z, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [0,%d]")
            ''' % tuple([t]*2)
    elif j == 1:
        replace_dict = {"SPR_var": f"{S1}", 
            "B_var": f"Min[ x - {S1} + {S2}, (x-{S1}+{S4})/(1+{T4}) - x ]", 
            "A_var": f"Min[ x - {S1} + {S3}, (x-{S1}+{S5})/(1+{T5}) - x, (x-{S1}+{S6})/(1+{T6}) - y, (x-{S1}+{S7})/(1+{T7}) - x - y ]",
            "mu_var": "{{ {{ {}, {}, {} }} }}".format(*model.RPMean_i.values()),
            "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*model.Covariance_iI.values())}
        math_query = ''' ("ALPHA [1,%d] COMPUTATION")
            SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {x, SPR, intlimit}, {y, -intlimit, B}, {z, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [1,%d]")
            ''' % tuple([t]*2)
    elif j == 2:
        replace_dict = {"SPR_var": f"{S2}", 
            "B_var": f"Min[ y - {S2} + {S1}, (y-{S2}+{S4})/(1+{T4}) - y ]", 
            "A_var":  f"Min[ y - {S2} + {S3}, (y-{S2}+{S5})/(1+{T5}) - x, (y-{S2}+{S6})/(1+{T6}) - y, (y-{S2}+{S7})/(1+{T7}) - x - y ]",
            "mu_var": "{{ {{ {}, {}, {} }} }}".format(*model.RPMean_i.values()),
            "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*model.Covariance_iI.values())}
        math_query = ''' ("ALPHA [2,%d] COMPUTATION")
            SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {y, SPR, intlimit}, {x, -intlimit, B}, {z, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [2,%d]")
            ''' % tuple([t]*2)
    elif j ==3:
        replace_dict = {"SPR_var": f"{S3}", 
            "B_var": f"Min[ y - {S3} + {S1}, (y-{S3}+{S5})/(1+{T5}) - y ]", 
            "A_var": f"Min[ y - {S3} + {S2},  (y-{S3}+{S4})/(1+{T4}) - z, (y-{S3}+{S6})/(1+{T6}) - y, (y-{S3}+{S7})/(1+{T7}) - z - y ]",
            "mu_var": "{{ {{ {}, {}, {} }} }}".format(*model.RPMean_i.values()),
            "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*model.Covariance_iI.values())}
        math_query = ''' ("ALPHA [3,%d] COMPUTATION")
            SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {y, SPR, intlimit}, {z, -intlimit, B}, {x, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [3,%d]")
            ''' % tuple([t]*2)
    elif j == 4:
        replace_dict = {"SPR_var": f"(y-{S2}+{S4})/(1+{T4}) - y", 
            "B_var": f"Max[ ({S4})/(1+{T4}) - x, (x-{S1}+{S4})/(1+{T4}) - x ]", 
            "A_var": f"Min[ (x+y)*(1+{T4}) - {S4} + {S3}, ((x+y)*(1+{T4})-{S4}+{S5})/(1+{T5}) - x, ((x+y)*(1+{T4})-{S4}+{S6})/(1+{T6}) - y, ((x+y)*(1+{T4})-{S4}+{S7})/(1+{T7}) - x - y ]",
            "mu_var": "{{ {{ {}, {}, {} }} }}".format(*model.RPMean_i.values()),
            "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*model.Covariance_iI.values())}
        math_query = ''' ("ALPHA [4,%d] COMPUTATION")
            SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {x, SPR, intlimit}, {y, B, intlimit}, {z, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [4,%d]")
            ''' % tuple([t]*2)
    elif j == 5: # UPDATED CONTINGENCY VERSION
        replace_dict = {"SPR_var": f"(z-{S3}+{S5})/(1+{T5}) - z", 
            "B_var": f"Max[ ({S5})/(1+{T5}) - x, (x-{S1}+{S5})/(1+{T5}) - x  ]", 
            "A_var": f"Min[ (x+z)*(1+{T5}) - {S5} + {S2}, ((x+z)*(1+{T5})-{S5}+{S4})/(1+{T4}) - x , ((x+z)*(1+{T5})-{S5}+{S6})/(1+{T6}) - z, ((x+z)*(1+{T5})-{S5}+{S7})/(1+{T7}) - x - z ]",
            "mu_var": "{{ {{ {}, {}, {} }} }}".format(*model.RPMean_i.values()),
            "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*model.Covariance_iI.values())}
        math_query = ''' ("ALPHA [5,%d] COMPUTATION")
            SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {x, SPR, intlimit}, {z, B, intlimit}, {y, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [5,%d]")
            ''' % tuple([t]*2)
    elif j == 6:
        replace_dict = {"SPR_var": f"(y-{S3}+{S6})/(1+{T6}) - y", 
            "B_var": f"Max[ ({S6})/(1+{T6}) - z, (z-{S2}+{S6})/(1+{T6}) - z ]", 
            "A_var": f"Min[ (z+x)*(1+{T6}) - {S6} + {S1}, ((z+x)*(1+{T6})-{S6}+{S4})/(1+{T4}) - z, ((z+x)*(1+{T6})-{S6}+{S5})/(1+{T5}) - x, ((z+x)*(1+{T6})-{S6}+{S7})/(1+{T7}) - z - x ]",
            "mu_var": "{{ {{ {}, {}, {} }} }}".format(*model.RPMean_i.values()),
            "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*model.Covariance_iI.values())}
        math_query = ''' ("ALPHA [6,%d] COMPUTATION")
            SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {z, SPR, intlimit}, {x, B, intlimit}, {y, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [6,%d]")
            ''' % tuple([t]*2)
    elif j == 7:
        replace_dict = {"SPR_var": f"((y+z)*(1+{T6})-{S6}+{S7})/(1+{T7}) - y - z", 
            "B_var": f"Max[ (z-{S3}+{S7})/(1+{T7}) - x - z, ((x+z)*(1+{T5})-{S5}+{S7})/(1+{T7}) - x - z ]", 
            "A_var": f"Max[ ({S7})/(1+{T7}) - x - y, (x-{S1}+{S7})/(1+{T7}) - x - y, (y-{S2}+{S7})/(1+{T7}) - x - y, ((x+y)*(1+{T4})-{S4}+{S7})/(1+{T7}) - x - y ]",
            "mu_var": "{{ {{ {}, {}, {} }} }}".format(*model.RPMean_i.values()),
            "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*model.Covariance_iI.values())}
        math_query = ''' ("ALPHA [7,%d] COMPUTATION")
            SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {x, SPR, intlimit}, {y, B, intlimit}, {z, A, intlimit}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [7,%d]")
            ''' % tuple([t]*2)
    else: raise Exception("ERROR: This program only works for i=[1..3], j=[1..7]")
    return RunMathematica(math_query, replace_dict, j, t)

def generate_alphas(model):                             
    alphas = {}
    for j, t in [(j,t) for t in idx["t"][:1] for j in [0]+idx["j"]]: 
        alphas[(j,t)] = alpha_integrals(model, j, t)
    for i,j in alphas.items(): print("  ALPHA",i,j)
    apply_alpha_contingency(model, alphas)
    for i,j in alphas.items(): print("  UPDATED ALPHA",i,j)
    import code;code.interact(local=locals())
##############################################################################################################################################################################################

generate_alphas(model)

## >> VARIABLES 

#  DECISION VARIABLES
model.P_ift = Var(idx["i"], idx["f"], idx["t"], domain=NonNegativeIntegers)
model.UTFW_jfwt = Var(idx["j"], idx["f"], idx["w"], idx["t"], domain=NonNegativeIntegers)
model.UTFR_jfrt = Var(idx["j"], idx["f"], idx["r"], idx["t"], domain=NonNegativeIntegers)
model.UTWR_jwrt = Var(idx["j"], idx["w"], idx["r"], idx["t"], domain=NonNegativeIntegers)
model.BUW_kwt = Var(idx["k"], idx["w"], idx["t"], domain=NonNegativeIntegers)
model.BUF_kft = Var(idx["k"], idx["f"], idx["t"], domain=NonNegativeIntegers)
model.ALLOC_ijt = Var(idx["i"], idx["j"], idx["t"], domain=NonNegativeIntegers)
#model.SP_jt = Var(idx["j"], idx["t"], domain=NonNegativeReals)

#  SYSTEM VARIABLES
model.BEGINVF_jft = Var(idx["j"], idx["f"], idx["t"], domain=NonNegativeIntegers)
model.BEGINVW_jwt = Var(idx["j"], idx["w"], idx["t"], domain=NonNegativeIntegers)
model.ENDINVF_jft = Var(idx["j"], idx["f"], idx["t"], domain=NonNegativeIntegers)
model.ENDINVW_jwt = Var(idx["j"], idx["w"], idx["t"], domain=NonNegativeIntegers)
#model.SPR_jt = Var(idx["j"], idx["t"], domain=NonNegativeReals)
model.ORDER_jrt = Var(idx["j"], idx["r"], idx["t"], domain=NonNegativeIntegers)
model.CUMORDER_jt = Var(idx["j"], idx["t"], domain=NonNegativeIntegers)
model.D_jrt = Var(idx["j"], idx["r"], idx["t"], domain=NonNegativeReals)
model.PC_ift = Var(idx["i"], idx["f"], idx["t"], domain=NonNegativeIntegers)
model.BACKORDER_jrt = Var(idx["j"], idx["r"], idx["t"], domain=NonNegativeReals)

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
model.Objective = Objective(expr=\
    sum(model.SP_jt[j,t] * model.ORDER_jrt[j,r,t] * (1 - model.LOST_jrt[j,r,t]) for t in idx["t"] for r in idx["r"] for j in idx["j"])\
    #-sum(model.P_ift[i,f,t] * model.PC_ift[i,f,t] for t in idx["t"] for f in idx["f"] for i in idx["i"])\
    -sum(model.P_ift[i,f,t] * model.VC_if[i,f] + model.FC_if[i,f] for t in idx["t"] for f in idx["f"] for i in idx["i"])\
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
    return model.BEGINVF_jft[i,f,t] == model.ENDINVF_jft[i,f,t-1] 
model.C451a = Constraint(idx["i"], idx["f"], PreSafeIdx(idx["t"]), rule=C451a)

def C451b(model, i, f): 
    return model.BEGINVF_jft[i,f,1] == model.INITIALINVF_if[i,f]
model.C451b = Constraint(idx["i"], idx["f"], rule=C451b)

def C452a(model, i, w, t): 
    return model.BEGINVW_jwt[i,w,t] == model.ENDINVW_jwt[i,w,t-1] + sum(SafeIdx(model.UTFW_jfwt, i,f,w,t-model.x_fw[f,w]) for f in idx["f"])
model.C452a = Constraint(idx["i"], idx["w"], PreSafeIdx(idx["t"]), rule=C452a)

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
    return model.BUF_kft[k,f,t] >= 1 - model.M * (1 - model.BSUF_kft[k,f,t])
model.C4514b = Constraint(idx["k"], idx["f"], idx["t"], rule=C4514b)

def C4515a(model, k, w, t): 
    return model.BUW_kwt[k,w,t] <= model.M * model.BSUW_kwt[k,w,t]
model.C4515a = Constraint(idx["k"], idx["w"], idx["t"], rule=C4515a)

def C4515b(model, k, w, t): 
    return model.BUW_kwt[k,w,t] >= 1 - model.M * (1 - model.BSUW_kwt[k,w,t])
model.C4515b = Constraint(idx["k"], idx["w"], idx["t"], rule=C4515b)

def CS4516a(model, k, f, t): 
    return sum(model.BSUF_kft[k,f,t_i] for t_i in range(1,t+1)) <= 1 + model.M * (1 - model.INTROBF_kft[k,f,t])
model.CS4516a = Constraint(idx["k"], idx["f"], idx["t"], rule=CS4516a)

def CS4516b(model, k, f, t): 
    return sum(model.BSUF_kft[k,f,t_i] for t_i in range(1,t+1)) >= 2 * model.QF_kft[k,f,t] - model.M * model.INTROBF_kft[k,f,t]
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
    return sum(model.BSUW_kwt[k,w,t_i] for t_i in range(1,t+1)) >= 2 * model.QW_kwt[k,w,t] - model.M * model.INTROBW_kwt[k,w,t]
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
    return model.INTROBINCURF_kf[k,f] == PyomoMinMax(sum(model.INTROBF_kft[k,f,t] for t in idx["t"]), 1, do_min=True)
model.CS4518 = Constraint(idx["k"], idx["f"], rule=CS4518)

def CS4519(model, k, w): 
    return model.INTROBINCURW_kw[k,w] == PyomoMinMax(sum(model.INTROBW_kwt[k,w,t] for t in idx["t"]), 1, do_min=True)
model.CS4519 = Constraint(idx["k"], idx["w"], rule=CS4519)

def CS4520a(model, k, f, t): 
    return model.BSUF_kft[k,f,t] >= model.SETUPF_kft[k,f,t]
model.CS4520a = Constraint(idx["k"], idx["f"], idx["t"], rule=CS4520a)

def CS4520b(model, k, f, t): 
    return SafeIdx(model.BSUF_kft, k,f,t-1) <= model.M * (1 - model.SETUPF_kft[k,f,t])
model.CS4520b = Constraint(idx["k"], idx["f"], idx["t"], rule=CS4520b)

def CS4520c(model, k, f, t): 
    return SafeIdx(model.BSUF_kft, k,f,t-1) >= model.BSUF_kft[k,f,t] - model.M * model.SETUPF_kft[k,f,t]
model.CS4520c = Constraint(idx["k"], idx["f"], idx["t"], rule=CS4520c)

def CS4521a(model, k, w, t): 
    return model.BSUW_kwt[k,w,t] >= model.SETUPW_kwt[k,w,t]
model.CS4521a = Constraint(idx["k"], idx["w"], idx["t"], rule=CS4521a)

def CS4521b(model, k, w, t): 
    return SafeIdx(model.BSUW_kwt, k,w,t-1) <= model.M * (1 - model.SETUPW_kwt[k,w,t])
model.CS4521b = Constraint(idx["k"], idx["w"], idx["t"], rule=CS4521b)

def CS4521c(model, k, w, t): 
    return SafeIdx(model.BSUW_kwt, k,w,t-1) >= model.BSUW_kwt[k,w,t] - model.M * model.SETUPW_kwt[k,w,t]
model.CS4521c = Constraint(idx["k"], idx["w"], idx["t"], rule=CS4521c)

def CS4522(model, i, f, t): 
    return model.ENDINVF_jft[i,f,t] == model.BEGINVF_jft[i,f,t] + model.P_ift[i,f,t] - sum(model.UTFW_jfwt[i,f,w,t] for w in idx["w"]) - sum(model.UTFR_jfrt[i,f,r,t] for r in idx["r"]) - sum(model.BUF_kft[k,f,t] * model.Y_ij[i,k] for k in idx["k"])
model.CS4522 = Constraint(idx["i"], idx["f"], idx["t"], rule=CS4522)

def CS4523(model, i, w, t): 
    return model.ENDINVW_jwt[i,w,t] == model.BEGINVW_jwt[i,w,t] - sum(model.UTWR_jwrt[i,w,r,t] for r in idx["r"]) - sum(model.BUW_kwt[k,w,t] * model.Y_ij[i,k] for k in idx["k"])
model.CS4523 = Constraint(idx["i"], idx["w"], idx["t"], rule=CS4523)

def CS4524(model, k, f, t): 
    return model.ENDINVF_jft[k,f,t] == model.BEGINVF_jft[k,f,t] + model.BUF_kft[k,f,t] - sum(model.UTFW_jfwt[k,f,w,t] for w in idx["w"]) - sum(model.UTFR_jfrt[k,f,r,t] for r in idx["r"])
model.CS4524 = Constraint(idx["k"], idx["f"], idx["t"], rule=CS4524)

def CS4525(model, k, w, t): 
    return model.ENDINVW_jwt[k,w,t] == model.BEGINVW_jwt[k,w,t] + model.BUW_kwt[k,w,t] - sum(model.UTWR_jwrt[k,w,r,t] for r in idx["r"])
model.CS4525 = Constraint(idx["k"], idx["w"], idx["t"], rule=CS4525)

def CS4526(model, i, j, t): 
    return model.ALLOC_ijt[i,j,t] == (sum(model.BEGINVW_jwt[j,w,t] for w in idx["w"]) + sum(model.BEGINVF_jft[j,f,t] for f in idx["f"])) * model.Y_ij[i,j]
model.CS4526 = Constraint(idx["i"], idx["j"], idx["t"], rule=CS4526)

def CS4527a(model, j, t): 
    return sum(model.BEGINVF_jft[j,f,t] for f in idx["f"]) + sum(model.BEGINVW_jwt[j,w,t] for w in idx["w"]) <= model.LE1 + model.M * model.OFFER_jt[j,t]
model.CS4527a = Constraint(idx["j"], idx["t"], rule=CS4527a)

def CS4527b(model, j, t): 
    return sum(model.BEGINVF_jft[j,f,t] for f in idx["f"]) + sum(model.BEGINVW_jwt[j,w,t] for w in idx["w"]) >= 1 - model.M * (1 - model.OFFER_jt[j,t])
model.CS4527b = Constraint(idx["j"], idx["t"], rule=CS4527b)

def CS4528a(model, j, r, t): # FAILED
    return model.ORDER_jrt[j,r,t] <= model.LE1 + model.M * SafeIdx(model.OP_jrt, j,r,t-model.x_r[r])
model.CS4528a = Constraint(idx["j"], idx["r"], idx["t"], rule=CS4528a)

def CS4528b(model, j, r, t): # FAILED
    return model.ORDER_jrt[j,r,t] >= 1 - model.M * (1 - SafeIdx(model.OP_jrt, j,r,t-model.x_r[r]))
model.CS4528b = Constraint(idx["j"], idx["r"], idx["t"], rule=CS4528b)

def CS4528c(model, j, r, t): # FAILED
    return SafeIdx(model.OP_jrt, j,r,t-model.x_r[r]) >= model.LOST_jrt[j,r,t]
model.CS4528c = Constraint(idx["j"], idx["r"], idx["t"], rule=CS4528c)

def CS4528d(model, j, r, t): 
    return SafeIdx(model.OFFER_jt, j,t-model.x_r[r]) <= model.M * (1 - model.LOST_jrt[j,r,t])
model.CS4528d = Constraint(idx["j"], idx["r"], idx["t"], rule=CS4528d)

def CS4528e(model, j, r, t): # FAILED
    return SafeIdx(model.OFFER_jt, j,t-model.x_r[r]) >= SafeIdx(model.OP_jrt, j,r,t-model.x_r[r]) - model.M * model.LOST_jrt[j,r,t]
model.CS4528e = Constraint(idx["j"], idx["r"], idx["t"], rule=CS4528e)

def CS4529(model, j, r, t): 
    return sum(SafeIdx(model.UTFR_jfrt, j,f,r,t-model.x_fr[f,r]) for f in idx["f"]) + sum(SafeIdx(model.UTWR_jwrt, j,w,r,t-model.x_wr[w,r]) for w in idx["w"]) <= model.ORDER_jrt[j,r,t] * (1 - model.LOST_jrt[j,r,t]) + SafeIdx(model.BACKORDER_jrt, j,r,t-1)
model.CS4529 = Constraint(idx["j"], idx["r"], idx["t"], rule=CS4529)

def CS4530(model, j, r, t): # FAILED
    return model.BACKORDER_jrt[j,r,t] == model.ORDER_jrt[j,r,t] * (1 - model.LOST_jrt[j,r,t]) + SafeIdx(model.BACKORDER_jrt, j,r,t-1) - sum(SafeIdx(model.UTFR_jfrt, j,f,r,t-model.x_fr[f,r]) for f in idx["f"]) - sum(SafeIdx(model.UTWR_jwrt, j,w,r,t-model.x_wr[w,r]) for w in idx["w"])
model.CS4530 = Constraint(idx["j"], idx["r"], idx["t"], rule=CS4530)

def CS4531(model, j, r, t): # FAILED?
    if t > model.x_r[r]: return model.ORDER_jrt[j,r,t] == model.D_jrt[j,r,t] - SafeIdx(model.BACKORDER_jrt, j,r,t-1)
    else: return model.ORDER_jrt[j,r,t] <= model.D_jrt[j,r,t] - SafeIdx(model.BACKORDER_jrt, j,r,t-1)
model.CS4531 = Constraint(idx["j"], idx["r"], idx["t"], rule=CS4531)

#def CS4532(model, j, t): # FAILED
#    return model.SP_jt[j,t] <= sum(model.SP_jt[i,t] * model.Y_ij[i,j] for i in idx["i"])
#model.CS4532 = Constraint(idx["j"], idx["t"], rule=CS4532)

#def CS4533(model, j, t): # FAILED
#    return model.SPR_jt[j,t] == model.SP_jt[j,t] * (1 + model.MARKUP)
#model.CS4533 = Constraint(idx["j"], idx["t"], rule=CS4533)

#def CS4534(model, i, f, t): 
#    return model.PC_ift[i,f,t] == model.VC_if[i,f] + model.FC_if[i,f] / model.P_ift[i,f,t]
#model.CS4534 = Constraint(idx["i"], idx["f"], idx["t"], rule=CS4534)

def CS4535(model, j, t): # FAILED
    return model.CUMORDER_jt[j,t] == sum(model.ORDER_jrt[j,r,t_i] for t_i in range(1, t+1) for r in idx["r"])
model.CS4535 = Constraint(idx["j"], idx["t"], rule=CS4535)

def CS463(model, j, r, t): # TEST
    return model.D_jrt[j,r,t] == model.alpha_jt[j,t] * model.Lambda_rt[r,t]
model.CS463 = Constraint(idx["j"], idx["r"], idx["t"], rule=CS463)


## >> SOLVE
print(">>Using the solver {NAME} in filepath {PATH}".format(NAME=config.solver_name, PATH=config.solver_path))
opt = SolverFactory(config.solver_name, executable=config.solver_path)  # solver_io=solver_io)
for key, value in config.solver_options.items(): opt.options[key] = value
try:
    start_time = time()
    results = opt.solve(model, keepfiles=True, tee=True)  # , symbolic_solver_labels=True)
    end_time = time()
    print("Printing values for all variables")
    print_vars(model)
    print("Time elapsed: {}".format(round(end_time - start_time)))
    #import sys; sys.stdout = open('model.txt', 'w'); model.display()
    results.write()
except Exception as e:
    print(e)
    pass
