import os
import subprocess

import config
from time import time

################################################################################################
from threading import Thread
import functools

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

def safe_execute(function, *args):
    try:
        return function(*args)
    except:
        return None
################################################################################################

class AlphaGenerator:
    def __init__(self, model, idx):
        self.model = model
        self.idx = idx


    def float_or_str(self, item):
        try: return float(item)
        except: return str(item).strip()


    def dict_replace(self, string, dict):
        for key, value in dict.items(): string = string.replace(key, value)
        return string


    @timeout(config.mathematica_timeout)
    def RunMathematica(self, query, values, j, t, get_last_only=True):
        #return t for testing
        start_time = time()
        script_dir = "{}\\alpha_jt[{},{}].m".format(config.math_script_dir, j, t)
        run_script = self.dict_replace(query, values)
        print(">>Running script =====\n{}\n<<End of script ======".format(run_script))
        with open(script_dir, 'w') as file: file.write(run_script)
        run_script = r'"{}" -script "{}"'.format(config.math_exe, script_dir)
        self.process = subprocess.Popen(run_script, stdout=subprocess.PIPE)
        script_return = [self.float_or_str(line) for line in self.process.stdout.readlines()]
        print(">>Returned: {} | Time elapsed (sec): {}".format(script_return, time() - start_time))
        return script_return[-1] if get_last_only else script_return


    def make_table(self, alphas):
        out = ""
        for index, item in enumerate(alphas.values()):
            end = "\n" if index % len([0]+self.idx["j"]) == 0 else ""
            out += (end + ("{:<"+str(24)+"}").format(item))
        return out


    def generate_alphas(self):
        if not os.path.exists(config.math_script_dir): os.makedirs(config.math_script_dir)                           
        self.temp_alphas = {}
        if config.alpha_generator_option == config.AlphaOptions.calculate_all_t:
            for j, t in [(j,t) for t in self.idx["t"] for j in [0]+self.idx["j"]]: # generate for all t
                self.temp_alphas[(j,t)] = self.mega_integrals(j, t)
        elif config.alpha_generator_option == config.AlphaOptions.calculate_first_t:
            for j, t in [(j,t) for t in self.idx["t"][:1] for j in [0]+self.idx["j"]]: # just generate for t=1
                self.temp_alphas[(j,t)] = self.mega_integrals(j, t)
            for other_t in self.idx["t"][1:]: # copy t=1 to other t's
                for j in [0]+self.idx["j"]: self.temp_alphas[(j,other_t)] = self.temp_alphas[(j,1)]
        else:
            raise Exception("Alpha generator option not valid!")
        self.apply_alpha_contingency(self.temp_alphas)

        return self.validate_alphas(self.temp_alphas)


    def is_float(self, item):
        try:
            float(item)
            return True
        except:
            return False


    def mega_integrals(self, j, t):
        S1, S2, S3, S4, S5, S6, S7 = [self.model.SPR_jt[j,t] for j in self.idx["j"]]
        T4, T5, T6, T7 = [self.model.Contingency_k[k] for k in self.idx["k"]] 
        out_options = []
        if j == 0:
            b_options = [f"{S2}", f"{S4}/(1+{T4}) - x"]
            a_options = [f"{S3}", f"{S5}/(1+{T5}) - x", f"{S6}/(1+{T6}) - y", f"{S7}/(1+{T7}) - x - y"]
            for bo, ao in [(bo,ao) for bo in b_options for ao in a_options]:
                replace_dict = {
                    "SPR_var": f"{S1}", 
                    "B_var": bo,
                    "A_var": ao,
                    "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                    "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values()) }
                math_query = ''' ("ALPHA [0,%d] COMPUTATION")
                    SPR = SPR_var;
                    B = B_var;
                    A = A_var;
                    mu = mu_var;  sig = sig_var;
                    h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                    mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                    f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                    intres = Integrate[f, {x, -intlimit, SPR}, {y, -intlimit, B}, {z, -intlimit, A}];
                    Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [0,%d]")
                    ''' % tuple([t]*2)
                try:
                    out_options += [float(self.RunMathematica(math_query, replace_dict, j, t))]
                except Exception as e:
                    print("Integral output ignored due to: " + str(e))
                    self.process.kill()
            return min(out_options)
        elif j == 1:
            b_options = [ f"(x - {S1} + {S2})", f"((x-{S1}+{S4})/(1+{T4}) - x)" ]
            a_options = [ f"(x - {S1} + {S3})", f"((x-{S1}+{S5})/(1+{T5}) - x)", f"((x-{S1}+{S6})/(1+{T6}) - y)", f"((x-{S1}+{S7})/(1+{T7}) - x - y)" ]
            for bo, ao in [(bo,ao) for bo in b_options for ao in a_options]:
                replace_dict = {"SPR_var": f"{S1}", 
                    "B_var": bo, 
                    "A_var": ao,
                    "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                    "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
                math_query = ''' ("ALPHA [1,%d] COMPUTATION")
                    SPR = SPR_var;
                    B = B_var;
                    A = A_var;
                    mu = mu_var;  sig = sig_var;
                    h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                    mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                    f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                    intres = Integrate[f, {x, SPR, intlimit}, {y, -intlimit, B}, {z, -intlimit, A}];
                    Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [1,%d]")
                    ''' % tuple([t]*2)
                try:
                    out_options += [float(self.RunMathematica(math_query, replace_dict, j, t))]
                except Exception as e:
                    print("Integral output ignored due to: " + str(e))
                    self.process.kill()
            return min(out_options)
        elif j == 2:
            b_options = [ f"y - {S2} + {S1}", f"(y-{S2}+{S4})/(1+{T4}) - y" ]
            a_options = [ f"y - {S2} + {S3}", f"(y-{S2}+{S5})/(1+{T5}) - x", f"(y-{S2}+{S6})/(1+{T6}) - y", f"(y-{S2}+{S7})/(1+{T7}) - x - y" ]
            for bo, ao in [(bo,ao) for bo in b_options for ao in a_options]:
                replace_dict = {"SPR_var": f"{S2}", 
                    "B_var": bo, 
                    "A_var": ao,
                    "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                    "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
                math_query = ''' ("ALPHA [2,%d] COMPUTATION")
                    SPR = SPR_var;
                    B = B_var;
                    A = A_var;
                    mu = mu_var;  sig = sig_var;
                    h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                    mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                    f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                    intres = Integrate[f, {y, SPR, intlimit}, {x, -intlimit, B}, {z, -intlimit, A}];
                    Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [2,%d]")
                    ''' % tuple([t]*2)
                try:
                    out_options += [float(self.RunMathematica(math_query, replace_dict, j, t))]
                except Exception as e:
                    print("Integral output ignored due to: " + str(e))
                    self.process.kill()
            return min(out_options)
        elif j == 3:
            b_options = [ f"z - {S3} + {S1}", f"(z-{S3}+{S5})/(1+{T5}) - z" ]
            a_options = [ f"z - {S3} + {S2}",  f"(z-{S3}+{S4})/(1+{T4}) - x", f"(z-{S3}+{S6})/(1+{T6}) - z", f"(z-{S3}+{S7})/(1+{T7}) - x - z" ]
            for bo, ao in [(bo,ao) for bo in b_options for ao in a_options]:
                replace_dict = {"SPR_var": f"{S3}", 
                    "B_var": bo, 
                    "A_var": ao,
                    "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                    "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
                math_query = ''' ("ALPHA [3,%d] COMPUTATION")
                    SPR = SPR_var;
                    B = B_var;
                    A = A_var;
                    mu = mu_var;  sig = sig_var;
                    h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                    mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                    f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                    intres = Integrate[f, {z, SPR, intlimit}, {x, -intlimit, B}, {y, -intlimit, A}];
                    Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [3,%d]")
                    ''' % tuple([t]*2)
                try:
                    out_options += [float(self.RunMathematica(math_query, replace_dict, j, t))]
                except Exception as e:
                    print("Integral output ignored due to: " + str(e))
                    self.process.kill()
            return min(out_options)
        elif j == 4:
            b_options = [ f"({S4})/(1+{T4}) - x", f"(x-{S1}+{S4})/(1+{T4}) - x" ]
            a_options = [ f"(x+y)*(1+{T4}) - {S4} + {S3}", f"((x+y)*(1+{T4})-{S4}+{S5})/(1+{T5}) - x", f"((x+y)*(1+{T4})-{S4}+{S6})/(1+{T6}) - y", f"((x+y)*(1+{T4})-{S4}+{S7})/(1+{T7}) - x - y" ]
            for bo, ao in [(bo,ao) for bo in b_options for ao in a_options]:
                replace_dict = {"SPR_var": f"(y-{S2}+{S4})/(1+{T4}) - y", 
                    "B_var": bo, 
                    "A_var": ao,
                    "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                    "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
                math_query = ''' ("ALPHA [4,%d] COMPUTATION")
                    SPR = SPR_var;
                    B = B_var;
                    A = A_var;
                    mu = mu_var;  sig = sig_var;
                    h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                    mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                    f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                    intres = Integrate[f, {x, SPR, intlimit}, {y, B, intlimit}, {z, -intlimit, A}];
                    Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [4,%d]")
                    ''' % tuple([t]*2)
                try:
                    out_options += [float(self.RunMathematica(math_query, replace_dict, j, t))]
                except Exception as e:
                    print("Integral output ignored due to: " + str(e))
                    self.process.kill()
            return min(out_options)
        elif j == 5:
            b_options = [ f"({S5})/(1+{T5}) - x", f"(x-{S1}+{S5})/(1+{T5}) - x" ]
            a_options = [ f"(x+z)*(1+{T5}) - {S5} + {S2}", f"((x+z)*(1+{T5})-{S5}+{S4})/(1+{T4}) - x ", f"((x+z)*(1+{T5})-{S5}+{S6})/(1+{T6}) - z", f"((x+z)*(1+{T5})-{S5}+{S7})/(1+{T7}) - x - z" ]
            for bo, ao in [(bo,ao) for bo in b_options for ao in a_options]:
                replace_dict = {"SPR_var": f"(z-{S3}+{S5})/(1+{T5}) - z", 
                    "B_var": bo, 
                    "A_var": ao,
                    "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                    "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
                math_query = ''' ("ALPHA [5,%d] COMPUTATION")
                    SPR = SPR_var;
                    B = B_var;
                    A = A_var;
                    mu = mu_var;  sig = sig_var;
                    h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                    mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                    f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                    intres = Integrate[f, {x, SPR, intlimit}, {z, B, intlimit}, {y, -intlimit, A}];
                    Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [5,%d]")
                    ''' % tuple([t]*2)
                try:
                    out_options += [float(self.RunMathematica(math_query, replace_dict, j, t))]
                except Exception as e:
                    print("Integral output ignored due to: " + str(e))
                    self.process.kill()
            return max(out_options)
        elif j == 6:
            b_options = [ f"({S6})/(1+{T6}) - y", f"(y-{S2}+{S6})/(1+{T6}) - y" ]
            a_options = [ f"(y+z)*(1+{T6}) - {S6} + {S1}", f"((y+z)*(1+{T6})-{S6}+{S4})/(1+{T4}) - y", f"((y+z)*(1+{T6})-{S6}+{S5})/(1+{T5}) - z", f"((y+z)*(1+{T6})-{S6}+{S7})/(1+{T7}) - y - z" ]
            for bo, ao in [(bo,ao) for bo in b_options for ao in a_options]:
                replace_dict = {"SPR_var": f"(z-{S3}+{S6})/(1+{T6}) - z", 
                    "B_var": bo, 
                    "A_var": ao,
                    "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                    "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
                math_query = ''' ("ALPHA [6,%d] COMPUTATION")
                    SPR = SPR_var;
                    B = B_var;
                    A = A_var;
                    mu = mu_var;  sig = sig_var;
                    h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                    mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                    f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                    intres = Integrate[f, {y, SPR, intlimit}, {z, B, intlimit}, {x, -intlimit, A}];
                    Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [6,%d]")
                    ''' % tuple([t]*2)
                try:
                    out_options += [float(self.RunMathematica(math_query, replace_dict, j, t))]
                except Exception as e:
                    print("Integral output ignored due to: " + str(e))
                    self.process.kill()
            return min(out_options)
        elif j == 7:
            return 1 - sum([self.temp_alphas[(other_j,t)] for other_j in [0]+self.idx["j"] if other_j != 7])
            # INTEGRALS DISABLED
            b_options = [ f"(z-{S3}+{S7})/(1+{T7}) - x - z", f"((x+z)*(1+{T5})-{S5}+{S7})/(1+{T7}) - x - z" ]
            a_options = [ f"({S7})/(1+{T7}) - x - y", f"(x-{S1}+{S7})/(1+{T7}) - x - y", f"(y-{S2}+{S7})/(1+{T7}) - x - y", f"((x+y)*(1+{T4})-{S4}+{S7})/(1+{T7}) - x - y" ]
            for bo, ao in [(bo,ao) for bo in b_options for ao in a_options]:
                replace_dict = {"SPR_var": f"((y+z)*(1+{T6})-{S6}+{S7})/(1+{T7}) - y - z", 
                    "B_var": bo, 
                    "A_var": ao,
                    "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                    "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
                math_query = ''' ("ALPHA [7,%d] COMPUTATION")
                    SPR = SPR_var;
                    B = B_var;
                    A = A_var;
                    mu = mu_var;  sig = sig_var;
                    h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                    mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                    f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                    intres = Integrate[f, {x, SPR, intlimit}, {y, B, intlimit}, {z, A, intlimit}];
                    Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [7,%d]")
                    ''' % tuple([t]*2)
                try:
                    out_options += [float(self.RunMathematica(math_query, replace_dict, j, t))]
                except Exception as e:
                    print("Integral output ignored due to: " + str(e))
                    self.process.kill()
            return max(out_options)
        else: raise Exception("ERROR: This program only works for i=[1..3], j=[1..7]")


    def apply_alpha_contingency(self, alphas):
        special_case_ks = [5, 6, 7]
        are_non0 = [(bool(self.model.Contingency_k[k]), k) for k in self.idx["k"]]
        case_number = sum([is_non0[0] for is_non0 in are_non0])
        print(">>Contingency Case: {}".format(case_number))
        update_ks = [is_non0[1] for is_non0 in are_non0 if is_non0[0]]
        if case_number == 0:  # Case when all thetas are 0
            pass # Update nothing
        elif case_number == 1:  # Case when exactly 1 theta is 0
            for t in self.idx["t"]:
                alphas[(update_ks[0],t)] = 1 - sum([alphas[(j,t)] for j in [0]+self.idx["j"] if j != update_ks[0]])
        elif case_numer > 1: # Case when more 1 thetas are 0
            for t in self.idx["t"]:
                for update_k in update_ks:
                    alphas[(update_k,t)] = (1 - sum([alphas[(j,t)] for j in [0]+self.idx["j"] if j not in update_ks])) * self.model.Contingency_k[update_k] / sum(self.model.Contingency_k[k_non0] for k_non0 in self.idx["k"] if k_non0 in update_ks)

        do_special_case = sum([bool(self.model.Contingency_k[k_special]) for k_special in special_case_ks])
        if do_special_case == 0:
            pass # 5,6,7 are all non zeroes
        else:
            if case_number == 1:
                for t in self.idx["t"]:
                    alphas[(7,t)] = 1 - sum([alphas[(j,t)] for j in [0]+self.idx["j"] if j != update_ks[0]])
            elif case_numer > 1: # Case when more 1 thetas are 0
                for t in self.idx["t"]:
                    for update_k in update_ks:
                        alphas[(7,t)] = (1 - sum([alphas[(j,t)] for j in [0]+self.idx["j"] if j not in update_ks])) * self.model.Contingency_k[update_k] / sum(self.model.Contingency_k[k_non0] for k_non0 in self.idx["k"] if k_non0 in update_ks)


    def validate_alphas(self, alphas, threshold=0.1):
        for t in self.idx["t"]:
            alpha_sum = sum([alphas[(j,t)] for j in [0]+self.idx["j"]])
            print(">>Sum(alphas t:{}) = {}".format(t, alpha_sum))
            if abs(1 - alpha_sum) > threshold: print(">> >> WARNING: Sum is not within [1 +/- threshold={}]".format(threshold))
        for t in self.idx["t"]: alphas.pop((0,t))  # remove alpha0 since it is not used by the self.model
        return alphas


    def alpha_integrals(self, j, t):
        raise Exception("Doesn't work for some SPR4 values")
        S1, S2, S3, S4, S5, S6, S7 = [self.model.SPR_jt[j,t] for j in self.idx["j"]]
        #S1, S2, S3, S4, S5, S6, S7 = ["S{}S".format(i) for i in range(1,8,)] # SYMBOLIC PROBLEM
        T4, T5, T6, T7 = [self.model.Contingency_k[k] for k in self.idx["k"]] 
        if j == 0:
            replace_dict = {
                "SPR_var": f"{S1}", 
                "B_var": f"Min[ {S2}, {S4}/(1+{T4}) - x ]",
                "A_var": f"Min[ {S3}, {S5}/(1+{T5}) - x, {S6}/(1+{T6}) - y, {S7}/(1+{T7}) - x - y ]",
                "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values()) }
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
                "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
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
                "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
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
                "B_var": f"Min[ z - {S3} + {S1}, (z-{S3}+{S5})/(1+{T5}) - z ]", 
                "A_var": f"Min[ z - {S3} + {S2},  (z-{S3}+{S4})/(1+{T4}) - x, (z-{S3}+{S6})/(1+{T6}) - z, (z-{S3}+{S7})/(1+{T7}) - x - z ]",
                "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
            math_query = ''' ("ALPHA [3,%d] COMPUTATION")
                SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
                h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                intres = Integrate[f, {z, SPR, intlimit}, {x, -intlimit, B}, {y, -intlimit, A}];
                Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [3,%d]")
                ''' % tuple([t]*2)
        elif j == 4:
            replace_dict = {"SPR_var": f"(y-{S2}+{S4})/(1+{T4}) - y", 
                "B_var": f"Max[ ({S4})/(1+{T4}) - x, (x-{S1}+{S4})/(1+{T4}) - x ]", 
                "A_var": f"Min[ (x+y)*(1+{T4}) - {S4} + {S3}, ((x+y)*(1+{T4})-{S4}+{S5})/(1+{T5}) - x, ((x+y)*(1+{T4})-{S4}+{S6})/(1+{T6}) - y, ((x+y)*(1+{T4})-{S4}+{S7})/(1+{T7}) - x - y ]",
                "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
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
                "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
            math_query = ''' ("ALPHA [5,%d] COMPUTATION")
                SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
                h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                intres = Integrate[f, {x, SPR, intlimit}, {z, B, intlimit}, {y, -intlimit, A}];
                Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [5,%d]")
                ''' % tuple([t]*2)
        elif j == 6:
            replace_dict = {"SPR_var": f"(z-{S3}+{S6})/(1+{T6}) - z", 
                "B_var": f"Max[ ({S6})/(1+{T6}) - y, (y-{S2}+{S6})/(1+{T6}) - y ]", 
                "A_var": f"Min[ (y+z)*(1+{T6}) - {S6} + {S1}, ((y+z)*(1+{T6})-{S6}+{S4})/(1+{T4}) - y, ((y+z)*(1+{T6})-{S6}+{S5})/(1+{T5}) - z, ((y+z)*(1+{T6})-{S6}+{S7})/(1+{T7}) - y - z ]",
                "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
            math_query = ''' ("ALPHA [6,%d] COMPUTATION")
                SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
                h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                intres = Integrate[f, {y, SPR, intlimit}, {z, B, intlimit}, {x, -intlimit, A}];
                Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [6,%d]")
                ''' % tuple([t]*2)
        elif j == 7:
            replace_dict = {"SPR_var": f"((y+z)*(1+{T6})-{S6}+{S7})/(1+{T7}) - y - z", 
                "B_var": f"Max[ (z-{S3}+{S7})/(1+{T7}) - x - z, ((x+z)*(1+{T5})-{S5}+{S7})/(1+{T7}) - x - z ]", 
                "A_var": f"Max[ ({S7})/(1+{T7}) - x - y, (x-{S1}+{S7})/(1+{T7}) - x - y, (y-{S2}+{S7})/(1+{T7}) - x - y, ((x+y)*(1+{T4})-{S4}+{S7})/(1+{T7}) - x - y ]",
                "mu_var": "{{ {{ {}, {}, {} }} }}".format(*self.model.RPMean_i.values()),
                "sig_var": "{{ {{ {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }}".format(*self.model.Covariance_iI.values())}
            math_query = ''' ("ALPHA [7,%d] COMPUTATION")
                SPR = SPR_var;  B = B_var;  A = A_var;  mu = mu_var;  sig = sig_var;
                h = 3;  r = { {x, y, z} };  intlimit = Infinity;
                mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
                f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
                intres = Integrate[f, {x, SPR, intlimit}, {y, B, intlimit}, {z, A, intlimit}];
                Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [7,%d]")
                ''' % tuple([t]*2)
        else: raise Exception("ERROR: This program only works for i=[1..3], j=[1..7]")
        return self.RunMathematica(math_query, replace_dict, j, t)