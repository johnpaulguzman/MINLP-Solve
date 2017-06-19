from pyomo.environ import *
from pyomo.dae import *
################################################
## SAVE
# model.l = ContinuousSet(bounds=(-10,10))
# model.y = Var(model.t,model.l)
# model.dydt = DerivativeVar(model.y, wrt=model.t)
# model.dydl2 = DerivativeVar(model.y, wrt=(model.l,model.l))
# discretizer = TransformationFactory('dae.collocation')
# discretizer.apply_to(model,nfe=20,ncp=6,scheme='LAGRANGE-RADAU')
################################################

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
    row_print(entries, padding=64, line_size=2)


model = ConcreteModel()
model.s = Set(initialize=['a','b'])
model.t = ContinuousSet(bounds=(0,5))
model.t1 = ContinuousSet(bounds=(0,5))
model.t2 = ContinuousSet(bounds=(0,5))
model.x = Var(model.t1,model.t2,model.s)
model.inz = Var(domain=NonNegativeIntegers)

def testy(model): 
    return model.inz<=20
model.testy = Constraint(rule=testy)

def _intX1(m,i,j,s):
    return (m.x[i,j,s]-m.inz)**2
model.intX1 = Integral(model.t1,model.t2,model.s,wrt=model.t1,rule=_intX1)

def _intX2(m,j,s):
    return (m.intX1[j,s]-m.inz)**2
model.intX2 = Integral(model.t2,model.s,wrt=model.t2,rule=_intX2)

def _obj(m):
    return sum(model.intX2[k] for k in m.s)
model.obj = Objective(rule=_obj, sense=maximize)

opt = SolverFactory("bonmin")
results = opt.solve(model, tee=True)

print("Printing values for all variables")
print_vars(model)
