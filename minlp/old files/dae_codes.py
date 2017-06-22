from pyomo.environ import *
from pyomo.dae import *
################################################
## SAVE
#model.t = ContinuousSet(bounds=(0,5))
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
model.t1 = ContinuousSet(bounds=(0,5))
model.t2 = ContinuousSet(bounds=(0,5))
model.x = Var(model.t1,model.t2)
model.var2 = Var(domain=NonNegativeIntegers)

def testy(model): 
    return model.var2<=20
model.testy = Constraint(rule=testy)

def _intX1(m,i,j):
    return (m.x[i,j]-m.var2-1)**3
model.intX1 = Integral(model.t1,model.t2,wrt=model.t1,rule=_intX1)

def _intX2(model,j, wat):
    print(wat)
    return model.intX1[j]
model.intX2 = Integral(model.t2,wat="wat",wrt=model.t2,rule=_intX2)

def _obj(m):
    return model.intX2
model.obj = Objective(rule=_obj, sense=maximize)

opt = SolverFactory("bonmin")
results = opt.solve(model, tee=True)

print("Printing values for all variables")
print_vars(model)



# Ampl Car Example (Extended)
#
# Shows how to convert a minimize final time optimal control problem
# to a format pyomo.dae can handle by removing the time scaling from
# the ContinuousSet.
#
# min int[0,T](u), u>0
# dt/dx = 1/v
# dv/dx = a/v
# da/dx = j/v
# a = u1 - u2 - R*v^2
# x(0)=0; x(T)=L
# v(0)=0; v(T)=0
# -3 <= u <= 1 (engine constraint)
# 0 <= p <= 1 (positive part of u)
# p >= u (get positive part of u constraint)
#
#       / 1,     v <= 7m/s
# u <= {                    (electric car constraint)
#       \ 1*7/v, v > 7m/s
#
# -1.5 <= dv/dx <= 0.8 (comfort constraint -> smooth driving)
# -0.5 <= da/dx <= 0.5 (comfort constraint -> jerk)
# v <= Vmax (40 kmh[0-250m) + 100 kmh[250-500m) + 75 kmh[500-750m) + 25 kmh[750-1000m))

from pyomo.environ import *
from pyomo.dae import *

m = ConcreteModel()
    
m.L = Param(initialize=1000.0) # Final position
m.R = Param(initialize=0.001)  # Friction factor
m.NFE = Param(initialize=50)  # Number of finite elements
m.NCP = Param(initialize=3)  # Number of finite elements
m.T = Param(initialize=200.0)   # Estimated time
zero = 1e-5

def _boundsV(m, i):
    if i<=250/value(m.L):
        return (zero,30/3.6)
    if i<=500/value(m.L):
        return (zero,40/3.6)
    if i<=750/value(m.L):
        return (zero,25/3.6)
    else:
        return (zero,15/3.6)


def _initV(m, i):
    return value(m.L)/value(m.T)


def _initT(m, i):
    return i*value(m.T)


m.chi = ContinuousSet(bounds=(0,1)) # Unscaled distance
m.x = Var(m.chi, bounds=(0,m.L))    # Scaled distance
m.t = Var(m.chi, bounds=(0,None), initialize=_initT)
m.v = Var(m.chi, bounds=_boundsV, initialize=_initV)
m.a = Var(m.chi, bounds=(-1.5,0.8), initialize=0)
m.j = Var(m.chi, bounds=(-0.5, 0.5), initialize=0)
m.u = Var(m.chi, bounds=(-3,1), initialize=0)
m.p = Var(m.chi, bounds=(0,1), initialize=0)

m.dt = DerivativeVar(m.t)
m.dx = DerivativeVar(m.x)
m.dv = DerivativeVar(m.v)
m.da = DerivativeVar(m.a)

def _ode1(m, i):
    if i==0:
        return Constraint.Skip
    return m.dx[i] == m.L


m.ode1 = Constraint(m.chi, rule=_ode1)

def _ode2(m, i):
    if i==0:
        return Constraint.Skip
    return m.dt[i] == m.L / m.v[i]


m.ode2 = Constraint(m.chi, rule=_ode2)

def _ode3(m, i):
    if i==0:
        return Constraint.Skip
    return m.dv[i] == m.L * m.a[i] / m.v[i]


m.ode3 = Constraint(m.chi, rule=_ode3)

def _ode4(m, i):
    if i==0:
        return Constraint.Skip
    return m.da[i] == m.L * m.j[i] / m.v[i]


m.ode4 = Constraint(m.chi, rule=_ode4)


def _alg1(m, i):
    if i==0:
        return Constraint.Skip
    return m.a[i] == m.u[i] - m.R*m.v[i]**2


m.alg1 = Constraint(m.chi, rule=_alg1)


def _electric(m, i):
    if i==0:
        return Constraint.Skip
    return m.u[i] <= 7 / m.v[i]


m.electric = Constraint(m.chi, rule=_electric)


def _power(m, i):
    if i==0:
        return Constraint.Skip
    return m.p[i] >= m.u[i]


m.power = Constraint(m.chi, rule=_power)


def _initial(m):
    yield m.t[0] == 0
    yield m.x[0] == 0
    yield m.v[0] == zero
    yield m.t[1] == m.T
    yield m.x[1] == m.L
    yield m.v[1] == 1


m.initial = ConstraintList(rule=_initial)


#discretizer = TransformationFactory('dae.finite_difference')
#discretizer.apply_to(m, nfe=value(m.NFE), wrt=m.chi, scheme='BACKWARD')
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=value(m.NFE), ncp=value(m.NCP), wrt=m.chi, scheme='LAGRANGE-RADAU')


def _consumption(m):
    last = 0
    consumption = 0
    for i in m.chi:
        consumption += (i - last)*m.p[i]
        last = i
    return consumption


#m.obj = Objective(expr=m.t[1])
m.obj = Objective(rule=_consumption)

solver = SolverFactory('ipopt')
solver.solve(m,tee=True)

print("final time = %6.2f" %(value(m.t[1])))

x = []
t = []
v = []
a = []
u = []

for i in m.chi:
    x.append(value(m.x[i]))
    t.append(value(m.t[i]))
    v.append(3.6*value(m.v[i]))
    a.append(10*value(m.a[i]))
    u.append(10*value(m.u[i]))


import matplotlib.pyplot as plt

plt.plot(x, v, label='v (km/h)')
plt.plot(x, a, label='a (x10 m/s2)')
plt.plot(x, u, label='u (x10 m/s2)')
plt.xlabel('distance')
plt.grid()
plt.legend()

plt.show()