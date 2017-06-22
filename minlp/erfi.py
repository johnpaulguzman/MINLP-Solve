import sympy as sym

h = 3 # if len(idx["i"]) != 3: raise Exception("This program only works for i=3")
x, y, z = sym.symbols('x y z', real=True)
a3, a4, SPR1 = sym.symbols('a3, a4, SPR1', real=True, positive=True)

r = sym.Matrix([ [x, y, z] ])
mu = sym.Matrix([ [model.RPMean_i[1], model.RPMean_i[2], model.RPMean_i[3]] ])
sig = sym.Matrix([ [model.Covariance_iI[1,1], model.Covariance_iI[1,2], model.Covariance_iI[1,3]],
                   [model.Covariance_iI[1,1], model.Covariance_iI[1,2], model.Covariance_iI[1,3]], 
                   [model.Covariance_iI[1,1], model.Covariance_iI[1,2], model.Covariance_iI[1,3]] ])
f = sym.exp(1/2 * ((r-mu) * sig.inv() * (r-mu).transpose())[0]) / sym.sqrt((2*sym.pi)**h * sig.det())

f2 = sym.Rational(63493635934241,1000000000000000)*sym.exp(3893/8)*sym.exp(-15*x)*sym.exp(x**2/2)*sym.exp(-17*y/4)*sym.exp(y**2/8)*sym.exp(-52*z)*sym.exp(2*z**2)

gx = sym.exp(-15*x)*sym.exp(x**2/2)
gy = sym.exp(-17*y/4)*sym.exp(y**2/8)
gz = sym.exp(-52*z)*sym.exp(2*z**2)

Gx = sym.integrate(sym.powsimp(gx), (x,SPR1, 100))
Gy = sym.integrate(sym.powsimp(gy), (y,-100, a4))
Gz = sym.integrate(sym.powsimp(gz), (z,-100, a3))

sym.pprint(Gx)
sym.pprint(Gy)
sym.pprint(Gz)

F2 = sym.Rational(63493635934241,1000000000000000)*sym.exp(sym.Rational(3893,8))*Gx*Gy*Gz

sym.pprint(F2)
sym.pprint(sym.simplify(F2))