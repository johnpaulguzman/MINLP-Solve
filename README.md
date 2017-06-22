# MINLP-Solve
0. System Requirements: Windows OS 64-bit, Python 3.6.1
1. Extract dependencies.zip.001 into dependencies folder at root level
2. In ./installer/ install python (check add python to PATH on installation)
3. Run: python ./installer/install_dependencies.py
4. Run solver: python ./minlp/solver.py


###  SNIPPETS ###
import mpmath as mp
import sympy as sp
h = sp.symbols("h")
r = sp.Matrix([[1,2,3]])
mu = sp.Matrix([[4,5,6]])
sig = sp.Matrix([[7,8,9],[10,-20,12],[13,14,15]])
f = sp.exp(1/2 * ((r-mu) * sig.inv() * (r-mu).transpose())[0]) / sp.sqrt((2*mp.pi)**h * sig.det())

x,y,z,A,B,C = sp.symbols("x y z A B C")
replacements = {"A": "SPR1", "B": "SPR2", "C": "SPR3"}
f = (x*y*z)
i = integrate(f, (x, -100000,A-y-z), (y, -100000,B-z), (z, C,100000))
istr = str(i)
for key, value in replacements.items(): istr = istr.replace(key, value)

## RESOURCES ##
SQRT 0 => https://groups.google.com/forum/#!topic/ampl/zv7z4HA0Qek

https://github.com/Pyomo/pyomo/blob/master/doc/GettingStarted/current/dae.txt

https://static1.squarespace.com/static/5492d7f4e4b00040889988bd/t/57bd0f93d482e927298cc9da/1472008085561/3_PyomoFundamentals.pdf

http://www.springer.com/cda/content/document/cda_downloaddocument/9781461432258-c1.pdf?SGWID=0-0-45-1300451-p174287165

https://software.sandia.gov/downloads/pub/pyomo/PyomoOnlineDocs.html#_parameters

https://projects.coin-or.org/Bonmin/browser/stable/1.5/Bonmin/doc/BONMIN_UsersManual.pdf?format=raw