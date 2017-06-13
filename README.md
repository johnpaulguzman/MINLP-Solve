# MINLP-Solve
0. System Requirements: Windows OS 64-bit, Python 3.6.1
1. Download dependencies.zip from : <DRIVE URL>
2. Extract dependencies.zip in root directory (Place the zip file in this folder and select the Extract here option)
3. In ./installer/ install python (check add python to PATH on installation) and then run: python ./installer/install_dependencies.py
4. Run solver: python ./minlp/solver.py


###  NOTES  ###

http://www.midaco-solver.com/index.php/more/faq
Constraints of the form LOWER < G(X) < UPPER can be transformed into two individual constraints:
  G(X) >  LOWER
  -G(X) > -UPPER
Inequality constraints G(X) >= 0, or flip signs to get <= 0
  g[1] = x[1] - 1.333333333  

A maximization problem can easily be transformed into a minimization problem by
switching the sign of the objective function: F(X) = - F(X).