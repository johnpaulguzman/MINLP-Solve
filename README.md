# MINLP-Solve

1. Download extract.zip from : https://drive.google.com/open?id=0BzJSPZqOmsTbV051ZVFBb3FXOGc
2. Extract extract.zip in root directory (Place the zip file in this folder and select the Extract here option)
3. In ./installer/ install python (check add python to PATH on installation) and then run: python ./installer/install_dependencies.py
4. Run solver: python ./minlp/example_MINLPc.py


###  NOTES  ###

http://www.midaco-solver.com/index.php/more/faq
Constraints of the form LOWER < G(X) < UPPER can be transformed into two individual constraints:
  G(X) >  LOWER
  -G(X) > -UPPER
Inequality constraints G(X) >= 0, or flip signs to get <= 0
  g[1] = x[1] - 1.333333333  

A maximization problem can easily be transformed into a minimization problem by
switching the sign of the objective function: F(X) = - F(X).