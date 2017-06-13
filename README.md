# MINLP-Solve
0. System Requirements: Windows OS 64-bit, Python 3.6.1
1. Download dependencies.zip from : <DRIVE URL>
2. Extract dependencies.zip in root directory (Place the zip file in this folder and select the Extract here option)
3. In ./installer/ install python (check add python to PATH on installation) and then run: python ./installer/install_dependencies.py
4. Run solver: python ./minlp/solver.py


###  SNIPPETS ###
model.testVar = Var(tuple(itertools.product(range(2), range(3))), domain=NonNegativeIntegers)
