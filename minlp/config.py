import os
current_directory = os.path.split(os.path.abspath(__file__))[0]

# Mathematica Variables
math_exe =  "C:\\Program Files\\Wolfram Research\\Mathematica\\11.0\\math.exe"
math_script = "{}\\math_script.m".format(current_directory) #add script1234etc

# Solver Variables
input_path = "{}\\Parameters.xlsx".format(current_directory)
solver_name = "bonmin"
solver_path = "{}\\..\\solvers\\CoinAll-1.6.0-win64-intel11.1\\bin\\bonmin.exe".format(current_directory)
solver_log = "{}\\solver.log".format(current_directory)
solver_options = {
    "halt_on_ampl_error" : "yes",
    "wantsol" : 1,
    "output_file" : "{}\\output.txt".format(current_directory),
    "max_iter" : 6000,
}

# XLSX READER Variables
info_delimiter = "&"
dims_delimiter = "_"
assign_delimiter = "="
range_infix = "-"
index_keyword = "INDEX_METADATA"
decimal_precision = 16  # max value