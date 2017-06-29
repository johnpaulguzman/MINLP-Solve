import os
from enum import Enum
current_directory = os.path.split(os.path.abspath(__file__))[0]

class AlphaOptions(Enum):
	calculate_first_t = 1
	calculate_all_t = 2

# Mathematica Variables
alpha_generator_option = AlphaOptions.calculate_first_t
halt_after_alpha_generate = True
math_exe =  "C:\\Program Files\\Wolfram Research\\Mathematica\\11.0\\math.exe"
math_script_dir = "{}\\math_scripts".format(current_directory)
mathematica_timeout = 30

# Solver Variables
input_path = "{}\\Parameters.xlsx".format(current_directory)
solver_name = "bonmin"
solver_path = "{}\\..\\solvers\\CoinAll-1.6.0-win64-intel11.1\\bin\\bonmin.exe".format(current_directory)
solver_log = "{}\\solver.log".format(current_directory)
solver_options = {
    "halt_on_ampl_error" : "yes",
    "wantsol" : 1,
    "output_file" : "{}\\output.txt".format(current_directory),
    "max_iter" : 10000,
#    "bonmin.time_limit" : 60*20,
#    "bonmin.node_limit" : 100,
}

# XLSX READER Variables
info_delimiter = "&"
dims_delimiter = "_"
assign_delimiter = "="
range_infix = "-"
index_keyword = "INDEX_METADATA"
derived_data_sheet = "Derived Data"
