import openpyxl
import os
#import code; code.interact(local=locals())

# to interactively run this file, execute the following on IDLE
# exec(open(r'C:\Users\pu\Pictures\MINLP-Solve\minlp\xlsx_reader.py').read(), globals())

current_dir = os.path.split(os.path.abspath(__file__))[0]
xlsx_path = current_dir + r"\Parameters.xlsx"
book = openpyxl.load_workbook(xlsx_path, data_only=True)
info_delimiter = "&"
dims_delimiter = "_"
assign_delimiter = "="
P = {} # Parameters

prev_row_head = None
for w in book.worksheets[0:1]:
    P_key, key_info, dims = None, [], []
    for row in w.values:
        if row[0] is None:
            P_key, key_info, dims = None, [], [] #maybe useless
            continue
        if prev_row_head is None:
            P_key, *key_info = row[0].split(info_delimiter)
            dims = list(P_key.split(dims_delimiter[1]))
        elif:
        	# unwrap values
        prev_row_head = row[0]


import code; code.interact(local=locals())