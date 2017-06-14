import openpyxl
import os
irange = lambda start, end: list(range(int(start), int(end)+1))
#import code; code.interact(local=locals())

# to interactively run this file, execute the following on IDLE
# exec(open(r'C:\Users\pu\Pictures\MINLP-Solve\minlp\xlsx_reader.py').read(), globals())

current_dir = os.path.split(os.path.abspath(__file__))[0]
xlsx_path = current_dir + r"\Parameters.xlsx"
book = openpyxl.load_workbook(xlsx_path, data_only=True)
info_delimiter = "&"
dims_delimiter = "_"
assign_delimiter = "="
range_infix = "-"
index_keyword = "INDEX_METADATA"
P = {} # Parameters
idx = {} # Indices

def extract_data_chunks(book):
    data, data_chunks, data_dump = [], [], []
    for w in book.worksheets:
        for row in w.values:
            data += [row]
        data += [(None,)]
    for d in data:
        if d[0] is None: # note
            if data_dump: # note
                data_chunks += [data_dump]
                data_dump = []
        else: data_dump += [d] # note
    return data_chunks

def init_idx(data_chunk):
    index_data = {}
    for index, index_range, *_ in data_chunk[1:]:
        lb, ub, *_ = index_range.split(range_infix)
        index_data[index] = irange(lb, ub)
    return index_data

data_chunks = extract_data_chunks(book)

for data_chunk in data_chunks:
    for row in data_chunk:
        if row[0] == index_keyword:
            idx = init_idx(data_chunk)
            break
        #elif:

import code; code.interact(local=locals()); exit()

prev_row_head = None
for w in book.worksheets[0:1]:
    P_key, key_info, dims = None, [], []
    for row in w.values:
        print(row)
        continue
        if row[0] is None: P_key, key_info, dims = None, [], []
        elif row[0] == index_keyword: pass # store indices
        elif prev_row_head is None:
            P_key, *key_info = row[0].split(info_delimiter)
            dims = list(P_key.split(dims_delimiter[1]))
        else: pass # unwrap values
        prev_row_head = row[0]

