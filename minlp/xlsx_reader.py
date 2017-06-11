import openpyxl

# exec(open(r'C:\Users\pu\Pictures\MINLP-Solve\minlp\xlsx_reader.py').read(), globals())

xlsx_path = r'C:\Users\pu\Pictures\MINLP-Solve\minlp\Parameters.xlsx'
book = openpyxl.load_workbook(xlsx_path, data_only=True)

prev_row_head = None
for w in book.worksheets[0:1]:
    for row in w.values:
        if prev_row_head is None: print("Prev is None")
        print("Current is ", row)
        prev_row_head = row[0]
