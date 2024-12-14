
    if rowsfunc1 != colsfunc1:
        print("Matrix must be square (rows == cols) to solve a system of linear equations.")
        return  # Kembali jika matriks bukan persegi

    matrix = []
    y = []

    print(f"Enter the elements for a {rowsfunc1}x{colsfunc1} matrix:")
    for i in range(rowsfunc1):
        while True:
            try: