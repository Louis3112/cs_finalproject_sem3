import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.context('fivethirtyeight')
import math

rows = np.random.randint(1,9)
colls = np.random.randint(1,9)
#tes
def clear():
  if os.name == 'nt':
    os.system('cls')
  else:
    os.system('clear')

def func1() : 
    rowsfunc1, colsfunc1 = map(int, input("Enter size of matrix (rows cols): ").split())

    if rowsfunc1 < 2 or colsfunc1 < 2:
        print("Matrix size must be at least 2x2.")
        return  # Kembali jika ukuran tidak valid
    
    if rowsfunc1 != colsfunc1:
        print("Matrix must be square (rows == cols) to solve a system of linear equations.")
        return  # Kembali jika matriks bukan persegi

    matrix = []
    y = []

    print(f"Enter the elements for a {rowsfunc1}x{colsfunc1} matrix:")
    for i in range(rowsfunc1):
        while True:
            try:
                row = list(map(int, input(f"Enter row #{i+1}: ").split()))
                if len(row) != colls:
                    raise ValueError(f"Row must have exactly {colsfunc1} elements.")
                matrix.append(row)
                break
            except ValueError as e:
                print(e)
    
    print("Enter elements for the vector (y):")
    for i in range(colsfunc1):
        while True:
            try:
                el = int(input(f"Enter element #{i+1}: "))
                y.append(el)
                break
            except ValueError:
                print("Please enter a valid integer.")

    A = np.array(matrix)
    y = np.array(y)

    # Periksa apakah determinan matriks adalah nol
    if np.linalg.det(A) == 0:
        print("The matrix is singular and cannot be solved.")
        return

    # Menyelesaikan sistem persamaan linear
    x = np.linalg.solve(A, y)
    print("\nSolution (x):")
    print(x)

def func2() :
    while True:
        print()
        print("\nMatrix Properties")
        print("1. Types of matrices")
        print("2. Behavior of matrix")
        print("3. Exit")
        try:
            choice = int(input("> "))
        except ValueError:
            print("\nInvalid input, choose from 1-2!")
            enter()
            continue
        
        if choice == 1:
            func2_1()
        elif choice == 2:
            func2_2()
        elif choice == 3:
            break
        else:
            print("\nInvalid input, choose from 1-2!")
            enter()
            continue
        
def func2_1():
    while True:    
        print("\nThere are a lot of types of matrices")
        print("1. Zero Matrix")
        print("2. Square Matrix")
        print("3. Rectangular Matrix")
        print("4. Diagonal Matrix")
        print("5. Identity Matrix")
        print("6. Scalar Matrix")
        print("7. Row Matrix")
        print("8. Column Matrix")
        print("7. Exit")
        print("Choose one to see the example")

        try:
            choice = int(input("> "))
        except ValueError:
            print("\nInvalid input, choose from 1-2!")
            enter()
            continue
        
        if choice == 1:
            print("\nZero Matrix or Null Matrix is a matrix with all its elements")
            print("having a VALUE of ZERO.")
            
            zeroMatrix = np.zeros((rows,colls))
            print("\n", zeroMatrix)
            enter()

        elif choice == 2:
            print("\nSquare Matrix is a matrix where the lengths")
            print("of row and the columns ARE the SAME.")

            squareMatrix = np.random.randint(9, size=(4,4))
            print("\n", squareMatrix)
            enter()

        elif choice == 3:
            print("\nRectangular Matrix is a matrix where the lengths")
            print("of row and the column are NOT the SAME.")

            rectangularMatrix = np.random.randint(9, size=(rows,colls))
            print("\n", rectangularMatrix)
            enter()

        elif choice == 4:
            print("\nDiagonal Matrix is a square matrix with the elements")
            print("on the MAIN DIAGONAL having real values while the others are ZERO")

            diagonalMatrix = np.diag([np.random.randint(1,9),np.random.randint(1,9),np.random.randint(1,9)])
            print("\n", diagonalMatrix)
            enter()
        
        elif choice == 5:
            print("\nIdentity Matrix is a square matrix with the elements")
            print("on the MAIN DIAGONAL having a value of '1' while the others are ZERO")

            identityMatrix = np.eye(4)
            print("\n", identityMatrix)
            enter()
        
        elif choice == 6:
            print("\nScalar Matrix is a square matrix with the elements")
            print("on the MAIN DIAGONAL having the same values while the others are ZERO")

            scalarMatrix = np.eye(4) * np.random.randint(1,9)
            print("\n", scalarMatrix)
            enter()
        
        elif choice == 7:
            print("\nRow Matrix is a matrix that only consist of one row")

            rowMatrix = np.array([np.random.randint(1,9), np.random.randint(1,9), np.random.randint(1,9)])
            print("\n", rowMatrix)
            enter()
        
        elif choice == 8:
            print("\nColumn Matrix is a matrix that only consist of one column")

            columnMatrix = np.array([[np.random.randint(1,9)], [np.random.randint(1,9)], [np.random.randint(1,9)]])
            print("\n", columnMatrix)
            enter()

        elif choice == 9:
            break

        else:  
            print("\nInvalid choice! Please choose from 1-9.")
            enter()
            continue

def func2_2():
    A = np.array([[np.random.randint(1,9), np.random.randint(1,9)], [np.random.randint(1,9),np.random.randint(1,9)]])
    B = np.array([[np.random.randint(1,9), np.random.randint(1,9)], [np.random.randint(1,9),np.random.randint(1,9)]])
    C = np.array([[np.random.randint(1,9), np.random.randint(1,9)], [np.random.randint(1,9),np.random.randint(1,9)]])
    k = np.random.randint(1,5)
    l = np.random.randint(1,5)

    while True:
        print("\nMatrix has a lot of behavior")
        print("Suppose A,B,C are matrix and k and l are scalar")
        print("1. A + B = B + A\t\tCommutative Addition")
        print("2. A + (B + C) = (A + B) + C\tAssociative Addition")
        print("3. k * (A + B) = kA + kB\tDistributive Multiplication With Scalar")
        print("4. (k+l) A = kA + lA\t\tDistributive Multiplication With Scalar")
        print("5. (kl) A = k(lA)\t\tAssociative Multiplication With Scalar")
        print("6. k(A * B) = kA(B) = A(kB)\tDistributive Multiplication With Scalar")
        print("7. A(BC) = (AB)C\t\tAssociative Multiplication")
        print("8. A(B + C) = AB + AC\t\tDistributive Addition")
        print("9. (A + B)C = AC + BC\t\tDistributive Addition")
        print("10. A * B != B * A\t\tNot Commutative Multiplication")
        print("11. If A * B = A * C, does not mean B = C ")
        print("12. If A * B = 0, it's either A = 0 and B = 0")
        print("                  or A != 0 and B != 0")
        print("13. Exit")
        print("Choose one to see the example")

        try:
            choice = int(input("> "))
        except ValueError:
            print("\nInvalid input, choose from 1-2!")
            enter()
            continue

        if choice == 1:
            print("\n1. A + B = B + A\t\tCommutative Addition")
            print("\nA + B")
            print("\n",A), print("\n+ "), print("\n",B)
            print("\n= "), print("\n",A + B)

            print("\nwhile B + A")
            print("\n",B), print("\n+ "), print("\n",A)
            print("\n= "), print("\n",B + A)

            print("\nTherefore, A + B = B + A is TRUE")
            enter()

        elif choice == 2:
            print("\n2. A + (B + C) = (A + B) + C\tAssociative Addition")
            print("\nA + (B + C)")
            print("\n",A), print("\n+ ( "), print("\n",B), print("\n+ "), print("\n",C), print("\n)")
            print("\n= "), print("\n",A + (B + C))

            print("\nwhile (A + B) + C")
            print("\n( "), print("\n",A), print("\n+ "), print("\n",B), print("\n)"), print("\n+ "),print("\n",C), 
            print("\n= "), print("\n",(A + B) + C)

            print("\nTherefore, A + (B + C) = (A + B) + C is TRUE")
            enter()

        elif choice == 3:
            print("\n3. k * (A + B) = kA + kB\tDistributive Multiplication With Scalar")
            print("\nk * (A + B)")
            print("\n",k," *"), print("\n( "), print("\n",A), print("\n+ "), print("\n",B), print("\n )")
            print("\n= "), print("\n",k * (A + B))

            print("\nwhile kA + kB")
            print("\n",k," *"), print("\n",A), print("\n+ "), print("\n",k," *"), print("\n",B),
            print("\n= "), print("\n",k*A + k*B)

            print("\nTherefore, k * (A + B) = kA + kB is TRUE")
            enter()

        elif choice == 4:
            print("\n4. (k+l) A = kA + lA\t\tDistributive Multiplication With Scalar")
            print("\n(k+l) A")
            print("\n(",k," + ",l,")"), print("\n* "), print("\n",A)
            print("\n= "), print("\n",(k+l) * A)

            print("\nwhile kA + lA")
            print("\n",k," *"), print("\n",A), print("\n+ "), print("\n",l," *"), print("\n",A),
            print("\n= "), print("\n",k*A + l*A)

            print("\nTherefore, (k+l) A = kA + lA is TRUE")
            enter()

        elif choice == 5:
            print("\n5. (kl) A = k(lA)\t\t\tAssociative Multiplication With Scalar")
            print("\n(kl) A")
            print("\n(",k," * ",l,")"), print("\n",A)
            print("\n= "), print("\n",(k * l) * A)

            print("\nwhile k(lA)")
            print("\n",k," *"), print("\n(",l," *"), print("\n",A), print("\n)") 
            print("\n= "), print("\n",k*(l*A))

            print("\nTherefore, (kl) A = k(lA) is TRUE")
            enter()
        
        elif choice == 6:
            print("\n6. k(A * B) = (kA) * B = A * (kB)\tDistributive Multiplication With Scalar")
            print("\nk(A * B)")
            print("\n",k," * "), print("\n("), print("\n",A), print("\n* "), print("\n",B)
            print("\n= "), print("\n",k * np.dot(A, B))

            print("\nwhile (kA) * B")
            print("\n(",k," * "), print("\n",A), print("\n)"), print("\n* "), print("\n",B)
            print("\n= "), print("\n",np.dot(k * A, B))

            print("\nwhile A * (kB)")
            print("\n",A), print("\n*"), print("\n("), print("\n(",k," * "), print("\n",B), print("\n)")
            print("\n= "), print("\n",np.dot(A, k * B))
            print("\nTherefore, k(A * B) = (kA) * B = A * (kB) is TRUE")
            enter()

        elif choice == 7:
            print("\n7. A(BC) = (AB)C\t\tAssociative Multiplication")
            print("\nA(BC)")
            print("\n",A), print("\n*"), print("\n("), print("\n",B), print("\n* "), print("\n",C), print("\n)")
            print("\n= "), print("\n",np.dot(A, np.dot(B, C)))

            print("\nwhile (AB)C")
            print("\n("), print("\n",A), print("\n*"), print("\n",B), print("\n)"), print("\n* "),print("\n",C)
            print("\n= "), print("\n",np.dot(np.dot(A, B), C))
            print("\nTherefore, A(BC) = (AB)C is TRUE")
            enter()

        elif choice == 8:
            print("\n8. A(B + C) = AB + AC\t\tDistributive Addition")
            print("\nA(B+C)")
            print("\n",A), print("\n*"), print("\n("), print("\n",B), print("\n+ "), print("\n",C), print("\n)")
            print("\n= "), print("\n",np.dot(A, B + C))

            print("\nwhile AB + AC")
            print("\n",A), print("\n*"), print("\n",B), print("\n+"), print("\n",A), print("\n*"), print("\n",C)
            print("\n= "), print("\n",np.dot(A, B) + np.dot(A, C))
            print("\nTherefore, A(B+C) = AB + AC is TRUE")
            enter()

        elif choice == 9:
            print("\n9. (A + B)C = AC + BC\t\tDistributive Addition")
            print("\n(A+B)C")
            print("\n("), print("\n",A), print("\n+"), print("\n",B), print("\n)"), print("\n* "), print("\n",C), 
            print("\n= "), print("\n",np.dot(A + B, C))

            print("\nwhile AC + BC")
            print("\n",A), print("\n*"), print("\n",C), print("\n+"), print("\n",B), print("\n*"), print("\n",C)
            print("\n= "), print("\n",np.dot(A, C) + np.dot(B, C))
            print("\nTherefore, (A+B)C = AC + BC is TRUE")
            enter()

        elif choice == 10:
            print("\n10. A * B != B * A\t\tNot Commutative Multiplication")
            print("\nA * B")
            print("\n",A), print("\n*"), print("\n",B)
            print("\n= "), print("\n",np.dot(A, B))

            print("\nwhile B * A")
            print("\n",B), print("\n*"), print("\n",A)
            print("\n= "), print("\n",np.dot(B, A))
            print("\nSince the results are different")
            print("Therefore, A * B != B * A is TRUE")
            enter()

        elif choice == 11:
            print("11. If A * B = A * C, does not mean B = C ")
            AKhusus = np.array([[0, 100], [0, 0]])
            BKhusus = np.array([[-6, 28], [0, 0]])
            CKhusus = np.array([[48, -19], [0, 0]])

            print("\nA * B")
            print("\n",AKhusus), print("\n*"), print("\n",BKhusus)
            print("\n= "), print("\n",np.dot(AKhusus, BKhusus))

            print("\nA * C")
            print("\n",AKhusus), print("\n*"), print("\n",CKhusus)
            print("\n= "), print("\n",np.dot(AKhusus, CKhusus))

            print("\nEven the results are same, but B != C")
            print("Therefore, if A * B = A * C, does not mean B = C is TRUE")
            enter()

        elif choice == 12:
            print("12. If A * B = 0, it's either A = 0 and B = 0")
            print("                  or A != 0 and B != 0")
            AKhusus = np.array([[1, 2], [2, 4]])
            BKhusus = np.array([[6, -4], [-3, 2]])

            print("\nSuppose A = 0 * B = 0")
            print(np.array([[0, 0], [0, 0]])), print("\n*")
            print(np.array([[0, 0], [0, 0]]))

            print("\nBut if A != 0 * B != 0")
            print("\n",AKhusus), print("\n*"), print("\n",BKhusus)
            print("\n= "), print("\n",np.dot(AKhusus, BKhusus))

            print("\nTherefore, if A * B = 0, it's either A = 0 and B = 0")
            print("                  or A != 0 and B != 0")
            enter()
        
        elif choice == 13:
            break

        else:  
            print("\nInvalid choice! Please choose from 1-13.")
            enter()
            continue

def func3() : 
    print("WIP") # Work in progress

def func4() : 
    print("WIP") # Work in progress
    
def func5() : 
    print("WIP") # Work in progress    

class nonlinear(): # func6
    def main_non_linear(self):
        while True:
            clear()
            print("Non Linear Equation Calculator")
            print("1. Quadratic Equation (Bisection Method)")
            print("0. Exit")
            choice = input("> ")
            if choice == '1':
                self.quadratic()
            elif choice == '0':
                return
            else:
                print("Invalid choice! Please choose from 1-3.")
                enter()

    def quadratic(self):
        while True:
            clear()
            print("Non Linear Equation Calculator")
            print("1. Regular Quadratic Equation")
            print("2. Bisection Method")
            choice = int(input("Enter choice: "))
            if choice == 1:
                clear()
                print("Quadratic Equation Calculator")
                a = int(input("Enter coefficient a: "))
                b = int(input("Enter coefficient b: "))
                c = int(input("Enter coefficient c: "))

                # Calculate the discriminant
                d = b ** 2 - 4 * a * c

                if d < 0:
                    print("The equation has no real solutions")
                elif d == 0:
                    x = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                    print(f"The equation has one solution: {x} ")
                else:
                    x1 = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                    x2 = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                    print(f"The equation has two solutions: {x1} or {x2}")

                # Calculate the y value for each of x
                x = np.linspace(-20, 20, 1000)
                y = a * x ** 2 + b * x + c

                # Plot the x, y pairs
                fig, ax = plt.subplots()
                ax.set_title("Quadratic Equations Graphics")
                ax.plot(x, y)

                ax.set_ylim(-100, 100)
                ax.set_xlim(-20, 20)

                ymin, ymax = ax.get_ylim()
                xmin, xmax = ax.get_xlim()

                ax.grid(True, color='gray', linewidth = 0.5)
                ax.hlines(y=0, xmin=xmin, xmax=xmax, colors='r')
                ax.vlines(x=0, ymin=ymin, ymax=ymax, color='black', label='full height')

                # Show the plot
                plt.show()
                
                print("Do you want to recalculate the equation? (y/n)")
                if input("> ") == "y":
                    continue
                else:
                    break
            elif choice == 2:
                clear()
                print("Bisection Method Calculator")
                print("Input Function (e.g x^2 - 4 / x***3 + 2)")
                self.function_input = input("> ")
                
                inter1 = int(input("Enter interval 1: "))
                inter2 = int(input("Enter interval 2: "))
                tol = float(input("Enter tolerance (e.g 0.00001): "))
                bisec = self.bisection1(self.bisec_func, inter1, inter2, tol)
                    
                if bisec is not None:
                    print(f'Approximate root is {bisec}')
                    enter()
                else:
                    print('Invalid Interval')
                    enter()
                    
    def bisection1(self, lin_func, inter1, inter2, tol):
        x1 = lin_func(inter1)
        x2 = lin_func(inter2)
        
        if x1 * x2 >= 0:
            return None
        loop = 1
        while True:
            midpoint = (inter1 + inter2) / 2
            x3 = lin_func(midpoint)
            
            if abs(x3) < tol :
                return midpoint

            if x1 * x3 < 0:
                inter2 = midpoint
            else:
                inter1 = midpoint
            loop += 1
            if abs(midpoint) - abs(x2) > tol:
                return midpoint
    
    def bisec_func(self, inter):
        x = inter
        return eval(self.function_input)

nonlin   = nonlinear()

def func7() : 
    print("WIP") # Work in progress

def func8() : 
    print()
    print("Simulation is a training method that demonstrates something")
    print("AS AN IMITATION form that is SIMILAR with real situation")
    print()
    print("Some simulation require RANDOM VARIABLE INPUT")
    print("to produce realistic OUTPUTS")
    print()
    enter()

    print()
    print("Random variable input can be generated with PSEUDO RANDOM NUMBER GENERATOR (PRNG)")
    print("One example of PRNG that can be used is Linear Congruential Generator (LCG)")
    print()
    print("The Formula of LGC is : ")
    print("Zi = (a * Zi-1 + c) * mod m")
    print("a = multiplier")
    print("c = increment")
    print("m = modulus")
    print()
    enter()

    print()
    print("Input start state ")
    start = int(input("> "))
    print("Input a (multiplier) ")
    a = int(input("> "))
    print("Input c (increment) ")
    c = int(input("> "))
    print("Input m (modulus) ")
    m = int(input("> "))
    print("Input iteration ")
    iteration = int(input("> "))
    print()
    data = {'iteration': [], 'calc': [], 'Zi': [], 'mod': [], 'Ui = Zi/m': []}
    
    print("a = ", a)
    print("c = ", c)
    print("m = ", m)
    print("start = ", start)
    print()

    print("Zi = (a * Zi-1 + c) * mod m")
    print("calc = (a * Zi-1 + c)")
    
    for i in range(iteration):
        calc = (a * start + c)
        mod = m
        zi = calc % m
        ui = zi / m
        
        data['iteration'].append(i+1)
        data['calc'].append(calc)
        data['mod'].append(m)
        data['Zi'].append(zi)
        data['Ui = Zi/m'].append(round(ui, 4))

        start = zi
    
    print(f"{'iteration':<12}{'calc':<10}{'mod':<7}{'Zi':<8}{'Ui = Zi/m':<12}")
    print("-" * 40)
    for i, calc, mod, zi, ui in zip(data['iteration'], data['calc'], data['mod'], data['Zi'],data['Ui = Zi/m']):
        print(f"{i:<12}{calc:<10}{mod:<7}{zi:<8}{ui:<12}")  
    print()
    enter()

    print()
    print("if you look carefully, around iteration", mod , "or", mod + 1, ", the result of Zi and Ui with iteration 1 are similar")
    print("The calc, Zi, Ui of iteration number 2 are similar with iteration" , mod + 1, "or" , mod + 2,)
    print("You might recognize other pattern as well")

    print()
    print("PNRG has a certain period where if it has reached that period")
    print("The resulting sequence of numbers will repeat from the beginning, starting the previous pattern")
    enter()

def func9() : 
    print("WIP") # Work in progress

def func10() : 
    print("WIP") # Work in progress

def markov() : 
    print("WIP") # Work in progress
        
def enter(): # biar gak perlu klik input() terus
    input("Press Enter to continue")

def main():
    while True:
        clear()
        print("================== Group 7 ===================")
        print("| 1. Rayhan Hendra Atmadja     075 - TI 23 C |")
        print("| 2. Adriano Emmanuel          082 - TI 23 C |")
        print("| 3. Cornelius Louis Nathan    085 - TI 23 C |")
        print("==============================================")
        print("\n========= Science Comp Implementation ========")
        print("1. Systems of Linear Equations") # Sistem Persamaan Linear
            # nanti terbagi menjadi 2, cara triangularisasi dan eliminasi gauss
        print("2. Matrix Practices and Operations")
            # ini bisa cuma ngeprint sifat" matrix, atau menampilkan penerapan matrix nya
        print("3. Invertible Matrik") # Invers Matrix
        print("4. LU Decomposition")
        print("5. Numeric Method of Linear Equations")  # Persamaan linear
            # sebenarnya aku bingung mau implemen ini, karena pptnya gk dikasih ama ibunya
        print("6. Numeric Method of Non Linear Equations")
            # ada 2 metode, metode biseksi (metode tertutup) dan newton rhapson (metode terbuka) (pilih salah satu aja gak seh?)
        print("7. Interpolation") # Interpolasi
            # ada 3, interpolasi linear, interpolasi kuadrat/non linear, polinom newton (pilih salah satu aja gak seh?)
        print("8. Basic Simulation")
            # sejauh ini, kalau simulasi..  paling mentok bisa nerapin Linear Congruential Generators (LCG) (PPT Slide 18)
            # LCG itu teori kalau misalkan hasil perhitungannya melebihi mod itu, maka hasil perhitungannya akan kembali seperti semula
        print("9. Iteration Theory") # Teori iterasi
            # iterasi jacobi dan gauss-seidler
        print("10. Monte-Carlo Simulation")
        print("11. Markov Chain")
        print("0. Exit\n")

        print("Choose from 1-11")
        try:
            choice = int(input("> "))
        except ValueError:
            print("\nInvalid input, choose from 1-12!")
            enter()
            continue

        if choice == 1:
            func1()
        elif choice == 2:
            func2()
        elif choice == 3:
            func3()
        elif choice == 4:
            func4()
        elif choice == 5:
            func5()
        elif choice == 6:
            nonlin.main_non_linear()
        elif choice == 7:
            func7()
        elif choice == 8:
            func8()
        elif choice == 9:
            func9()
        elif choice == 10:
            func10()
        elif choice == 11:
            markov()
        elif choice == 0:
            print("Thank you")
            break
        else:
            print("Invalid choice! Please choose from 1-11.")
            enter()
            continue


if __name__ == "__main__" : main() 