import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

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
            print("Non Linear Equation Calculator\n")
            print("Bisection Method is a numerical method")
            print("for finding the root of a nonlinear equation")
            print("Using the bisection method, we divide the interval")
            print("in half and then check if the function is increasing")
            print("or decreasing on each half of the interval\n")
            print("1. Quadratic Equation (Bisection Method)")
            print("0. Exit")
            choice = input("> ")
            if choice == '1':
                self.bisection()
            elif choice == '0':
                return
            else:
                print("Invalid choice! Please choose from 1-3.")
                enter()

    def bisection(self):
        clear()
        print("Bisection Method Calculator")
        print("Input Function (e.g x**2 - 4 / x***3 + 2)")
        self.function_input = input("> ")
        
        inter1 = int(input("Enter interval 1: "))
        inter2 = int(input("Enter interval 2: "))
        tol = float(input("Enter tolerance (e.g 0.00001): "))
        root = self.bisection1(self.bisec_func, inter1, inter2, tol)
            
        if root is not None:
            print(f'Approximate root is {root} after {self.iteration} iteration')
            enter()
        else:
            print('Invalid Interval')
            enter()
                
                    
    def bisection1(self, lin_func, inter1, inter2, tol):
        x1 = lin_func(inter1)
        x2 = lin_func(inter2)
        
        if x1 * x2 >= 0:
            return None
        self.iteration = 1
        while True:
            midpoint = (inter1 + inter2) / 2
            x3 = lin_func(midpoint)
            
            if abs(x3) < tol :
                return midpoint

            if x1 * x3 < 0:
                inter2 = midpoint
            else:
                inter1 = midpoint
            self.iteration += 1
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

class MonteCarlo():
    def monte_carlo_main(self): 
        while True:
            clear()
            print("Monte Carlo Simulation Calculator\n")
            print("Monte Carlo is a simulation technique used to approximate the probability of an event\n")
            print("by running a large number of simulations and then analyzing the results\n")
            print("This Program will implement Monte Carlo \
Simulation to calculate the probability of winning a roulette\n")
            print("1. Roulette Simulation")
            print("0. Exit")
            choice = input("> ")
            if choice == '1':
                while True:
                    clear()
                    print("Roulette Simulation Calculator")
                    print("Bet options: red, green, black, even, odd")
                    bet_option = input("Enter your bet option: ").strip().lower()
                    while bet_option not in {"red", "green", "black", "even", "odd"}:
                        print("Invalid option. Please choose from: red, green, black, even, odd.")
                        bet_option = input("Enter your bet option: ").strip().lower()

                    try:
                        iterations = int(input("Enter the number of iterations for the simulation: "))
                        if iterations <= 0:
                            raise ValueError("Number of iterations must be greater than 0.")
                    except ValueError as e:
                        print(f"Invalid input: {e}")
                        enter()
                    # Run simulation
                    win_percentage, tot_red, tot_black, tot_green, tot_even, \
                tot_odd = self.simulate_roulette(bet_option, iterations)

                    # Display results
                    print(f"Bet option\t\t: {bet_option.capitalize()}")
                    print(f"Number of iterations\t: {iterations}")
                    print(f"Total Red Spins\t\t: {tot_red}")
                    print(f"Total Black Spins\t: {tot_black}")
                    print(f"Total Green Spins\t: {tot_green}")
                    print(f"Total Even Spins\t: {tot_even}")
                    print(f"Total Odd Spins\t\t: {tot_odd}")
                    print(f"Win percentage\t\t: {win_percentage:.4f}%")
                    print("Do you want to run another simulation? (y/n)")
                    answer = input("> ")
                    if answer.lower() == "y":
                        continue
                    else:
                        return
            elif choice == '0':
                return
            else:
                print("Invalid choice! Please choose from 1-3.")
                enter()
            
            
    def simulate_roulette(self, bet_option, iterations):
        # Define roulette wheel segments
        red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
        black_numbers = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
        green_numbers = {0}

        wins = 0
        tot_red, tot_black, tot_green, tot_even, tot_odd = 0, 0, 0, 0, 0

        for _ in range(iterations):
            # Simulate a spin of the roulette wheel
            spin_result = random.randint(0, 36)

            # Check if the spin result matches the bet
            if bet_option == "red" and spin_result in red_numbers:
                wins += 1
            elif bet_option == "black" and spin_result in black_numbers:
                wins += 1
            elif bet_option == "green" and spin_result in green_numbers:
                wins += 1
            elif bet_option == "even" and spin_result != 0 and spin_result % 2 == 0:
                wins += 1
            elif bet_option == "odd" and spin_result % 2 == 1:
                wins += 1
                
                
            if spin_result in red_numbers: 
                tot_red += 1 
            elif spin_result in black_numbers: 
                tot_black += 1 
            elif spin_result in green_numbers: 
                tot_green += 1 
                
            if spin_result % 2 == 0: 
                tot_even += 1 
            elif spin_result % 2 == 1: 
                tot_odd += 1 

        win_percentage = (wins / iterations) * 100
        return win_percentage, tot_red, tot_black, tot_green, tot_even, tot_odd
            
monte = MonteCarlo()

def markov() : 
    while True:
        clear()
        print("Markov Chain Calculator\n")
        print("Markov Chain is a mathematical model that")
        print("describes the transitions between states in a system")
        print("")
        print("1. Markov Chain")
        print("0. Exit")
        choice = input("> ")
        if choice == '1':
            clear()
            print("Markov Chain Weather Prediction")
            print("Enter transition matrix (e.g 0.3):")
            print("Sunny -> Sunny: a")
            print("Sunny -> Rainy: b")
            print("Rainy -> Sunny: c") 
            print("Rainy -> Rainy: d")
            a = float(input("Enter a: "))
            b = float(input("Enter b: "))
            c = float(input("Enter c: "))
            d = float(input("Enter d: "))

            print("Enter starting probabilities (e.g 0.4):")
            e = float(input("Sunny : "))
            f = float(input("Rainy : "))
            iteration = int(input("Enter total iteration: "))
            transition_matrix = {
                'Sunny': {'Sunny': a, 'Rainy': b},
                'Rainy': {'Sunny': c, 'Rainy': d}
            }
            starting_probabilities = {'Sunny': e, 'Rainy': f}

            # Choose the starting state randomly based on starting probabilities
            current_state = random.choices(
                population=list(starting_probabilities.keys()),
                weights=list(starting_probabilities.values())
            )[0]

            # Generate a sequence of states using the transition matrix
            num_iterations = iteration
            states_sequence = [current_state]

            for _ in range(num_iterations):
                next_state = random.choices(
                    population=list(transition_matrix[current_state].keys()),
                    weights=list(transition_matrix[current_state].values())
                )[0]
                states_sequence.append(next_state)
                current_state = next_state

            print(states_sequence)
            input("Press Enter to continue...")
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")
        
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
            monte.monte_carlo_main()
        elif choice == 11:
            markov()
        elif choice == 0:
            print("Thank you")
            break
        else:
            print("Invalid choice! Please choose from 1-11.")
            enter()
            continue


if __name__ == "__main__" : 
    main() 