import numpy as np
import matplotlib.pyplot as plt

def inverse_matrix(matrix_el):
    A = np.array(matrix_el, dtype=np.float64)

    n = A.shape[0]

    print("n: ", n)

    if n != A.shape[1]:
        raise ValueError("Matrix must be square")
    
    augmented = np.column_stack((A, np.eye(n)))

    for i in range(n):
        max_element = abs(augmented[i:, i]).argmax() + i

        if max_element != i:
            augmented[[i, max_element]] = augmented[[max_element, i]]
            
        if np.isclose(augmented[i, i], 0):
            raise ValueError("Matrix is not invertible")
        
        augmented[i] = augmented[i] / augmented[i, i]

        for j in range(n):
            if i != j:
                augmented[j] -= augmented[i] * augmented[j, i]

    return augmented[:, n:]

def lu_decomposition(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    U = [[0]*n for _ in range(n)]
    print("L: ", L)
    print("U: ", U)

    for i in range(n):
        for j in range(i, n): 
            U[i][j] = A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                L[j][i] = (A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]

    return L, U

def gauss_seidel(A, b, x0=None, tolerance=1e-10, max_iterations=1000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(A)

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.array(x0, dtype=float)

    for iteration in range(max_iterations):
        x_old = x.copy()

        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            
            x[i] = (b[i] - sigma) / A[i][i]
        
        if np.linalg.norm(x - x_old) < tolerance:
            return x, iteration + 1
        
    raise ValueError(f"Method did not converge after {max_iterations} iterations")

def plot_linear_equation_point_slope(point, slope, x_range=(-10, 10)):
    x0, y0 = point
    
    x = np.linspace(x_range[0], x_range[1], 100)
    
    y = slope * (x - x0) + y0
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Line: y - {y0} = {slope}(x - {x0})')
    
    plt.plot(x0, y0, 'ro', label='Given Point')

    plt.title('Point-Slope Form Linear Equation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.show()

def print_matrix(matrix_el):
    for row in matrix_el:
        print(" ".join(f"{x:8.4f}" for x in row))

def main():
    # INVERS AND LU DEC
    A = [
        [4, 7, 2],
        [2, 6, 1],
        [1, 5, 3]
    ]

    # inverse = inverse_matrix(A)

    # print("Original matrix:")
    # print_matrix(A)

    # print("Inverse matrix:")
    # print_matrix(inverse)

    # print("Inverse matrix with numpy:")
    # print_matrix(np.linalg.inv(A))

    # print("LU decomposition:")
    # L, U = lu_decomposition(A)
    # print("L:")
    # print_matrix(L)
    # print("U:")
    # print_matrix(U)

    # GAUSS SEIDEL
    # A = [
    #     [4, -1, 0],
    #     [-1, 4, -1],
    #     [0, -1, 4]
    # ]
    # b = [15, 10, 10]
    
    # try:
    #     solution, iterations = gauss_seidel(A, b)
        
    #     print("Solution:", solution)
    #     print("Iterations:", iterations)
        
    #     print("\nVerification:")
    #     print("A * x =", np.dot(A, solution))
    #     print("b     =", b)
    
    # except ValueError as e:
    #     print(e)

    plot_linear_equation_point_slope(point=(0, 5), slope=-3/4)


if __name__ == "__main__":
    main()