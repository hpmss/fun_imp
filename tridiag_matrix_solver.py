import numpy as np
import subprocess
import os
import tempfile
import time
import ast

"""
    Mx = r

    M is a square n x n matrix.
    x,r is a n x 1 vector.

    This uses Thomas's algorithm .
    Stage 1: Apply LU decomposition ( LUx = r,where Ux = p => Lp = r ) with down-ward sweep straight into Ux = p
    Stage 2: Solve Ux = p straight for x

    Input: create a file with name 'mat_input.txt' in the same folder as 'tridiag_matrix_solver.py' (os.getcwd())
    Example input (should only be 2 lines):
                M = [[2,3,0,0],[6,3,9,0],[0,2,5,2],[0,0,4,3]]
                r_trans = [[21,69,34,22]]
    Where "_trans" indicates transpose if you need to.
    Run the function example() in the main() function to see an example tri-diagonal matrix solving.
"""
PATH = os.getcwd() + "\\mat_input.txt"
N = None
M_mat = None
x_vec = None
r_vec = None
U = None
p = None

def matrix_input():
    M = None
    r = None
    try:
        with open(PATH,'r') as f:
            for line in f:
                inp = line.split("=")
                inp = [s.strip() for s in inp]
                if inp[0].startswith("M"):
                    if inp[0].endswith("trans"):
                        M = np.asarray(ast.literal_eval(inp[1])).T
                    else:
                        M = np.asarray(ast.literal_eval(inp[1]))
                else:
                    if inp[0].endswith("trans"):
                        r = np.asarray(ast.literal_eval(inp[1])).T
                    else:
                        r = np.asarray(ast.literal_eval(inp[1]))
    except FileNotFoundError as err:
        print("mat_input.txt not found.")
        print("mat_input.txt should have path: " + PATH)

    precondition(M,r)



def precondition(M,r):
    global M_mat,x_vec,r_vec,N
    assert len(M.shape) == 2, "M must be a 2-dimensional matrix"
    assert M.shape[0] == M.shape[1] , "M is not a square matrix"
    assert M.shape[0] == r.shape[0],"M,r are not in the same dimension"
    N = M.shape[0]
    num = len(np.where(M != 0)[0])
    assert num == N + 2*(N-1),"M is not a tri-diagonal matrix."
    M_mat = M
    r_vec = r
    x_vec = np.zeros((N,1))
    U_decomposition()

def U_decomposition():
    global U,p
    U = np.diag(np.diag(np.ones((N,N)))) # Contains ys
    p = np.zeros((N,1))
    for n in range(N):
        if n == 0:
            y_0 = M_mat[0,1] / M_mat[0,0]
            p_0 = r_vec[0,0] / M_mat[0,0]
            U[0,1] = y_0
            p[0,0] = p_0
        elif n == N - 1:
            p_n = (r_vec[n,0] - M_mat[n,n-1] * p[n-1,0]) / (M_mat[n,n] - M_mat[n,n-1] * U[n-1,n])
            p[n,0] = p_n
        else:
            y_n = M_mat[n,n+1] / (M_mat[n,n] - M_mat[n,n-1] * U[n-1,n])
            p_n = (r_vec[n,0] - M_mat[n,n-1] * p[n-1,0]) / (M_mat[n,n] - M_mat[n,n-1] * U[n-1,n])
            U[n,n+1] = y_n
            p[n,0] = p_n
    print("Decomposed U matrix: \n",U)
    print("Decomposed p vector: \n",p)
    solve_for_x_vec()

def solve_for_x_vec():
    global x_vec
    x_vec[N-1,0] = p[N-1,0]
    # _n means going backwards start at N-2 towards 0 instead of starting at 0
    for _n in range(N-2,-1,-1):
        x_n = p[_n] - U[_n,_n + 1] * x_vec[_n + 1,0]
        x_vec[_n,0] = x_n
    print("Result for x: \n" , x_vec)
    print("Re-checking result:" ,"\n---Original r vector--- \n" , r_vec , "\n---Calculated r vector--- \n",np.dot(M_mat,x_vec))

def example():
    N = 5
    M = np.asarray([[2,3,0,0],[6,3,9,0],[0,2,5,2],[0,0,4,3]])
    r = np.asarray([[21,69,34,22]]).T
    precondition(M,r)

def main():
    matrix_input()

main()
