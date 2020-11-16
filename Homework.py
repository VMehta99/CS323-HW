import numpy

# FOR FRACTION PRINTING:
numpy.set_printoptions(
 linewidth=180
);
# NUMBER 3 WORK:
# LU DECOMP ALGO:
def LUDecomp(table):
    rows, columns = numpy.shape(table)
    L = numpy.zeros((rows, columns))
    U = numpy.zeros((rows, columns))
    for i in range(columns):
        for j in range(i):
            sum = 0
            for k in range(j):
                sum += L[i][k] * U[k][j]
            L[i][j] = (table[i][j] - sum) / U[j][j]
        L[i][i] = 1
        for j in range(i, columns):
            sum1 = 0
            for k in range(i):
                sum1 += L[i][k] * U[k][j]
            U[i][j] = table[i][j] - sum1
    return L, U

def forward_sub(L, b):
    n = L.shape[0]
    x = numpy.zeros(n)
    for i in range(n):
        tmp = b[i]
        for j in range(i-1):
            tmp -= L[i,j] * x[j]
        x[i] = tmp / L[i,i]
    return x

def backward_sub(U, b):
    n = U.shape[0]
    x = numpy.zeros(n)
    for i in range(n-1, -1, -1):
        tmp = b[i]
        for j in range(i+1, n):
            tmp -= U[i,j] * x[j]
        x[i] = tmp / U[i,i]
    return x


def lu_solve(L, U, b):
    y = forward_sub(L, b)
    x = backward_sub(U, y)
    return x

def Gaussian_Elimination(A,b):
    n =  len(A)
    for pivot_row in range(0,n-1):
        if A[pivot_row,pivot_row] == 0:
            return
        for row in range(pivot_row+1, n):
            A[row][pivot_row] = A[row][pivot_row]/A[pivot_row][pivot_row]
            for col in range(pivot_row + 1, n):
                A[row][col] = A[row][col] - A[row][pivot_row]*A[pivot_row][col]
            b[row] = b[row] - A[row][pivot_row]*b[pivot_row]
  
    x = numpy.zeros(n)
    k = n-1
    x[k] = b[k]/A[k,k]
    while k >= 0:
        x[k] = (b[k] - numpy.dot(A[k,k+1:],x[k+1:]))/A[k,k]
        k = k-1
    return x
# MAIN METHOD:s
if __name__ == "__main__":
    matrix = numpy.array([
        [21 ,32 ,14 ,8 ,6 ,9 ,11 ,3 ,5], 
        [17 ,2 ,8 ,14 ,55 ,23 ,19 ,1 ,6], 
        [41 ,23 ,13 ,5 ,11 ,22 ,26 ,7 ,9],
        [12 ,11 ,5 ,8 ,3 ,15 ,7 ,25 ,19],
        [14 ,7 ,3 ,5 ,11 ,23 ,8 ,7 ,9],
        [2 ,8 ,5 ,7 ,1 ,13 ,23 ,11 ,17],
        [11 ,7 ,9 ,5 ,3 ,8 ,26 ,13 ,17],
        [23 ,1 ,5 ,19 ,11 ,7 ,9 ,4 ,16],
        [31 ,5 ,12 ,7 ,13 ,17 ,24 ,3 ,11]
    ])
    L, U = LUDecomp(matrix)
    b = [2,5,7,1,6,9,4,8,3];
    # PRINTING ANSERS:
    print("\n\n")
    print("Num Analysis HW 3| number 3")
    print("Vedant Mehta : VM439")
    print("\n Original Matrix:")
    print(numpy.matrix(matrix))

    print("\n\n")
    print("L=")
    print(numpy.matrix(L))
    print("\n")
    print("U=")
    print(numpy.matrix(U))
    print("\n\n")

    print("\n\n")
    print("Answer:")

    x = lu_solve(L,U,b)

    for num in range(len(x)):
        print("x" +str(num+1)+ "= " + str(x[num]))




# NUMBER 4 WORK:
    matrix = numpy.array([
        [21.0    ,67.0    ,88.0    ,73.0],
        [76.0    ,63.0 ,7.0 ,20.0],
        [0.0, 85.0    ,56.0    ,54.0],
        [19.3   ,43.0    ,30.2    ,29.4]
    ])
    b = numpy.array([141.0, 109.0, 218.0, 93.7]);


# PRINTING ANSERS:
    print("\n\n")
    print("Num Analysis HW 3 | number 4 ")
    print("Vedant Mehta : VM439")
    
# PART A:
    print("\n\n")
    print("PART A:")
    print("\n Original Matrix:")
    print(numpy.matrix(matrix))
    print("\n\n")
    
    print("x vector: (single precision):")

    x = Gaussian_Elimination(numpy.float32(matrix),numpy.float32(b))

    for num in range(len(x)):
        print("x" +str(num+1)+ "= " + str(x[num]))


# PART B:
    matrix = numpy.array([
        [21.0    ,67.0    ,88.0    ,73.0],
        [76.0    ,63.0 ,7.0 ,20.0],
        [0.0, 85.0    ,56.0    ,54.0],
        [19.3   ,43.0    ,30.2    ,29.4]
    ])
    b = numpy.array([141.0, 109.0, 218.0, 93.7]);
    print("\n\n")
    print("PART B:")

    print("\n Original Matrix:")
    print(numpy.matrix(matrix))
    print("\n\n")

    print("x vector: (double precision)")
    x = Gaussian_Elimination(numpy.float64(matrix),numpy.float64(b))

    for num in range(len(x)):
        print("x" +str(num+1)+ "= " + str(x[num]))

    print("\n\n")
    print("r vector: (single precision)")
    r = numpy.float32(b - numpy.dot(numpy.float64(matrix), x))

    for num in range(len(r)):
        print("r" +str(num+1)+ "= " + str(r[num]))


# PART C:
    b = numpy.array([141.0, 109.0, 218.0, 93.7]);

    print("\n\n")
    print("PART C:")
    print("\n\n")
    
    print("z vector:")
    # Ar = z
    z = Gaussian_Elimination(numpy.float64(matrix),numpy.array(r))

    for num in range(len(z)):
        print("z" +str(num+1)+ "= " + str(z[num]))

    print("\n\n")
    print("improved solution (x+z): ")
    x = numpy.add(z,x);

    for num in range(len(x)):
        print("x" +str(num+1)+ "= " + str(x[num]))

#PART D:
    print("\n\n")
    print("PART D:")

# x1= -1.0000000000000984
# x2= 1.999999999999556
# x3= -3.0000000000015627
# x4= 4.00000000000232

    for itter in range(7):
        print("[START OF ROUND " + str(itter+1) + "]: ")
        matrix = numpy.array([
            [21.0    ,67.0    ,88.0    ,73.0],
            [76.0    ,63.0 ,7.0 ,20.0],
            [0.0, 85.0    ,56.0    ,54.0],
            [19.3   ,43.0    ,30.2    ,29.4]
        ])
        b = numpy.array([141.0, 109.0, 218.0, 93.7]);

        print("\n")
        print("r vector: (single precision)")
        r = numpy.float32(b - numpy.dot(numpy.float64(matrix), x))

        for num in range(len(r)):
            print("r" +str(num+1)+ "= " + str(r[num]))

        b = numpy.array([141.0, 109.0, 218.0, 93.7]);
        
        print("\n\n")
        print("z vector:")
        # Ar = z
        z = Gaussian_Elimination(numpy.float64(matrix),numpy.array(r))

        for num in range(len(z)):
            print("z" +str(num+1)+ "= " + str(z[num]))

        print("\n\n")
        print("improved solution (x+z): ")
        x = numpy.add(z,x);

        for num in range(len(x)):
            print("x" +str(num+1)+ "= " + str(x[num]))
        
        print("\n\n[END OF ROUND " + str(itter+1) + "]\n\n")
    else:
        print("[Complete] No further improvments observed. \n")
        print("PLEASE NOTE: ALL ANSWERS TO QUESTIONS 4 AND 3 ARE OUTPUTED ABOVE.\n")
        print("Num Analysis HW 3 | number 4 | number 3 ")
        print("Vedant Mehta : VM439\n")
