import numpy
import fractions
from pandas import *

#FOR FRACTION PRINTING:
numpy.set_printoptions(
    formatter={
        'all':lambda x: str(fractions.Fraction(x).limit_denominator())
    },linewidth=80
);


# JUST FOR FORMATING AND DESIGN:
class color:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

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


# MAIN METHOD:
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

    # PRINTING ANSERS:
    print("\n\n")
    print(color.BOLD + "Num Analysis HW 3" + color.END)
    print(color.UNDERLINE +"Vedant Mehta : VM439" + color.END)
    print("\n Original Matrix:")
    print(numpy.matrix(matrix))

    print("\n\n")
    print(color.BOLD + 'LU DECOMPOSITION WITH ELEMENTS AS FRACTIONS' + color.END)
    print("L=")
    print(numpy.matrix(L))
    print("\n")
    print("U=")
    print(numpy.matrix(U))
    print("\n\n")

    print(color.BOLD + "LU DECOMPOSITION WITH ELEMENTS AS DECIMAL:" + color.END)
    print("L= (printed as Data Frame)")
    print(DataFrame(L))
    print("\n")
    print("U= (printed as Data Frame)")
    print(DataFrame(U))