import numpy as np


def coefficientsOfPayMatrix(A):
    temp = A[0][0]
    for i in range(len(A)):
        temp = A[i][0]
        for j in range(len(A[i])):
            if (j == len(A[i]) - 1):
                A[i][j] = temp
            if (A[i][j] < temp):
                temp = A[i][j]

    for i in range(len(A)):

        temp = A[0][i - 1]
        for j in range(len(A[i])):
            if(len(A)<len(A[i])) :
                if (j == len(A) - 1):
                    A[j][i] = temp
                    break
                if (A[j][i] > temp):
                    temp = A[j][i]
            else:
                if (j == len(A)-1):
                    A[j][i-1] = temp

                if (A[j][i-1] > temp):
                    temp = A[j][i-1]

def findLowerPriceOfGame(A):
    temp = A[len(A)-1][0]
    for i in range(len(A)):
        if (A[len(A)-1][i] < temp):
            temp = A[len(A)-1][i]
    return temp


def findUpperPriceOfGame(A):
    temp = A[len(A) - 1][0]
    for i in range(len(A)):
        for j in range(len(A[i])-1):
            if (A[len(A) - 1][j] < temp):
                temp = A[len(A) - 1][j]
            if(j==len(A)):
                break
        break
    return temp


def findoptimalCleanStrategy(A, matrixALowerPriceOfGame):
    for i in range(len(A)-1):
        for j in range(len(A[i])-1):
            if (len(A) < len(A[i])):
                if (A[i][j] == matrixALowerPriceOfGame and A[len(A)-1][j] == matrixALowerPriceOfGame and A[i][len(A[i])-1] == matrixALowerPriceOfGame):
                    print("Стратегия A", i + 1, "является чистой оптимальной для компании A,статегия B", j + 1,
                          "является такой для B")

                    if(len(Demand_On_Thousand_Products)==len(A)):
                        print("Реализовано будет:", Demand_On_Thousand_Products[i][i], "тысяч единиц продукции")
            else:
                if (A[i][j] == matrixALowerPriceOfGame and A[len(A)-1][j] == matrixALowerPriceOfGame and A[i][len(A[i])-1] == matrixALowerPriceOfGame):

                    print("Стратегия A", i + 1, "является чистой оптимальной для компании A,статегия B", j + 1,
                          "является такой для B")
                    print("Реализовано будет:", Demand_On_Thousand_Products[i][i], "тысяч единиц продукции")


def conclusion(A,matrixALowerPriceOfGame,matrixAUpperPriceOfGame):
    if (matrixALowerPriceOfGame == matrixAUpperPriceOfGame):
        print("Существует седловая точка")
        findoptimalCleanStrategy(A, matrixALowerPriceOfGame)
        if (matrixALowerPriceOfGame > 0):
            print("Предприятие A выиграло на ", matrixALowerPriceOfGame, "единиц")
        else:
            print("Предприятие A проиграло на ", -matrixALowerPriceOfGame, "единиц")
    else:
        print("Седловая точка отсутствует")
# task1
Price = [12, 8, 5]
Cost_Price_CompanyOne = [6, 3, 2]
Cost_Price_CompanySecond = [8, 2, 1]
# function of demand
# Y=6-0.5*X
# Prices
Price_CompanyOne = [[10, 10, 10],
                    [6, 6, 6],
                    [2, 2, 2]]
Price_CompanySecond = [[10, 6, 2],
                       [10, 6, 2],
                       [10, 6, 2]]
#
Average_Price_Companies = [[10, 8, 6], [8, 6, 4], [6, 4, 2]]
# demand on 1 thousand products
Demand_On_Thousand_Products = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
# partition of production that population buy
Partition_Of_Production_That_Population_Buy = [[0.57, 0.42, 0.25], [0.8, 0.4, 0.3], [0.92, 0.85, 0.72]]
n = 4
m = 4
k = 3

A = [[0.0] * m for i in range(n)]
for i in range(k):
    for j in range(k):
        A[i][j] = Demand_On_Thousand_Products[i][j] * (
                    Partition_Of_Production_That_Population_Buy[i][j] * (Price[i] - Cost_Price_CompanyOne[i]) - (
                        1 - Partition_Of_Production_That_Population_Buy[i][j]) * (
                                Price[j] - Cost_Price_CompanySecond[j]))

print(A)

coefficientsOfPayMatrix(A)

matrixALowerPriceOfGame = findLowerPriceOfGame(A)
print("Нижняя цена игры:", matrixALowerPriceOfGame)

matrixAUpperPriceOfGame = findUpperPriceOfGame(A)
print("Верхняя цена игры:", matrixAUpperPriceOfGame)

conclusion(A,matrixALowerPriceOfGame,matrixAUpperPriceOfGame)

for x in A:
    for y in x:
        print(y, end=" ")
    print()
# Task2
AB =[[-2, 0, 3, -1, 1, 0],
    [-1, 5, -2, -2, -1, 0],
    [-3, -4, 0, -2, -2, 0],
    [3, 5, 3, 3, 1, 0],
    [0, 0, 0, 0, 0, 0]]
AC = np.array([[4, -4, -1, 0, 0],
              [7, 6, 2, 6, 0],
              [5, 4, -6, 0, 0],
              [0, 0, 0, 0, 0]])
AD = np.array([[-6, 5, -3, 2 , 0],
              [-3, 4, 3, -6, 0],
              [-3, 7, 5, -3, 0],
              [-3, -1, -4, 8, 0],
              [-6, 1, -6, 5, 0],
              [0, 0, 0, 0, 0]])

#1
print("Part1")
coefficientsOfPayMatrix(AB)
for x in AB:
    for y in x:
        print(y, end=" ")
    print()

matrixABLowerPriceOfGame = findLowerPriceOfGame(AB)
print("Нижняя цена игры:", matrixABLowerPriceOfGame)

matrixABUpperPriceOfGame = findUpperPriceOfGame(AB)
print("Верхняя цена игры:", matrixABUpperPriceOfGame)


conclusion(AB,matrixABLowerPriceOfGame,matrixABUpperPriceOfGame)

for x in AB:
    for y in x:
        print(y, end=" ")
    print()
#2
print("Part2")
coefficientsOfPayMatrix(AC)
for x in AC:
    for y in x:
        print(y, end=" ")
    print()


matrixACLowerPriceOfGame = findLowerPriceOfGame(AC)
print("Нижняя цена игры:", matrixACLowerPriceOfGame)

matrixACUpperPriceOfGame = findUpperPriceOfGame(AC)
print("Верхняя цена игры:", matrixACUpperPriceOfGame)


conclusion(AC,matrixACLowerPriceOfGame,matrixACUpperPriceOfGame)

for x in AC:
    for y in x:
        print(y, end=" ")
    print()
#3
print("Part3")

temp = A[0][0]
for i in range(len(AD)):
    temp = AD[i][0]
    for j in range(len(AD[i])):
        if (j == len(AD[i]) - 1):
            AD[i][j] = temp
        if (AD[i][j] < temp):
            temp = AD[i][j]

for i in range(4):
    temp = AD[0][i]
    for j in range(6):
            if (j == 5):
                AD[j][i] = temp
            if (AD[j][i] > temp):
                temp = AD[j][i]
print(AD)

temp = AD[5][0]
for i in range(len(AD)-1):
    if (AD[len(AD)-1][i] < temp):
        temp = AD[len(AD) - 1][i]

matrixADLowerPriceOfGame= temp
print("Нижняя цена игры:", matrixADLowerPriceOfGame)

matrixADUpperPriceOfGame = findUpperPriceOfGame(AD)
print("Верхняя цена игры:", matrixADUpperPriceOfGame)


conclusion(AD,matrixADLowerPriceOfGame,matrixADUpperPriceOfGame)

for x in AD:
    for y in x:
        print(y, end=" ")
    print()